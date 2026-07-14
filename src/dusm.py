#!/usr/bin/python3
"""D-U-S-M: Domain Unified Symbolic discovery, fully distributed via Ray Actors."""
import logging
import time
from itertools import product
from math import ceil

import numpy as np
import ray

from sample_multidim import MultidimSample
from query_multidim import MultidimQuery
from discovery_shared import (
    ht_descriptive_queries,
    _merge_domain_queries,
    adapted_querystring,
    pos2query,
    add_vertex2tree,
)
from ducm import discover_ducm
from profiling_helpers import DriverProfiler, WorkerProfiler, log_profiling_summary

LOG_FORMAT = '| %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)


# ---------------------------------------------------------------------------
# Ray Actor + sample rebuilding (duplicated from ducm.py)
# ---------------------------------------------------------------------------

@ray.remote
class ChunkWorker:
    """Persistent Ray Actor that owns one chunk of the sample.

    The chunk, patternset, dict_iter, and parent_dict live here permanently.
    Only the query object crosses the wire per match call.
    """

    def __init__(self, chunk, patternset, chunk_id, supp, profile: bool = False):
        self.chunk = chunk
        self.patternset = patternset
        self.chunk_id = chunk_id
        self.supp = supp
        self.dict_iter = {}
        self.parent_dict = {}
        self._profiler = WorkerProfiler(profile)

    def match(self, query):
        def _do_match():
            result = query.match_sample_distributed(
                sample=self.chunk,
                supp=self.supp,
                dict_iter=self.dict_iter,
                patternset=self.patternset,
                parent_dict=self.parent_dict,
                chunk_id=self.chunk_id,
            )
            matched, self.dict_iter, _, new_parent_dict, matched_traces = result
            self.parent_dict.update(new_parent_dict)
            return matched, matched_traces

        return self._profiler.record(_do_match)

    def get_profile_stats(self):
        """Return timing summary + a pstats text dump for this actor's chunk."""
        return self._profiler.stats(self.chunk_id, self.chunk._sample_size)


def rebuild_sample_from_array(original_sample) -> MultidimSample:
    sample = MultidimSample()
    sample.set_sample(list(original_sample))
    sample.calc_sample_typeset(calculate_all=True)
    return sample


def _build_chunk_workers(sample, all_patternset, supp, distributions=4, profile: bool = False):
    """Split `sample` into chunks and spin up one persistent ChunkWorker per chunk.

    Returns (workers, chunk_offsets).
    """
    chunks = np.array_split(sample._sample, distributions)
    chunks = [rebuild_sample_from_array(chunk) for chunk in chunks]

    workers = []
    chunk_offsets = []
    offset = 0
    for i, chunk in enumerate(chunks):
        chunk_id = i + 1
        chunk_patternset = {
            domain: {local_t: all_patternset[domain][local_t + offset]
                     for local_t in range(chunk._sample_size)}
            for domain in all_patternset
        }
        workers.append(ChunkWorker.remote(chunk, chunk_patternset, chunk_id, supp, profile))
        chunk_offsets.append(offset)
        offset += chunk._sample_size

    return workers, chunk_offsets


def _match_distributed(query, workers, chunk_offsets, driver_profiler: DriverProfiler = None):
    """Dispatch `query` to all persistent chunk workers and merge the results.

    Collects results as they arrive and stops early as soon as one worker
    reports non-matching.

    If `driver_profiler` is given (an enabled DriverProfiler), dispatch and
    collection timing plus early-stop counts are recorded on it. Passing
    None (or a disabled DriverProfiler) costs nothing extra.
    """
    dp = driver_profiler if driver_profiler is not None else DriverProfiler(enabled=False)

    query.set_query_matchtest('smarter')
    with dp.time_dispatch():
        futures = [worker.match.remote(query) for worker in workers]
    future_to_offset = {f: off for f, off in zip(futures, chunk_offsets)}

    remaining = list(futures)
    matched_traces = []
    matching = True

    with dp.time_collection():
        while remaining:
            ready, remaining = ray.wait(remaining, num_returns=1)
            matched, chunk_matched_traces = ray.get(ready[0])
            if not matched:
                matching = False
                dp.record_early_stop()
                break  # one non-match is enough; remaining actors finish in background
            off = future_to_offset[ready[0]]
            matched_traces.extend(t + off for t in chunk_matched_traces)

    return matching, matched_traces


def _shutdown_workers(workers):
    for w in workers:
        ray.kill(w)


# ---------------------------------------------------------------------------
# Outer loop (duplicated from discovery_shared._domain_separated_discovery),
# with the merge-phase matching step distributed via Ray Actors.
# ---------------------------------------------------------------------------

def _domain_separated_discovery_distributed(sample, supp, matchtest, max_query_length, per_domain_fn,
                                              profile: bool = False) -> dict:
    """Outer discovery loop for D-U-S-M.

    Each domain is discovered independently via per_domain_fn (D-U-C-M,
    itself distributed), then results are merged across domains. The
    merge-phase matching step, which matches candidate merged queries
    against the full sample, is also distributed across persistent
    ChunkWorker actors.

    Args:
        sample: MultidimSample instance.
        supp: Support threshold in [0, 1].
        matchtest: 'smarter' for D-U-S-M.
        max_query_length: Maximum query length.
        per_domain_fn: Callable(domain_sample, supp, max_query_length,
                       domain_patternset, profile) -> result_dict.
        profile: If True, collect profiling data for both the per-domain
            discovery phase (one entry per domain, taken from each
            per_domain_fn call's own 'profiling' key) and the merge phase
            (its own dedicated DriverProfiler + worker pool), and attach
            everything to the result dict under 'profiling'.
    """
    sample_set = sample._sample
    sample_size = len(sample_set)
    domain_cnt = sample_set[0].split(' ')[0].count(';')

    if supp == 1.0:
        _, min_trace_length = sample.get_sample_min_trace()
        max_query_length = min(max_query_length, min_trace_length)
    query_dict = {}
    dict_iter = {}
    parent_dict = {}
    dim_sample_dict = sample.get_dim_sample_dict()

    domain_query_list = []
    all_dictionary = {}
    query_list = {}
    gen_event = ';' * domain_cnt
    gen_event_list = [i for i in gen_event]
    all_patternset = {}
    per_domain_profiling = []

    t_discovery_start = time.perf_counter()

    for domain, domain_sample in dim_sample_dict.items():
        vert_db = domain_sample.get_att_vertical_sequence_database()
        all_patternset[domain] = {trace_id: set() for trace_id in range(sample_size)}
        for key, value in vert_db.items():
            for letter, pos_dict in value.items():
                for trace_id, positions in pos_dict.items():
                    if len(positions) >= 2:
                        all_patternset[domain][trace_id].add(letter)

        domain_patternset = {domain: all_patternset[domain]}
        result_dict = per_domain_fn(domain_sample, supp, max_query_length, domain_patternset, profile)

        if profile:
            per_domain_profiling.append(result_dict.get('profiling'))

        instance_dictionary = {}
        dom_sample_size = domain_sample._sample_size
        trace_list = list(range(dom_sample_size))

        for result_query in result_dict['matching_dict'].values():
            if supp < 1:
                querystring = result_query._query_string
                if result_dict['dict_iter'] is None:
                    # distributed (ducm) result: no centralized dict_iter,
                    # matched traces were already collected on the query object
                    trace_list = list(result_query._query_matched_traces)
                else:
                    trace_list = []
                    for trace in range(dom_sample_size):
                        if '$' in querystring:
                            if trace in result_dict['dict_iter'][querystring]:
                                trace_list.append(trace)
                        else:
                            if trace in result_dict['dict_iter'][querystring]:
                                if result_dict['dict_iter'][querystring][trace] != -1:
                                    trace_list.append(trace)
            instance_dictionary = result_query.query_pos_dict(
                vert_db, domain_sample, instance_dictionary, trace_list=trace_list)

        pos_dict = instance_dictionary
        domain_queryset = set(result_dict['queryset'])
        query_list.update(result_dict['matching_dict'].items())
        if result_dict['matching_dict'] and result_dict['dict_iter'] is not None:
            dict_iter.update(result_dict['dict_iter'])
        if 'parent_dict' in result_dict:
            parent_dict.update(result_dict['parent_dict'])

        for domain_query in domain_queryset:
            if gen_event == domain_query or not domain_query:
                continue
            all_dictionary[domain_query] = {
                'trace_instances': pos_dict[domain_query],
                'occurences': len(pos_dict[domain_query])
            }
        domain_query_list.append(list(domain_queryset))

        if domain == 0:
            mixed_query_tree = result_dict['query_tree']
        else:
            domain_tree = result_dict['query_tree']
            root_vertex_domain = domain_tree.get_root()
            root_vertex = mixed_query_tree.get_root()
            child_vertices = root_vertex_domain.child_vertices
            for child_vertex in child_vertices:
                mixed_query_tree.insert_query_string(
                    root_vertex, child_vertex.query_string,
                    query=child_vertex.query, search_for_parents=False)

    t_discovery_end = time.perf_counter()             # <-- END phase 1
    #print(f"Per-domain discovery took {t_discovery_end - t_discovery_start:.4f}s")

    descriptive_query_list = set()
    seen = set()
    non_empty_keys = {i: dom_list for i, dom_list in enumerate(domain_query_list)}
    empty_domains = set()
    for dom, dom_list in non_empty_keys.items():
        if matchtest == 'smarter':
            dom_list.append('')
        elif len(dom_list) == 1:
            empty_domains.add(dom)
        non_empty_keys[dom] = dom_list
    for dom in empty_domains:
        del non_empty_keys[dom]

    query_pairs_v2 = sorted(product(*non_empty_keys.values()))
    matchings = {}
    pair_dict_matching = {}
    pair_dict_non_matching = {}
    non_matching = set()
    seen = set()

    t_merge_start = time.perf_counter()

    # Persistent worker pool for merge-phase matching, built once against
    # the full (all-domain) sample and reused for every candidate query.
    merge_workers, merge_chunk_offsets = _build_chunk_workers(sample, all_patternset, supp, profile=profile)
    merge_dp = DriverProfiler(enabled=profile)
    merge_dp.start(logger=LOGGER)
    merge_querycount = 0

    for pair in query_pairs_v2:
        pair_dict_matching[pair] = []
        pair_dict_non_matching[pair] = []
        parent_tuples = []
        parent_tuple_count = []
        parent_tuple_doms = []
        for idx, domain, dom_tuple in zip(range(len(pair)), non_empty_keys.keys(), pair):
            if dom_tuple:
                if dom_tuple in parent_dict:
                    parent_string = parent_dict[dom_tuple]._query_string
                    if parent_string == gen_event:
                        parent_string = ''
                else:
                    query = MultidimQuery()
                    query.set_query_string(dom_tuple, recalculate_attributes=False)
                    parent = query._parent()
                    parent_string = parent._query_string
            else:
                parent_string = ''
            parent_tuple_list = list(pair)
            parent_tuple_list[idx] = parent_string
            parent_tuple = tuple(parent_tuple_list)

            if parent_tuple != pair:
                parent_tuple_count.append(len([i for i in parent_tuple if i]))
                parent_tuples.append(parent_tuple)
                parent_tuple_doms.append([i for i, element in enumerate(parent_tuple) if element])

        if '' in pair or gen_event in pair:
            domain_indeces_dict = {dom_idx: pair[i] for i, dom_idx in
                                   enumerate(non_empty_keys.keys())
                                   if pair[i] not in ['', gen_event]}
            domain_indeces = list(non_empty_keys.keys())
        else:
            domain_indeces = list(non_empty_keys.keys())
            domain_indeces_dict = {domain_idx: domain_string
                                   for domain_idx, domain_string in zip(domain_indeces, pair)}
        if len(domain_indeces_dict) <= 1:
            if domain_indeces_dict:
                querystring = list(domain_indeces_dict.values())[0]
                pair_dict_matching[pair].append(querystring)
                descriptive_query_list.add(querystring)
                if querystring in parent_dict:
                    parentstring = parent_dict[querystring]._query_string
                else:
                    parent = query._parent()
                    parentstring = parent._query_string
                if parentstring == gen_event and matchtest == 'smarter':
                    parentstring = ''
                parent_vertex = mixed_query_tree.find_vertex(parentstring)
                if not mixed_query_tree.find_vertex(querystring):
                    mixed_query_tree.insert_query_string(
                        parent_vertex, querystring,
                        query=query_list[querystring], search_for_parents=False)

        else:
            poss_queries_v2 = _merge_domain_queries(domain_indeces_dict, all_dictionary, max_query_length, supp)
            poss_query_list = set()
            if poss_queries_v2:
                adapted_qs_dict = adapted_querystring(domain_indeces_dict, query_list)
                for pair2 in poss_queries_v2:
                    if pair2 and domain_indeces_dict:
                        querystring = pos2query(domain_indeces_dict, pair2, adapted_qs_dict, max_query_length)
                        if querystring == gen_event or not querystring:
                            continue
                        else:
                            poss_query_list.add(querystring)

            matching_queryset = set()
            if poss_query_list:
                for querystring in poss_query_list:
                    poss_query = MultidimQuery()
                    poss_query.set_query_string(querystring, recalculate_attributes=False)
                    parent = poss_query._parent()
                    querystring = poss_query._query_string
                    poss_query.set_query_matchtest('smarter')
                    parent_dict[querystring] = parent
                    if parent._query_string not in non_matching:
                        seen.add(querystring)
                        merge_querycount += 1
                        match, _matched_traces = _match_distributed(
                            poss_query, merge_workers, merge_chunk_offsets, driver_profiler=merge_dp)
                        matchings[querystring] = match
                        if match and len(querystring.split()) <= max_query_length:
                            query_dict[querystring] = poss_query
                            pair_dict_matching[pair].append(querystring)
                            matching_queryset.add(querystring)
                            mixed_query_tree = add_vertex2tree(
                                poss_query, parent, mixed_query_tree,
                                gen_event, parent_dict, matchtest)
                        else:
                            pair_dict_non_matching[pair].append(querystring)
                            non_matching.add(querystring)
            descriptive_query_list.update(matching_queryset)

    # gather per-worker profiling stats before shutting the merge workers down
    merge_worker_stats = ray.get([w.get_profile_stats.remote() for w in merge_workers]) if profile else None

    _shutdown_workers(merge_workers)

    t_merge_end = time.perf_counter()  # <-- END phase 2
    #print(f"Merging phase took {t_merge_end - t_merge_start:.4f}s")

    result_dict = {}
    result_dict['queryset'], mixed_query_tree = ht_descriptive_queries(mixed_query_tree, descriptive_query_list)
    result_dict['matching_dict'] = query_dict
    result_dict['domain_queries'] = [len(domain_list) for domain_list in domain_query_list]
    result_dict['merged queries'] = len(descriptive_query_list)

    if profile:
        merge_dp.stop()
        merge_timeline_path = merge_dp.write_ray_timeline('dusm_merge_timeline.json', logger=LOGGER)
        merge_profiling = merge_dp.build_summary(merge_querycount, merge_worker_stats, merge_timeline_path)
        log_profiling_summary(LOGGER, merge_profiling, label='D-U-S-M merge phase profiling')

        result_dict['profiling'] = {
            'per_domain_discovery_time': t_discovery_end - t_discovery_start,
            'merge_phase_time': t_merge_end - t_merge_start,
            'per_domain_profiling': per_domain_profiling,
            'merge_phase_profiling': merge_profiling,
        }
        _log_dusm_top_level_summary(result_dict['profiling'], domain_cnt, len(descriptive_query_list))

    return result_dict


def _log_dusm_top_level_summary(profiling: dict, domain_cnt: int, merged_query_count: int) -> None:
    """Print a short recap tying the per-domain and merge-phase reports together.

    Detailed per-domain and merge-phase breakdowns are logged individually
    (via log_profiling_summary) at the point each phase finishes; this is
    just the top-level roll-up.
    """
    total = profiling['per_domain_discovery_time'] + profiling['merge_phase_time']
    LOGGER.info('===== D-U-S-M top-level profiling summary =====')
    LOGGER.info('Domains discovered:          %i', domain_cnt)
    LOGGER.info('Per-domain discovery total:  %.3fs (%.1f%%)',
                profiling['per_domain_discovery_time'],
                100.0 * profiling['per_domain_discovery_time'] / total if total else 0.0)
    LOGGER.info('Merge phase total:           %.3fs (%.1f%%)',
                profiling['merge_phase_time'],
                100.0 * profiling['merge_phase_time'] / total if total else 0.0)
    LOGGER.info('Merged queries found:        %i', merged_query_count)
    LOGGER.info('See "D-U-C-M (per-domain) profiling" and "D-U-S-M merge phase profiling" '
                'logs above for the detailed per-phase breakdowns.')
    LOGGER.info('=================================================')


# ---------------------------------------------------------------------------
# D-U-S-M entry point (duplicated wiring from dus.py, pointed at the
# distributed per-domain and merge routines above)
# ---------------------------------------------------------------------------

def _per_domain_ducm(domain_sample, supp, max_query_length, domain_patternset, profile: bool = False):
    """Call D-U-C-M (Ray-distributed) for a single domain (per_domain_fn in the outer loop)."""
    return discover_ducm(
        sample=domain_sample,
        supp=supp,
        max_query_length=max_query_length,
        find_descriptive_only=False,
        all_patternset=domain_patternset,
        profile=profile,
    )


def discover_dusm(sample, supp: float, max_query_length: int = -1, profile: bool = False) -> dict:
    """D-U-S-M: per-domain D-U-C-M, then Ray-distributed merge-phase matching.

    Both the per-domain discovery step and the cross-domain merge-phase
    matching step are distributed across persistent ChunkWorker Ray actors.

    Args:
        sample: MultidimSample instance.
        supp: Support threshold in [0, 1].
        max_query_length: Maximum query length (-1 = auto-compute).
        profile: If True, collect timing/profiling data for the per-domain
            discovery phase and the merge phase separately, and attach both
            to the result dict under 'profiling'. See discover_ducm and
            _domain_separated_discovery_distributed for details on what's
            captured.

    Returns:
        Result dict with keys: queryset, matching_dict, domain_queries,
        merged queries. When profile=True, also includes 'profiling'.
    """
    domain_cnt = sample._sample[0].split(' ')[0].count(';')
    if domain_cnt == 1:
        return discover_ducm(sample=sample, supp=supp, max_query_length=max_query_length, profile=profile)

    if max_query_length == -1:
        threshold = ceil(sample._sample_size * supp)
        trace_length = sorted([len(trace.split()) for trace in sample._sample])
        max_query_length = trace_length[sample._sample_size - threshold]

    return _domain_separated_discovery_distributed(
        sample=sample,
        supp=supp,
        matchtest='smarter',
        max_query_length=max_query_length,
        per_domain_fn=_per_domain_ducm,
        profile=profile,
    )
