#!/usr/bin/python3
"""D-U-S-M: Domain Unified Symbolic discovery, fully distributed via Ray Actors."""
import logging
import time
from collections import deque
from itertools import product
from math import ceil

import numpy as np
import ray

from sample_multidim import MultidimSample
from query_multidim import MultidimQuery
from hyper_linked_tree import HyperLinkedTree
from discovery_shared import (
    _next_queries_multidim,
    ht_descriptive_queries,
    _merge_domain_queries,
    adapted_querystring,
    pos2query,
    add_vertex2tree,
)

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

    def __init__(self, chunk, patternset, chunk_id, supp):
        self.chunk = chunk
        self.patternset = patternset
        self.chunk_id = chunk_id
        self.supp = supp
        self.dict_iter = {}
        self.parent_dict = {}

    def match(self, query):
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


def rebuild_sample_from_array(original_sample) -> MultidimSample:
    sample = MultidimSample()
    sample.set_sample(list(original_sample))
    sample.calc_sample_typeset(calculate_all=True)
    return sample


def _build_chunk_workers(sample, all_patternset, supp, distributions=4):
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
        workers.append(ChunkWorker.remote(chunk, chunk_patternset, chunk_id, supp))
        chunk_offsets.append(offset)
        offset += chunk._sample_size

    return workers, chunk_offsets


def _match_distributed(query, workers, chunk_offsets):
    """Dispatch `query` to all persistent chunk workers and merge the results.

    Collects results as they arrive and stops early as soon as one worker
    reports non-matching.
    """
    query.set_query_matchtest('smarter')
    futures = [worker.match.remote(query) for worker in workers]
    future_to_offset = {f: off for f, off in zip(futures, chunk_offsets)}

    remaining = list(futures)
    matched_traces = []
    matching = True

    while remaining:
        ready, remaining = ray.wait(remaining, num_returns=1)
        matched, chunk_matched_traces = ray.get(ready[0])
        if not matched:
            matching = False
            break  # one non-match is enough; remaining actors finish in background
        off = future_to_offset[ready[0]]
        matched_traces.extend(t + off for t in chunk_matched_traces)

    return matching, matched_traces


def _shutdown_workers(workers):
    for w in workers:
        ray.kill(w)


# ---------------------------------------------------------------------------
# D-U-C-M: per-domain distributed discovery (duplicated from ducm.py)
# ---------------------------------------------------------------------------

def discover_ducm(sample, supp: float, max_query_length: int = -1,
                                      only_types: bool = False, find_descriptive_only: bool = True,
                                      all_patternset=None) -> dict:
    """D-U-C with matching distributed across persistent Ray Actor workers.

    Each worker permanently owns a chunk of the sample and its local dict_iter.
    Per BFS step only the query is sent; workers return one boolean each.

    Args:
        sample: MultidimSample instance.
        supp: Support threshold in [0, 1].
        max_query_length: Maximum query length (-1 = auto-compute from support).
        only_types: If True, skip variable introduction.
        find_descriptive_only: If True, return only descriptive queries.
        all_patternset: Pre-computed patternset (per-domain per-trace).

    Returns:
        Result dict with keys: queryset, querycount, matching_dict,
        non_matching_dict, dict_iter, query_tree, parent_dict, patternset.
    """
    if max_query_length == -1:
        threshold = ceil(sample._sample_size * supp)
        trace_length = sorted([len(trace.split()) for trace in sample._sample])
        max_query_length = trace_length[sample._sample_size - threshold]

    if supp == 1.0:
        _, min_trace_length = sample.get_sample_min_trace()
        max_query_length = min(max_query_length, min_trace_length)

    domain_cnt = sample._sample_event_dimension
    gen_event = ';' * domain_cnt
    gen_event_list = list(gen_event)

    att_vsdb = sample.get_att_vertical_sequence_database()
    sample_size = sample._sample_size
    vsdb = {}

    if all_patternset:
        patternset = {}
        for domain, dom_vsdb in att_vsdb.items():
            patternset[domain] = set()
            for key, value in dom_vsdb.items():
                new_key = ''.join(gen_event_list[:domain] + [key] + gen_event_list[domain:])
                vsdb[new_key] = value
                if not only_types:
                    for item in value.keys():
                        if len(value[item]) >= 2:
                            patternset[domain].add(key)
                            break
    else:
        patternset = {}
        all_patternset = {}
        for domain, dom_vsdb in att_vsdb.items():
            patternset[domain] = set()
            all_patternset[domain] = {trace_id: set() for trace_id in range(sample_size)}
            for key, value in dom_vsdb.items():
                new_key = ''.join(gen_event_list[:domain] + [key] + gen_event_list[domain:])
                vsdb[new_key] = value
                if not only_types:
                    for item in value.keys():
                        if len(value[item]) >= 2:
                            all_patternset[domain][item].add(key)
                            patternset[domain].add(key)

    sample_sized_support = ceil(sample._sample_size * supp)
    alphabet = sorted({symbol for symbol, value in vsdb.items() if len(value) >= sample_sized_support})

    # Build per-chunk patternsets remapped to local trace IDs, then start
    # one persistent Actor per chunk. Actors are created once for this call.
    workers, chunk_offsets = _build_chunk_workers(sample, all_patternset, supp)

    query = MultidimQuery()
    query.set_query_string(gen_event)

    matching_dict = {gen_event: query}
    non_matching_dict = {}
    parent_dict = {gen_event: query}
    querycount = 1

    children = _next_queries_multidim(query, alphabet, max_query_length, patternset)
    parent_dict.update({child._query_string: query for child in children})

    stack = deque(children)
    query_tree = HyperLinkedTree(
        ceil(supp * sample._sample_size),
        event_dimension=sample._sample_event_dimension,
    )

    start_time = time.time()
    last_print_time = start_time

    while stack:
        query = stack.pop()
        querystring = query._query_string
        query.set_query_matchtest('smarter')
        querycount += 1

        current_time = time.time()
        if current_time - last_print_time > 300:
            LOGGER.info(
                'Current query: %s; stack size: %i; query count: %i',
                querystring, len(stack), querycount,
            )
            last_print_time = current_time

        matching, matched_traces = _match_distributed(query, workers, chunk_offsets)
        query._query_matched_traces = matched_traces

        if not matching:
            non_matching_dict[querystring] = query
        else:
            matching_dict[querystring] = query

            if parent_dict[querystring]._query_string == gen_event:
                parentstring = ''
            else:
                parentstring = parent_dict[querystring]._query_string

            parent_vertex = query_tree.find_vertex(parentstring)
            if not query_tree.find_vertex(querystring):
                vertex = query_tree.insert_query_string(
                    parent_vertex, querystring, query=query, search_for_parents=False,
                )
                vertex.matched_traces = query._query_matched_traces

            children = _next_queries_multidim(query, alphabet, max_query_length, patternset)
            if children:
                stack.extend(children)
                parent_dict.update({child._query_string: query for child in children})

    _shutdown_workers(workers)

    result_dict = {}
    if find_descriptive_only:
        queryset, query_tree = ht_descriptive_queries(query_tree, set(matching_dict.keys()))
        result_dict['queryset'] = queryset - {gen_event}
    else:
        result_dict['queryset'] = set(matching_dict.keys()) - {gen_event} - {''}

    result_dict['querycount'] = querycount
    result_dict['parent_dict'] = parent_dict
    result_dict['matching_dict'] = matching_dict
    result_dict['dict_iter'] = None
    result_dict['query_tree'] = query_tree
    result_dict['non_matching_dict'] = non_matching_dict
    result_dict['patternset'] = patternset

    return result_dict


# ---------------------------------------------------------------------------
# Outer loop (duplicated from discovery_shared._domain_separated_discovery),
# with the merge-phase matching step distributed via Ray Actors.
# ---------------------------------------------------------------------------

def _domain_separated_discovery_distributed(sample, supp, matchtest, max_query_length, per_domain_fn) -> dict:
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
                       domain_patternset) -> result_dict.
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
        result_dict = per_domain_fn(domain_sample, supp, max_query_length, domain_patternset)

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
    print(f"Per-domain discovery took {t_discovery_end - t_discovery_start:.4f}s")

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
    merge_workers, merge_chunk_offsets = _build_chunk_workers(sample, all_patternset, supp)

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
                        match, _matched_traces = _match_distributed(
                            poss_query, merge_workers, merge_chunk_offsets)
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

    _shutdown_workers(merge_workers)

    t_merge_end = time.perf_counter()  # <-- END phase 2
    print(f"Merging phase took {t_merge_end - t_merge_start:.4f}s")

    result_dict = {}
    result_dict['queryset'], mixed_query_tree = ht_descriptive_queries(mixed_query_tree, descriptive_query_list)
    result_dict['matching_dict'] = query_dict
    result_dict['domain_queries'] = [len(domain_list) for domain_list in domain_query_list]
    result_dict['merged queries'] = len(descriptive_query_list)
    return result_dict


# ---------------------------------------------------------------------------
# D-U-S-M entry point (duplicated wiring from dus.py, pointed at the
# distributed per-domain and merge routines above)
# ---------------------------------------------------------------------------

def _per_domain_ducm(domain_sample, supp, max_query_length, domain_patternset):
    """Call D-U-C-M (Ray-distributed) for a single domain (per_domain_fn in the outer loop)."""
    return discover_ducm(
        sample=domain_sample,
        supp=supp,
        max_query_length=max_query_length,
        find_descriptive_only=False,
        all_patternset=domain_patternset,
    )


def discover_dusm(sample, supp: float, max_query_length: int = -1) -> dict:
    """D-U-S-M: per-domain D-U-C-M, then Ray-distributed merge-phase matching.

    Both the per-domain discovery step and the cross-domain merge-phase
    matching step are distributed across persistent ChunkWorker Ray actors.

    Args:
        sample: MultidimSample instance.
        supp: Support threshold in [0, 1].
        max_query_length: Maximum query length (-1 = auto-compute).

    Returns:
        Result dict with keys: queryset, matching_dict, domain_queries,
        merged queries.
    """
    domain_cnt = sample._sample[0].split(' ')[0].count(';')
    if domain_cnt == 1:
        return discover_ducm(sample=sample, supp=supp, max_query_length=max_query_length)

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
    )
