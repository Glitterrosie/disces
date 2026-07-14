#!/usr/bin/python3
"""D-U-S: Domain Unified Symbolic discovery algorithm — Ray-distributed."""
import logging
from itertools import product
from math import ceil

import ray

from duc import discover_duc
from query_multidim import MultidimQuery
from discovery_shared import (
    add_vertex2tree,
    _merge_domain_queries,
    adapted_querystring,
    pos2query,
    ht_descriptive_queries,
)
from profiling_helpers import CallTimer, DriverProfiler, log_sequential_profiling_summary

LOG_FORMAT = '| %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)


@ray.remote
def _per_domain_duc(domain_sample, supp, max_query_length, domain_patternset, profile: bool = False):
    return discover_duc(
        sample=domain_sample,
        supp=supp,
        max_query_length=max_query_length,
        find_descriptive_only=False,
        all_patternset=domain_patternset,
        profile=profile,
    )


def discover_dus_dimension(sample, supp: float, max_query_length: int = -1, profile: bool = False) -> dict:
    """D-U-S: per-domain D-U-C in parallel via Ray, then merge across domains.

    Args:
        sample: MultidimSample instance.
        supp: Support threshold in [0, 1].
        max_query_length: Maximum query length (-1 = auto-compute).
        profile: If True, collect profiling data for both phases:
            - per-domain discovery, run as Ray tasks (each gets its own real
              cProfile since it runs in a separate Ray worker process, via
              discover_duc's own `profile` flag);
            - the merge phase, which runs sequentially in the driver process
              and gets its own cProfile + per-call match timing.
            Everything is attached to the result dict under 'profiling'.

    Returns:
        Result dict with keys: queryset, matching_dict, domain_queries,
        merged queries. When profile=True, also includes 'profiling'.
    """
    domain_cnt = sample._sample[0].split(' ')[0].count(';')
    if domain_cnt == 1:
        return discover_duc(sample=sample, supp=supp, max_query_length=max_query_length, profile=profile)

    if max_query_length == -1:
        threshold = ceil(sample._sample_size * supp)
        trace_length = sorted(len(trace.split()) for trace in sample._sample)
        max_query_length = trace_length[sample._sample_size - threshold]

    if not ray.is_initialized():
        ray.init()

    return _domain_separated_discovery_dus(
        sample=sample,
        supp=supp,
        matchtest='smarter',
        max_query_length=max_query_length,
        profile=profile,
    )


def _domain_separated_discovery_dus(sample, supp, matchtest, max_query_length, profile: bool = False) -> dict:
    sample_set = sample._sample
    sample_size = len(sample_set)
    domain_cnt = sample_set[0].split(' ')[0].count(';')

    if supp == 1.0:
        _, min_trace_length = sample.get_sample_min_trace()
        max_query_length = min(max_query_length, min_trace_length)

    dim_sample_dict = sample.get_dim_sample_dict()
    gen_event = ';' * domain_cnt

    all_patternset = {}
    domain_patternsets = {}
    vert_dbs = {}

    for domain, domain_sample in dim_sample_dict.items():
        vert_db = domain_sample.get_att_vertical_sequence_database()
        vert_dbs[domain] = vert_db
        all_patternset[domain] = {trace_id: set() for trace_id in range(sample_size)}
        for key, value in vert_db.items():
            for letter, pos_dict in value.items():
                for trace_id, positions in pos_dict.items():
                    if len(positions) >= 2:
                        all_patternset[domain][trace_id].add(letter)
        domain_patternsets[domain] = {domain: all_patternset[domain]}

    # dispatch/collection timing only here — no cProfile at the driver level,
    # since driver-side work in this phase is just "spawn tasks, wait for
    # them"; each task's own cProfile (inside discover_duc) is unaffected
    # since it runs in a separate Ray worker process.
    domain_dp = DriverProfiler(enabled=profile, use_cprofile=False)
    domain_dp.start()

    # distribute per domain
    with domain_dp.time_dispatch():
        futures = {
            domain: _per_domain_duc.remote(
                domain_sample,
                supp,
                max_query_length,
                domain_patternsets[domain],
                profile,
            )
            for domain, domain_sample in dim_sample_dict.items()
        }

    # collect results
    with domain_dp.time_collection():
        domain_results = {domain: ray.get(future) for domain, future in futures.items()}

    domain_dp.stop()
    per_domain_profiling = (
        {domain: domain_results[domain].get('profiling') for domain in domain_results}
        if profile else None
    )

    # prepare data structures for merge
    query_dict = {}
    dict_iter = {}
    parent_dict = {}
    domain_query_list = []
    all_dictionary = {}
    query_list = {}
    mixed_query_tree = None

    for domain, domain_sample in dim_sample_dict.items():
        result_dict = domain_results[domain]
        vert_db = vert_dbs[domain]
        dom_sample_size = domain_sample._sample_size
        trace_list = list(range(dom_sample_size))

        instance_dictionary = {}
        for result_query in result_dict['matching_dict'].values():
            if supp < 1:
                trace_list = []
                for trace in range(dom_sample_size):
                    querystring = result_query._query_string
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
        if result_dict['matching_dict']:
            dict_iter.update(result_dict['dict_iter'])
        if 'parent_dict' in result_dict:
            parent_dict.update(result_dict['parent_dict'])

        for domain_query in domain_queryset:
            if gen_event == domain_query or not domain_query:
                continue
            all_dictionary[domain_query] = {
                'trace_instances': pos_dict[domain_query],
                'occurences': len(pos_dict[domain_query]),
            }
        domain_query_list.append(list(domain_queryset))

        if mixed_query_tree is None:
            mixed_query_tree = result_dict['query_tree']
        else:
            domain_tree = result_dict['query_tree']
            root_vertex_domain = domain_tree.get_root()
            root_vertex = mixed_query_tree.get_root()
            for child_vertex in root_vertex_domain.child_vertices:
                mixed_query_tree.insert_query_string(
                    root_vertex, child_vertex.query_string,
                    query=child_vertex.query, search_for_parents=False)

    # merge
    descriptive_query_list = set()
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

    # sequential merge phase, in-process — gets its own cProfile since the
    # domain_dp above didn't claim one, plus per-call match_sample timing.
    merge_dp = DriverProfiler(enabled=profile)
    merge_dp.start(logger=LOGGER)
    merge_call_timer = CallTimer(enabled=profile)
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
            poss_queries_v2 = _merge_domain_queries(
                domain_indeces_dict, all_dictionary, max_query_length, supp)
            poss_query_list = set()
            if poss_queries_v2:
                adapted_qs_dict = adapted_querystring(domain_indeces_dict, query_list)
                for pair2 in poss_queries_v2:
                    if pair2 and domain_indeces_dict:
                        querystring = pos2query(
                            domain_indeces_dict, pair2, adapted_qs_dict, max_query_length)
                        if querystring == gen_event or not querystring:
                            continue
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
                        match = merge_call_timer.record(
                            poss_query.match_sample,
                            sample, supp, parent_dict=parent_dict,
                            patternset=all_patternset, dict_iter=dict_iter,
                        )
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

    merge_dp.stop()

    result_dict = {}
    result_dict['queryset'], mixed_query_tree = ht_descriptive_queries(
        mixed_query_tree, descriptive_query_list)
    result_dict['matching_dict'] = query_dict
    result_dict['domain_queries'] = [len(domain_list) for domain_list in domain_query_list]
    result_dict['merged queries'] = len(descriptive_query_list)

    if profile:
        merge_profiling = {
            'wall_clock_total': merge_dp.wall_clock_total,
            'call_stats': merge_call_timer.stats(),
            'cprofile_text': merge_dp.driver_cprofile_text(),
            'querycount': merge_querycount,
        }
        log_sequential_profiling_summary(LOGGER, merge_profiling, label='D-U-S-dimension merge phase profiling')

        result_dict['profiling'] = {
            'domain_dispatch_time': domain_dp.dispatch_time_total,
            'domain_collection_time': domain_dp.collection_time_total,
            'per_domain_profiling': per_domain_profiling,
            'merge_phase_profiling': merge_profiling,
        }
        _log_dusd_top_level_summary(result_dict['profiling'], domain_cnt, len(descriptive_query_list))

    return result_dict


def _log_dusd_top_level_summary(profiling: dict, domain_cnt: int, merged_query_count: int) -> None:
    total = (profiling['domain_dispatch_time'] + profiling['domain_collection_time']
             + profiling['merge_phase_profiling']['wall_clock_total'])
    LOGGER.info('===== D-U-S-dimension top-level profiling summary =====')
    LOGGER.info('Domains discovered:        %i', domain_cnt)
    LOGGER.info('Domain task dispatch time: %.3fs', profiling['domain_dispatch_time'])
    LOGGER.info('Domain task collection time: %.3fs', profiling['domain_collection_time'])
    LOGGER.info(
        'Merge phase wall clock:    %.3fs (%.1f%% of total)',
        profiling['merge_phase_profiling']['wall_clock_total'],
        100.0 * profiling['merge_phase_profiling']['wall_clock_total'] / total if total else 0.0,
    )
    LOGGER.info('Merged queries found:     %i', merged_query_count)
    LOGGER.info('See "D-U-S-dimension merge phase profiling" log above for the merge-phase '
                'breakdown, and profiling["per_domain_profiling"][domain] for each domain\'s '
                'own D-U-C profiling.')
    LOGGER.info('=========================================================')
