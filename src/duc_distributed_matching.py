#!/usr/bin/python3
"""D-U-C: Domain Unified Constant discovery algorithm."""
import logging
import time
from collections import deque
from sample_multidim import MultidimSample
from sample import Sample
from math import ceil
import ray

import numpy as np

from query_multidim import MultidimQuery
from hyper_linked_tree import HyperLinkedTree
from discovery_shared import (
    _next_queries_multidim,
    ht_descriptive_queries,
)

LOG_FORMAT = '| %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)


def discover_duc_distributed_matching(sample, supp: float, max_query_length: int = -1,
                                      only_types: bool = False, find_descriptive_only: bool = True,
                                      all_patternset=None) -> dict:
    """D-U-C-M: distributed matching version of D-U-C.

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
    distributions = 4
    chunks = np.array_split(sample._sample, distributions)
    chunks = [rebuild_sample_from_array(chunk) for chunk in chunks]
    chunks_dict = {i + 1: [chunk, dict()] for i, chunk in enumerate(chunks)}

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

    query = MultidimQuery()
    query.set_query_string(gen_event)

    matching_dict = {gen_event: query}
    non_matching_dict = {}
    parent_dict = {gen_event: query}
    #dict_iter = {}
    dictionary = {}
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

        parent = parent_dict[querystring]
        parentstring = parent._query_string
        # dict iter updates need to be passed back
        # patternset could be minimized
        # decide which traces get in which distribution

        futures = [distributed_matching.remote(sample=value[0], query=query, supp=supp, dict_iter=value[1],
                                               all_patternset=all_patternset, parent_dict=parent_dict, chunk_id=key)
                                               for key, value in chunks_dict.items()]
        
        # combine the matching result and update each chunk's dict_iter
        results = [ray.get(future) for future in futures]
        for _, new_dict_iter, chunk_id, updated_parent_dict, query_matched_traces in results:
            chunks_dict[chunk_id][1] = new_dict_iter
            parent_dict.update(updated_parent_dict)
            query._

        matching = all(m for m, _, _, _ in results)


        dictionary[querystring] = matching

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

@ray.remote
def distributed_matching(sample, query, supp, dict_iter, all_patternset, parent_dict, chunk_id):

    return query.match_sample_distributed(
            sample=sample, supp=supp, dict_iter=dict_iter,
            patternset=all_patternset, parent_dict=parent_dict, 
            chunk_id=chunk_id
        )


def rebuild_sample_from_array(original_sample) -> MultidimSample:
    sample = MultidimSample()
    sample.set_sample(list(original_sample))
    sample.calc_sample_typeset(calculate_all=True)
    return sample