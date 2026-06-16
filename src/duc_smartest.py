#!/usr/bin/python3
"""D-U-C: Domain Unified Constant discovery algorithm."""
import logging
import time
from collections import deque
from math import ceil

import numpy as np
import os
import ray
import pprint

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


def discover_duc_smartest(sample, supp, max_query_length, only_types=False, find_descriptive_only=True, all_patternset = None) -> dict:
    """Query Discovery by using unified bottom up depth-first search with smarter matching.

    Args:
        sample: Sample instance.
        supp: Float between 0 and 1 which describes the requested support.

    Returns:
        Set of queries if a query has been discovered, None otherwise.
    """
    if max_query_length == -1:
        threshold = ceil(sample._sample_size * supp)
        trace_length = sorted([len(trace.split()) for trace in sample._sample])

        max_query_length = trace_length[sample._sample_size - threshold]
    query_dict= {}
    matching_dict = {}
    non_matching_dict = {}
    domain_cnt = sample._sample_event_dimension
    alphabet = set()
    if supp == 1.0:
        _,min_trace_length= sample.get_sample_min_trace()
        max_query_length = min(max_query_length, min_trace_length)
    gen_event= ';' * domain_cnt
    gen_event_list = [i for i in gen_event]
    att_vsdb = sample.get_att_vertical_sequence_database()
    # if not find_descriptive_only:
    sample_size = sample._sample_size
    vsdb = {}
    if all_patternset:
        patternset ={}
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
        patternset ={}
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
                            # break


    sample_sized_support = ceil(sample._sample_size * supp)
    alphabet = {symbol for symbol,value in vsdb.items() if len(value) >= sample_sized_support}
    parent_dict = {}
    alphabet=sorted(alphabet)
    query = MultidimQuery()
    query.set_query_string(gen_event)
    querystring= query._query_string
    matching_dict[querystring] = query
    dict_iter = {}
    matching = True
    querycount=1
    dictionary= {}

    children = _next_queries_multidim(query,alphabet, max_query_length, patternset)

    parent_dict.update({child._query_string: query for child in children})
    grand_children = []
    non_descriptive = set()

    start_time = time.time()
    last_print_time = start_time

    thread_collection = []

    for child in children:
        stack = deque()
        query_tree = HyperLinkedTree(ceil(supp * sample._sample_size), event_dimension=sample._sample_event_dimension)
        parent_dict = {}

        parent_dict[querystring] = query
        parent_dict[child._query_string] = query
        stack.append(child)
        thread_collection.append((stack,query_tree,parent_dict))


    futures = [_process_query.remote(stack, 1, sample, supp, dict_iter, all_patternset, patternset, matching_dict, non_matching_dict, non_descriptive, query_tree, parent_dict, alphabet, max_query_length, gen_event, dictionary) for stack,query_tree,parent_dict in thread_collection]
    results = ray.get(futures)
    #_process_query(thread_collection[0][0],1,sample,supp,dict_iter,all_patternset,patternset, matching_dict, non_matching_dict, non_descriptive, thread_collection[0][1], thread_collection[0][2], alphabet, max_query_length, gen_event, dictionary)
    for dict in results:
        result_query_tree = dict['query_tree']
        result_matching_dict = dict['matching_dict']
        query_tree.add_subtree_to_vertex(query_tree.get_root(), result_query_tree)
        for query_string, query in result_matching_dict.items():
            matching_dict[query_string] = query

    #['', '$x0; $x0;', '$x0; $x0; $x0;', '$x0; $x0; $x1; $x1;', '$x0; $x1; $x0; $x1;', '$x0; $x1; $x1; $x0;']
    #['', 'aa;']

    #     return {
    #     'stack': stack,
    #     'querycount': querycount,
    #     'matching_dict': matching_dict,
    #     'non_matching_dict': non_matching_dict,
    #     'non_descriptive': non_descriptive,
    #     'query_tree': query_tree,
    #     'parent_dict': parent_dict,
    #     'dict_iter': dict_iter,
    #     'dictionary': dictionary,
    # }

            
    result_dict = {}
    if find_descriptive_only:
        queryset, query_tree = ht_descriptive_queries(query_tree, set(matching_dict.keys()))
        result_dict['queryset'] = queryset - {gen_event}
        
    else:
        result_dict['queryset'] = set(matching_dict.keys()) - {gen_event} - {''}

    result_dict['querycount'] = sum([result['querycount'] for result in results])
    result_dict['parent_dict'] = parent_dict
    result_dict['matching_dict'] = matching_dict
    result_dict['dict_iter'] = dict_iter
    result_dict['query_tree'] = query_tree
    result_dict['non_matching_dict'] = non_matching_dict
    result_dict['patternset'] = patternset



    return result_dict


@ray.remote
def _process_query(stack, querycount, sample, supp, dict_iter, all_patternset, patternset,
                   matching_dict, non_matching_dict, non_descriptive, query_tree, parent_dict, alphabet,
                   max_query_length, gen_event, dictionary):
    while stack:
        query = stack.pop()
        querystring = query._query_string
        query.set_query_matchtest('smarter')
        querycount += 1
        current_time = time.time()
        parent = parent_dict[querystring]
        parentstring = parent._query_string
        matching = query.match_sample(sample=sample, supp=supp, dict_iter=dict_iter, patternset=all_patternset, parent_dict=parent_dict)
        dictionary.update({querystring: matching})

        if not matching:
            non_matching_dict[querystring] = query
        else:
            matching_dict[querystring] = query
            non_descriptive.add(parentstring)

            if parent_dict[querystring]._query_string == gen_event:
                parentstring = ''
            else:
                parentstring = parent_dict[querystring]._query_string
            parent_vertex = query_tree.find_vertex(parentstring)
            if not query_tree.find_vertex(querystring):
                vertex = query_tree.insert_query_string(parent_vertex, querystring, query=query, search_for_parents=False)
                vertex.matched_traces = query._query_matched_traces
            children = _next_queries_multidim(query, alphabet, max_query_length, patternset)
            if children:
                stack.extend(children)
                parent_dict.update({child._query_string: query for child in children})

    return {
        'matching_dict': matching_dict,
        'query_tree': query_tree,
        'parent_dict': parent_dict,
        'querycount': querycount,
    }