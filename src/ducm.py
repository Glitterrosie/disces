#!/usr/bin/python3
"""D-U-C-M: Domain Unified Constant — distributed matching via Ray Actors."""
import logging
import time
from collections import deque
from sample_multidim import MultidimSample
from math import ceil
import ray

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


@ray.remote
class ChunkWorker:
    """Persistent Ray Actor that owns one chunk of the sample.

    The chunk, patternset, dict_iter, and parent_dict live here permanently.
    Only the query object crosses the wire per BFS step.
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


def discover_ducm(sample, supp: float, max_query_length: int = -1,
                                      only_types: bool = False, find_descriptive_only: bool = True,
                                      all_patternset=None, partition_fn=None) -> dict:
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
        partition_fn: Callable(sample, distributions) -> list of trace-id lists,
            one per worker. Defaults to partition_traces_by_length.

    Returns:
        Result dict with keys: queryset, querycount, matching_dict,
        non_matching_dict, dict_iter, query_tree, parent_dict, patternset.
    """
    if partition_fn is None:
        partition_fn = partition_traces_by_length
    distributions = 4
    chunk_id_maps = partition_fn(sample, distributions)
    chunks = [
        rebuild_sample_from_array([sample._sample[trace_id] for trace_id in id_map])
        for id_map in chunk_id_maps
    ]

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
    # one persistent Actor per chunk. Actors are created once for the whole run.
    workers = []
    for i, (chunk, id_map) in enumerate(zip(chunks, chunk_id_maps)):
        chunk_id = i + 1
        chunk_patternset = {
            domain: {local_t: all_patternset[domain][id_map[local_t]]
                     for local_t in range(chunk._sample_size)}
            for domain in all_patternset
        }
        workers.append(ChunkWorker.remote(chunk, chunk_patternset, chunk_id, supp))

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

        # send only the query to each worker; all heavy state stays in the Actor
        futures = [worker.match.remote(query) for worker in workers]
        future_to_id_map = dict(zip(futures, chunk_id_maps))

        # collect results as they arrive; stop as soon as one reports non-matching
        remaining = list(futures)
        query._query_matched_traces = []
        matching = True

        while remaining:
            ready, remaining = ray.wait(remaining, num_returns=1)
            matched, chunk_matched_traces = ray.get(ready[0])
            if not matched:
                matching = False
                break  # one non-match is enough; remaining actors finish in background
            id_map = future_to_id_map[ready[0]]
            query._query_matched_traces.extend(id_map[t] for t in chunk_matched_traces)

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


def rebuild_sample_from_array(original_sample) -> MultidimSample:
    sample = MultidimSample()
    sample.set_sample(list(original_sample))
    sample.calc_sample_typeset(calculate_all=True)
    return sample


def partition_traces_naive(sample, distributions):
    """Split trace ids into `distributions` contiguous chunks, in sample order.

    Matches the original np.array_split behaviour: sizes differ by at most
    one, with the first `sample_size % distributions` chunks getting the
    extra trace. No balancing or reordering by length.

    Returns:
        List of `distributions` lists of original trace ids.
    """
    sample_size = sample._sample_size
    base, remainder = divmod(sample_size, distributions)

    chunk_id_maps = []
    start = 0
    for i in range(distributions):
        size = base + (1 if i < remainder else 0)
        chunk_id_maps.append(list(range(start, start + size)))
        start += size

    return chunk_id_maps


def partition_traces_by_length(sample, distributions):
    """Split trace ids into `distributions` chunks balanced by total trace length.

    Uses greedy LPT (longest-processing-time-first): traces are assigned,
    longest first, to whichever chunk currently has the smallest total
    length, which keeps the workers' total workload close to even. Each
    chunk is then reordered ascending by trace length, so a worker starts
    with its shortest trace and matches faster traces first.

    Returns:
        List of `distributions` lists of original trace ids.
    """
    lengths = [len(trace.split()) for trace in sample._sample]
    order = sorted(range(len(lengths)), key=lambda trace_id: lengths[trace_id], reverse=True)

    chunk_id_maps = [[] for _ in range(distributions)]
    chunk_loads = [0] * distributions
    for trace_id in order:
        worker = min(range(distributions), key=lambda w: chunk_loads[w])
        chunk_id_maps[worker].append(trace_id)
        chunk_loads[worker] += lengths[trace_id]

    for id_map in chunk_id_maps:
        id_map.sort(key=lambda trace_id: lengths[trace_id])

    return chunk_id_maps
