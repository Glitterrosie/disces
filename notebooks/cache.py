import os
import ray
ray.init(runtime_env={"env_vars": {"PYTHONPATH": os.path.dirname(os.path.abspath(__file__))}}, ignore_reinit_error=True)


def domain_unified_discovery_smarter(sample, supp, max_query_length, only_types=False, find_descriptive_only=True,
                                     all_patternset = None) -> dict:
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

    stack_collection = []
    stack= deque()
    #stack_collection.append(stack)
    dict_iter = {}
    matching = True
    querycount=1
    dictionary= {}
    parent_dict[querystring] = query

    children = _next_queries_multidim(query,alphabet, max_query_length, patternset)
    parent_dict.update({child._query_string: query for child in children})
    grand_children = []
    non_descriptive = set()
    
    stack.extend(children)
    query_tree = HyperLinkedTree(ceil(supp*sample._sample_size), event_dimension=sample._sample_event_dimension)
    print(query_tree.query_strings_to_list())
    
    start_time = time.time()
    last_print_time = start_time

    for child in children:
        temp_stack = deque()
        temp_stack.append(child)
        stack_collection.append(temp_stack)

    futures = [_process_query.remote(stack, querycount, last_print_time, sample, supp, dict_iter, all_patternset, patternset, matching_dict, non_matching_dict, non_descriptive, query_tree, parent_dict, alphabet, max_query_length, gen_event, dictionary) for stack in stack_collection]
    results = ray.get(futures)
    pprint.pprint(results)
    #print(results)

    # query tree
    for dict in results:
        result_query_tree = dict['query_tree']
        query_tree.insert_query_string(existing_vertex=query_tree.get_root(), query_array=result_query_tree.query_strings_to_list(), query_string= result_query_tree.query_strings_to_list()[-1],query=query, search_for_parents=False)
    print(query_tree.query_strings_to_list())

    #     return {
    #     'stack': stack,
    #     'querycount': querycount,
    #     'last_print_time': last_print_time,
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

    result_dict['querycount'] =  querycount
    result_dict['parent_dict'] = parent_dict
    result_dict['matching_dict'] = matching_dict
    result_dict['dict_iter'] = dict_iter
    result_dict['query_tree'] = query_tree
    result_dict['non_matching_dict'] = non_matching_dict
    result_dict['patternset'] = patternset



    return result_dict


@ray.remote
def _process_query(stack, querycount, last_print_time, sample, supp, dict_iter, all_patternset, patternset,
                   matching_dict, non_matching_dict, non_descriptive, query_tree, parent_dict, alphabet,
                   max_query_length, gen_event, dictionary):
    query = stack.pop()
    querystring = query._query_string
    query.set_query_matchtest('smarter')
    querycount += 1
    current_time = time.time()
    if current_time - last_print_time > 300:
        LOGGER.info('Current query: %s; current stack size: %i; Current Query count: %i', querystring, len(stack), querycount)
        last_print_time = current_time
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
        'stack': stack,
        'querycount': querycount,
        'last_print_time': last_print_time,
        'matching_dict': matching_dict,
        'non_matching_dict': non_matching_dict,
        'non_descriptive': non_descriptive,
        'query_tree': query_tree,
        'parent_dict': parent_dict,
        'dict_iter': dict_iter,
        'dictionary': dictionary,
    }