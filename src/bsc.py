#!/usr/bin/python3
"""B-S-C: Bottom-up Separated Constant discovery algorithm."""
import logging
import time
from math import ceil
from copy import deepcopy
import numpy as np
from query_multidim import MultidimQuery
from error import EmptySampleError, InvalidQuerySupportError
from discovery_shared import combine_all, merge_event_arrays
from sample_multidim import MultidimSample
from hyper_linked_tree import HyperLinkedTree, Vertex

PATTER_TYPE_SPLIT_DISCOVERY_ORDER = [
        'type_first'
        ]
FULL_CHECK_ON_SAMPLE = False

#Logger Configuration:
LOG_FORMAT = '| %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

MAX_QUERY_LENGTH = float('inf')

def discovery_bu_pts_multidim(sample:MultidimSample, supp:float, use_tree_structure:bool=False, use_smart_matching:bool=False, discovery_order:str='type_first',
        max_query_length:int=-1 ,find_descriptive_only:bool=True,
        all_patternset = None) -> list:
    """
        Query Discovery by using a variant of pattern-type-split (BU) algorithm.
        Creates separate structures for pattern- and type-queries and combines them afterwards.
        Lastly tries to syntactically prune non-descriptive queries.
        The algorithm mines each dimension separately and joins them afterwards.

        Args:
            sample: Sample instance.

            supp: Float between 0 and 1 which describes the requested support.

            use_tree_structure [= False]: Optional parameter to decide whether the result is stored into a hyperlinked tree or not.

            discovery_order [= "type_first"]: Optional order to decide whether the starting to discover type-queries first ("type-first") or in any other order (not implemented
            yet).

            use_smart_matching [= False]: Uses a iterative dictionary based approach to solve matchings instead of regex

            max_query_length [= -1]: No queries higher than this length will be found. In case of -1, the query length is unbounded.

            find_descriptive_only [= True]: If true, only the descriptive queries will be return (= True). Otherwise all queries matching the sample will be return (= False).

        Returns:
            list: A list of all discovered queries and a dictionary of stat's.

        Raises:
            EmptySampleError: The given sample is empty.
            InvalidQuerySupportError: Supp is less than 0 or greater than 1.
            NotImplementedError: Selected discovery_order is not available

        Passes:
            IndexError: from _build_mixed_query_tree(...)
            TypeError: from _build_mixed_query_tree(...)
            ValueError: from _build_type_tree(...)
            ValueError: from _build_pattern_tree(...)
            ValueError: from _build_mixed_query_tree(...)
    """
    # LOGGER.info('Pattern-Type-Split (BU) - Started')

    if not isinstance(sample, MultidimSample):
        raise EmptySampleError('No sample given.')
    if not sample._sample or sample._sample_size < 1:
        raise EmptySampleError('The given sample is empty. (Sample size: ' + str(sample._sample_size) + ')')
    if supp < 0.0 or supp > 1.0:
        raise InvalidQuerySupportError(f'Support {supp} has to be between 0 and 1.')

    # Init statistics
    stats = {}
    stats["param match test count"] = 0
    stats["match_test_runtime_list"] = []

    # Init smart matching
    if use_smart_matching:
        if all_patternset:
            param_smart_matching = ({}, all_patternset, {})
        else:
            param_smart_matching = ({}, {}, {})
    else:
        param_smart_matching = None
    global MAX_QUERY_LENGTH
    if supp == 1.0:
        _, MAX_QUERY_LENGTH = sample.get_sample_min_trace()
    if max_query_length > 0:
        MAX_QUERY_LENGTH = min(max_query_length, MAX_QUERY_LENGTH)
    # Start of core algorithm
    if discovery_order =='type_first':
        # LOGGER.info(">> S type tree")
        time_build_type_tree_start = time.time()
        type_tree = _build_type_tree_multidim(stats, sample, supp, use_tree_structure, param_smart_matching)
        time_build_type_tree = time.time() - time_build_type_tree_start
        # try:
        #     # LOGGER.info(">> F type tree [%s + %s] with %s", str(len(type_tree.vertices_to_list())), str(type_tree.collision_counter), str(time_build_type_tree))
        # except AttributeError:
        #     # LOGGER.info(">> F types [%s] with %s",str(len(type_tree)), str(time_build_type_tree))

        # LOGGER.info(">> S pattern tree")
        time_build_pattern_tree_start = time.time()
        pattern_tree = _build_pattern_tree_multidim(stats, sample, supp, use_tree_structure, param_smart_matching)
        time_build_pattern_tree = time.time() - time_build_pattern_tree_start
        # try:
        #     # LOGGER.info(">> F pattern tree [%s + %s] with %s", str(len(pattern_tree.vertices_to_list())), str(pattern_tree.collision_counter), str(time_build_pattern_tree))
        # except AttributeError:
        #     # LOGGER.info(">> F patterns [%s] with %s", str(len(pattern_tree)), str(time_build_pattern_tree))

        # LOGGER.info(">> S mixed_query tree")
        time_build_mixed_query_tree_start = time.time()
        mixed_query_tree = _build_mixed_query_tree_multidim(stats, sample, supp, type_tree, pattern_tree, param_smart_matching, find_descriptive_only=find_descriptive_only)
        time_build_mixed_query_tree = time.time() - time_build_mixed_query_tree_start
        # try:
        #     # LOGGER.info(">> F mixed_query tree [%s + %s] with %s", str(len(mixed_query_tree.vertices_to_list())), str(mixed_query_tree.collision_counter),  str(time_build_mixed_query_tree))
        # except AttributeError:
        #     # LOGGER.info(">> F mixed queries [%s] with %s", str(len(mixed_query_tree)), str(time_build_mixed_query_tree))

        # LOGGER.info(">> S descriptive process")
        time_extract_descriptive_start = time.time()
        if use_tree_structure:
            if find_descriptive_only:
                descriptive_set = set()
                descriptive_query_set = []
                int_support = ceil(supp*sample._sample_size)
                for vertex in mixed_query_tree.vertices_to_list():
                    if vertex.is_frequent(int_support):
                        if vertex.descriptive:
                            descriptive_set.add(vertex.query_string)
                            descriptive_query_set.append(vertex.query)

                typeset = list(sample.get_supported_typeset(supp))
                final_descriptive_mixed_query_set = set()
                for query_string in descriptive_set:
                    variable_num_list = []
                    variable_num = 0
                    descriptive = True
                    while True:
                        variable = "$x" + str(variable_num)
                        if variable in query_string:
                            variable_num_list.append(variable_num)
                            variable_num += 1
                        else:
                            break
                    for variable_num in variable_num_list:
                        for new_element in typeset + [str("$x" + str(var_num)) for var_num in range(0, variable_num)]:
                            new_query_string = query_string.replace("$x"+str(variable_num), new_element)
                            old_var_list = [str("$x" +str(var_num)) for var_num in range (variable_num, max(variable_num_list)+1)]
                            for i in range(1, len(old_var_list)):
                                new_query_string = new_query_string.replace(old_var_list[i], old_var_list[i-1])
                            existing_vertex = mixed_query_tree.find_vertex(new_query_string)
                            if existing_vertex and existing_vertex.is_frequent(int_support):
                                descriptive = False
                                break
                        if not descriptive:
                            break
                    if descriptive:
                        final_descriptive_mixed_query_set.add(query_string)
                descriptive_mixed_queries = final_descriptive_mixed_query_set
            else:
                descriptive_mixed_queries = mixed_query_tree.query_strings_to_set(frequent_items_only=True)
        else:
            if find_descriptive_only:
                descriptive_mixed_queries = _find_descriptive_querystrings(stats, mixed_query_tree)
            else:
                descriptive_mixed_queries = mixed_query_tree
        time_extract_descriptive = time.time() - time_extract_descriptive_start
        # LOGGER.info(">> F descriptive process with %s", str(time_extract_descriptive))
        time_total = time.time() - time_build_type_tree_start
        # LOGGER.info("> F total process with %s", str(time_total))
    else:
        raise NotImplementedError("No exception for wrong input defined!")


    # Save collected statistics
    stats["time_build_type_tree"] = time_build_type_tree
    stats["time_build_pattern_tree"] = time_build_pattern_tree
    stats["time_build_mixed_query_tree"] = time_build_mixed_query_tree
    stats["time_total"] = time_total
    #match_test_runtime_average = (sum(match_test_runtime_list)/len(match_test_runtime_list))
    #stats["param avg match test runtime for single position"] = match_test_runtime_average
    #stats["param avg match test runtime for whole query"] = match_test_runtime_average*query._query_string_length
    #stats["param supported typeset"] = sample._get_supported_typeset(supp)

    # LOGGER.info('Pattern-Type-Split (BU) - Finished')
    result_dict = {}
    result_dict['queryset'] = descriptive_mixed_queries
    if use_tree_structure and param_smart_matching:
        result_dict['dict_iter'] = param_smart_matching[0]
        result_dict['patternset'] = param_smart_matching[1]
        result_dict['parent_dict'] = param_smart_matching[2]
        matchingset = {}
        for querystring in descriptive_mixed_queries:
            vertex = mixed_query_tree.find_vertex(querystring)
            if querystring:
                matchingset[querystring] = vertex.query
        result_dict['matching_dict'] = matchingset
        result_dict['querycount'] = len(mixed_query_tree.vertices_to_list())
        result_dict['query_tree'] = mixed_query_tree
    MAX_QUERY_LENGTH = float('inf')

    return (descriptive_mixed_queries, stats, result_dict)


def discover_bsc(sample: MultidimSample, supp: float, max_query_length: int = -1,
                 find_descriptive_only: bool = True, all_patternset=None) -> dict:
    """B-S-C: pattern-type-split bottom-up discovery. Public entry point.

    Returns the result_dict (third element of the internal tuple).
    """
    _, _, result_dict = discovery_bu_pts_multidim(
        sample=sample,
        supp=supp,
        use_smart_matching=True,
        discovery_order='type_first',
        use_tree_structure=True,
        max_query_length=max_query_length,
        find_descriptive_only=find_descriptive_only,
        all_patternset=all_patternset,
    )
    # The tree root (empty string) leaks into the queryset when no queries are
    # found; filter it out to stay consistent with the other algorithms.
    result_dict['queryset'] = result_dict.get('queryset', set()) - {''}
    return result_dict


def _search_type_multidim(sample:MultidimSample, given_pat:str, given_event:list, mutation_index:int, s_n_next_event:dict, s_n_same_event:dict, supp:float,
        already_existing_tree:HyperLinkedTree|None=None, current_vertex:Vertex=None, param_smart_matching:tuple | None=None) -> set:
    """
        Recursive iteration to add further frequent symbols, checking if they are still frequent and iterating the process until support condition can no longer be achieved.
        In the case of hyperlinked trees, a current_vertex is required to store the found results.

        Args:
            sample: Sample instance.

            given_pat: Pattern to increment with frequent symbols. The pattern is immutable and will be further extended by given_event.

            given_event: An array containg some entries for different dimensions. It will be joint togehter and extends given_pat to form the new pattern that will be matched.
                Moreover only the dimension between the 'last' entry and the last dimension are mutable.

            mutation_index: It marks the first mutable position in the given_event.

            s_n_next_event: Dictionary of frequent items per dimension, which are currently still available to put in further events.

            s_n_same_event: Dictionary of frequent items per dimension, which are currently still available to put in the current event.

            supp: Float between 0 and 1 which describes the requested support.

            already_existing_tree [= None]: An hyperlinked tree can be given to store all found queries there

            current_vertex [= None]: If a hyperlinked tree is given, then a vertex has to be given as well to store the results to this vertex.

            param_smart_matching [= None]: requires a tuple (Dictionary, Variable set) to evaluate the queries on the sample

        Returns:
            set: Set of type-queries fitting support. Set is empty if the HyperLinkedTree is used.

        Raises:
            None

        Passes:
            ValueError: from matching_smarter
    """
    s_temp = {}
    type_queries = set()

    # s-extension
    if given_pat == "":
        current_pat = ";".join(given_event)+";"
    else:
        current_pat = given_pat + " " + ";".join(given_event)+";"

    explore_queue = []
    if given_pat.count(' ')+1 < MAX_QUERY_LENGTH:
        for dimension, alphabet in s_n_next_event.items():
            s_temp[dimension] = set()
            for symbol in alphabet:
                new_event_string = (";"*(dimension))+str(symbol)+(";"*(len(s_n_next_event)-dimension))
                new_pattern = current_pat+" "+new_event_string
                new_query = MultidimQuery()
                new_query.set_query_string(new_pattern)

                if not param_smart_matching:
                    pattern_is_frequent = new_query.match_sample(sample, supp)
                else:
                    new_query.set_query_matchtest('smarter')
                    pattern_is_frequent = new_query.match_sample(sample, supp, dict_iter= param_smart_matching[0], patternset=param_smart_matching[1],
                            parent_dict=param_smart_matching[2], max_query_length=MAX_QUERY_LENGTH)
                if pattern_is_frequent:
                    s_temp[dimension].add(symbol)
                    if already_existing_tree:
                        new_query_vertex = already_existing_tree.insert_query_string(current_vertex, new_pattern, query=new_query, search_for_parents=False,
                                set_descriptive_property=True)
                        new_query_vertex.query_next_insert_index = current_vertex.query_next_insert_index[0] * np.ones(len(s_n_next_event), dtype=np.int8)
                        new_query_vertex.query_next_insert_index[0:dimension+1] += np.ones(dimension+1, dtype=np.int8)
                        already_existing_tree.set_match_results(new_query_vertex, new_query._query_matched_traces)
                        explore_queue.append(new_query_vertex)
                    else:
                        type_queries.add(new_pattern)
        if already_existing_tree:
            for vertex in explore_queue:
                new_mutation_index = np.where(vertex.query_next_insert_index[:-1] != vertex.query_next_insert_index[1:])[0]
                if len(new_mutation_index):
                    new_mutation_index = new_mutation_index[0] + 1
                else:
                    new_mutation_index = sample._sample_event_dimension
                _search_type_multidim(sample, current_pat, vertex.query_array[-1], new_mutation_index, s_temp, s_temp, supp, already_existing_tree, vertex,
                        param_smart_matching=param_smart_matching)
        else:
            for dimension, alphabet in s_temp.items():
                for symbol in alphabet:
                    next_event = [""] * (len(s_temp))
                    next_event[dimension] = symbol
                    new_mutation_index = dimension+1

                    type_queries |= _search_type_multidim(sample, current_pat, next_event, new_mutation_index, s_temp, s_temp, supp, param_smart_matching=param_smart_matching)

    # i-extension
    explore_queue = []
    s_temp_same_event = {}
    for dimension, alphabet in s_n_same_event.items():
        if dimension < mutation_index:
            continue
        s_temp_same_event[dimension] = set()
        for symbol in alphabet:
            new_event = given_event.copy()
            new_event[dimension] = symbol
            if given_pat == "":
                new_pattern = ";".join(new_event)+";"
            else:
                new_pattern = given_pat + " " + ";".join(new_event)+";"

            new_query = MultidimQuery()
            new_query.set_query_string(new_pattern)
            if not param_smart_matching:
                pattern_is_frequent = new_query.match_sample(sample, supp)
            else:
                new_query.set_query_matchtest('smarter')
                pattern_is_frequent = new_query.match_sample(sample, supp, dict_iter= param_smart_matching[0], patternset=param_smart_matching[1],
                        parent_dict=param_smart_matching[2], max_query_length=MAX_QUERY_LENGTH)
            if pattern_is_frequent:
                s_temp_same_event[dimension].add(symbol)
                if already_existing_tree:
                    new_query_vertex = already_existing_tree.insert_query_string(current_vertex, new_pattern, query=new_query, search_for_parents=False,
                            set_descriptive_property=True)
                    new_query_vertex.query_next_insert_index = np.copy(current_vertex.query_next_insert_index)
                    new_query_vertex.query_next_insert_index[:dimension+1] = new_query_vertex.query_next_insert_index[0] * np.ones(dimension+1, dtype=np.int8)
                    already_existing_tree.set_match_results(new_query_vertex, new_query._query_matched_traces)
                    explore_queue.append(new_query_vertex)
                else:
                    type_queries.add(new_pattern)
    if already_existing_tree:
        for vertex in explore_queue:
            new_mutation_index = np.where(vertex.query_next_insert_index[:-1] != vertex.query_next_insert_index[1:])[0]
            if len(new_mutation_index):
                new_mutation_index = new_mutation_index[0] + 1
            else:
                new_mutation_index = sample._sample_event_dimension
            _search_type_multidim(sample, given_pat, vertex.query_array[-1], new_mutation_index, s_temp, s_temp_same_event, supp, already_existing_tree, vertex,
                    param_smart_matching=param_smart_matching)
    else:
        for dimension, alphabet in s_temp_same_event.items():
            for symbol in alphabet:
                new_event = given_event.copy()
                new_event[dimension] = symbol
                mutation_index = dimension+1

                type_queries |= _search_type_multidim(sample, given_pat, new_event, mutation_index, s_temp,s_temp_same_event, supp, param_smart_matching=param_smart_matching)

    return type_queries

def _build_type_tree_multidim(stats:dict, sample:MultidimSample, supp:float, use_tree_structure:bool=False, param_smart_matching:tuple|None=None) -> HyperLinkedTree|set:
    """
        Builds a set with all queries containing only types and fitting the sample with the given support.
        The dimensions are mined separately and joint afterwards.
        If a tree structure shall be used, then the results are stored in a hyperlinked tree.

        Args:
            stats: Dictionary for statistical evaluation.

            sample: Sample instance.

            supp: Float between 0 and 1 which describes the requested support.

            use_tree_structure [= False]: Optional parameter to decide whether the result is stored into a hyperlinked tree or not.

            param_smart_matching [= None]: requires a tuple (Dictionary, Variable set) to evaluate the queries on the sample

        Returns:
            HyperLinkedTree: All found type-queries fitting the support arranged in a HyperLinkedTree if use_tree-Structure == True.
            set: Otherwise a set of type-queries which fitting support.

        Raises:
            None

        Passes:
            ValueError: from _search_type
    """
    complete_vsdb = sample.get_att_vertical_sequence_database()

    s_init = { i : { key for key in complete_vsdb[i].keys() if len(complete_vsdb[i][key]) >= ceil(sample._sample_size * supp) } for i in complete_vsdb.keys() }

    type_queries = None
    if use_tree_structure:
        type_queries = HyperLinkedTree(ceil(supp*sample._sample_size), event_dimension=sample._sample_event_dimension)
        root_vertex = type_queries.get_root()
        root_vertex.query_next_insert_index = np.zeros(len(s_init), dtype=np.int8)
        for dim in s_init:
            for symbol in s_init[dim]:
                current_event  = [""] * (len(s_init))
                current_event[dim] = symbol
                mutation_index = dim+1
                querystring = ";".join(current_event)+";"
                param_smart_matching[0][querystring] = {key: value[0] for key, value in complete_vsdb[dim][symbol].items()}
                new_query_vertex = type_queries.insert_query_string(root_vertex, ";".join(current_event)+";", query=MultidimQuery(querystring), search_for_parents=False,
                        set_descriptive_property=True)
                new_query_vertex.query_next_insert_index = np.zeros(len(s_init), dtype=np.int8)
                new_query_vertex.query_next_insert_index[:dim+1] = np.ones(dim+1, dtype=np.int8)
                type_queries.set_match_results(new_query_vertex, list(complete_vsdb[dim][symbol].keys()))

                _search_type_multidim(sample, "", current_event, mutation_index, s_init, s_init, supp, already_existing_tree=type_queries, current_vertex=new_query_vertex,
                        param_smart_matching=param_smart_matching)
    else:
        type_queries = {''}
        for dimension, alphabet in s_init.items():
            for symbol in alphabet:
                current_event  = [""] * (len(s_init))
                current_event[dimension] = symbol
                mutation_index = dimension+1

                type_queries.add(";".join(current_event)+";")
                type_queries |= _search_type_multidim(sample, "", current_event, mutation_index, s_init, s_init, supp, param_smart_matching=param_smart_matching)

    return type_queries

def _search_var_multidim(sample:MultidimSample, next_var_number:int, given_pattern:str, given_event:list, mutation_index:int, s_n:dict, s_n_same_event:dict, supp:float, stats:dict,
        allow_new_variables:bool, already_existing_tree:HyperLinkedTree|None=None, current_vertex:Vertex|None=None, param_smart_matching:tuple|None=None) -> set:
    """
        Recursive iteration to add further variables, checking if they are still frequent and iterating the process until support condition can no longer be achieved.
        In the case of hyperlinked trees, a current_vertex is required to store the found results.

        Args:
            sample: Sample instance.

            next_var_number: The current number of the next inserted variable.

            given_pat: Pattern to increment with frequent symbols. The pattern is immutable and will be further extended by given_event.

            given_event: An array containg some entries for different dimensions. It will be joint togehter and extends given_pat to form the new pattern that will be matched.
                Moreover only the dimension between the 'last' entry and the last dimension are mutable.

            mutation_index: It marks the first mutable position in the given_event.

            s_n: Dictionary of frequent items per dimension, which are currently still available to put in further events.

            s_n_same_event: Dictionary of frequent items per dimension, which are currently still available to put in the current event.

            supp: Float between 0 and 1 which describes the requested support.

            stats: Dictionary for statistical evaluation.

            allow_new_variables: Boolean, if the recursion is allowed to insert new variables. It is set to false, if a newly inserted variable can not fit support, then it should
            not try to insert a variable later.

            already_existing_tree [= None]: Optional parameter to decide whether the result is stored into a hyperlinked tree or not.

            current_vertex [= None]: The Current vertex to which the found queries will be added to.

            param_smart_matching [= None]: requires a tuple (Dictionary, Variable set) to evaluate the queries on the sample

        Returns:
            set: Set of pattern-queries containing only variables and fitting support. Is empty if already_existing_tree is not None.

        Raises:
            None

        Passes:
            ValueError: from matching_smarter
    """
    s_temp = {}
    event_dim = sample._sample_event_dimension
    pattern_queries = set()
    allow_new_variables_for_next_iter = allow_new_variables.copy()

    # new Variable possible?
    if given_pattern == "":
        current_pattern = ";".join(given_event)+";"
    else:
        current_pattern = given_pattern + " " + ";".join(given_event)+";"

    for dim, allow_new_variable in enumerate(allow_new_variables):
        if allow_new_variable:
            if dim < mutation_index:
                single_var_pattern = current_pattern    + ' ' + ';'*dim + '$x' + str(next_var_number) + ';'*(event_dim - dim)
                double_var_pattern = single_var_pattern + ' ' + ';'*dim + '$x' + str(next_var_number) + ';'*(event_dim - dim)

                double_pat_query = MultidimQuery()
                double_pat_query.set_query_string(double_var_pattern)

                if not param_smart_matching:
                    dpq_is_frequent = double_pat_query.match_sample(sample, supp)
                else:
                    double_pat_query.set_query_matchtest('smarter')
                    dpq_is_frequent = double_pat_query.match_sample(sample, supp, dict_iter= param_smart_matching[0], patternset=param_smart_matching[1],
                            parent_dict=param_smart_matching[2], max_query_length=MAX_QUERY_LENGTH)

                if not dpq_is_frequent:
                    allow_new_variables_for_next_iter[dim] = False
            else:
                new_event = given_event.copy()
                new_event[dim] = '$x'+str(next_var_number)
                double_var_pattern = given_pattern + " " + ";".join(given_event) + "; " + ';'*dim + '$x'+str(next_var_number) + ';'*(event_dim - dim)

                double_pat_query = MultidimQuery()
                double_pat_query.set_query_string(double_var_pattern)
                dpq_is_frequent = double_pat_query.match_sample(sample, supp)

                if not dpq_is_frequent:
                    allow_new_variables[dim] = False

    # s-extension
    if given_pattern.count(' ')+1 < MAX_QUERY_LENGTH:
        for (dim, alphabet) in s_n.items():
            s_temp[dim] = set()
            for j in alphabet:
                new_pattern = current_pattern + ' ' + ";"*dim + j + ";"*(event_dim-dim)
                new_query = MultidimQuery()
                new_query.set_query_string(new_pattern)

                if not param_smart_matching:
                    pattern_is_frequent = new_query.match_sample(sample, supp)
                else:
                    new_query.set_query_matchtest('smarter')
                    pattern_is_frequent = new_query.match_sample(sample, supp, dict_iter= param_smart_matching[0], patternset=param_smart_matching[1],
                            parent_dict=param_smart_matching[2], max_query_length=MAX_QUERY_LENGTH)
                if pattern_is_frequent:
                    s_temp[dim].add(j)
                    if already_existing_tree:
                        new_query_vertex = already_existing_tree.insert_query_string(current_vertex, new_pattern, query=new_query, search_for_parents=False)
                        new_query_vertex.query_next_insert_index = np.zeros(event_dim, dtype=np.int8)
                        new_query_vertex.query_next_insert_index[:dim+1] += np.ones(dim+1, dtype=np.int8)
                        already_existing_tree.set_match_results(new_query_vertex, new_query._query_matched_traces)
                    else:
                        pattern_queries.add(new_pattern)

        if already_existing_tree:
            for vertex in current_vertex.child_vertices:
                _search_var_multidim(sample, next_var_number+1, single_var_pattern, s_temp, supp, stats, True, already_existing_tree, current_vertex)
            if allow_new_variables:
                s_temp.add("$x"+str(next_var_number))
                _search_var_multidim(sample, next_var_number+1, single_var_pattern, s_temp, supp, stats, True, already_existing_tree, current_vertex)
        else:
            for (dim, alphabet) in s_temp.items():
                for j in alphabet:
                    new_event = [""]*event_dim
                    new_event[dim] = j
                    pattern_queries |= _search_var_multidim(sample, next_var_number, current_pattern, new_event, dim+1, s_temp, s_temp, supp, stats, allow_new_variables_for_next_iter)
            for dim, allow_new_variable in enumerate(allow_new_variables_for_next_iter):
                if allow_new_variable:
                    new_s_temp = deepcopy(s_temp)
                    new_event = [""]*event_dim
                    new_event[dim] = "$x"+str(next_var_number)
                    new_s_temp[dim].add("$x"+str(next_var_number))
                    pattern_queries |= _search_var_multidim(sample, next_var_number+1, current_pattern, new_event, dim+1, new_s_temp, new_s_temp, supp, stats,
                            allow_new_variables_for_next_iter)
    else:
        for (dim, alphabet) in s_n.items():
            s_temp[dim] = set()

    # i-extension
    s_temp_same_event = {}
    for (dim, alphabet) in s_n_same_event.items():
        if dim >= mutation_index:
            s_temp_same_event[dim] = set()

            for j in alphabet:
                new_event = given_event.copy()
                new_event[dim] = j

                new_pattern = given_pattern + ' ' + ";".join(new_event) + ";"
                new_query = MultidimQuery()
                new_query.set_query_string(new_pattern)

                if not param_smart_matching:
                    pattern_is_frequent = new_query.match_sample(sample, supp)
                else:
                    new_query.set_query_matchtest('smarter')
                    pattern_is_frequent = new_query.match_sample(sample, supp, dict_iter= param_smart_matching[0], patternset=param_smart_matching[1],
                            parent_dict=param_smart_matching[2], max_query_length=MAX_QUERY_LENGTH)
                if pattern_is_frequent:
                    s_temp_same_event[dim].add(j)
                    if already_existing_tree:
                        new_query_vertex = already_existing_tree.insert_query_string(current_vertex, new_pattern, query=new_query, search_for_parents=False)
                        next_vertex.query_next_insert_index = np.zeros(event_dimension, dtype=np.int8)
                        next_vertex.query_next_insert_index[:dim+1] += np.ones(dim+1, dtype=np.int8)
                        already_existing_tree.set_match_results(new_query_vertex, matched_traces)
                    else:
                        pattern_queries.add(new_pattern)
    if already_existing_tree:
        for vertex in current_vertex._child_vertices:
            _search_type(sample, vertex._query_string, s_temp, supp, already_existing_tree, vertex, param_smart_matching)
        if allow_new_variables:
            s_temp.add("$x"+str(next_var_number))
            _search_var(sample, next_var_number+1, single_var_pattern, s_temp, supp, stats, True, already_existing_tree, current_vertex)
    else:
        for (dim, alphabet) in s_temp_same_event.items():
            for j in alphabet:
                new_event = given_event.copy()
                new_event[dim] = j
                pattern_queries |= _search_var_multidim(sample, next_var_number, given_pattern, new_event, dim+1, s_temp, s_temp_same_event, supp, stats,
                        allow_new_variables.copy())
            for allow_new_variable in allow_new_variables:
                if allow_new_variable:
                    new_s_temp = deepcopy(s_temp)
                    new_s_temp_same_event = deepcopy(s_temp)

                    new_event = given_event.copy()
                    new_event[dim] = "$x"+str(next_var_number)

                    new_s_temp[dim].add("$x"+str(next_var_number))
                    new_s_temp_same_event[dim].add("$x"+str(next_var_number))

                    pattern_queries |= _search_var_multidim(sample, next_var_number+1, given_pattern, new_event, dim+1, new_s_temp, new_s_temp_same_event, supp, stats,
                            allow_new_variables.copy())
    return pattern_queries

def _check_frequency_of_query(stats:dict, sample:MultidimSample, supp:float, query:MultidimQuery, param_smart_matching:tuple|None=None) -> bool:
    """
        A helping function to determine which frequency test shall used based on the given 'param_smart_matching' value.

        Args:
            stats: Dictionary for statistical evaluation.

            sample: Sample instance.

            supp: Float between 0 and 1 which describes the requested support.

            query: Frequency will be checked for the given query.

            param_smart_matching [= None]: requires a tuple (Dictionary, Variable set) to evaluate the queries on the sample

        Returns:
            bool: Returns True if the query matches and False otherwise.

        Raises:
            None

        Passes:
            ValueError: from matching_smarter
    """
    if not param_smart_matching:
        pattern_is_frequent = query.match_sample(sample, supp)
    else:
        query.set_query_matchtest('smarter')
        pattern_is_frequent = query.match_sample(sample, supp, dict_iter= param_smart_matching[0], patternset=param_smart_matching[1], parent_dict=param_smart_matching[2], max_query_length=MAX_QUERY_LENGTH)
    return pattern_is_frequent

def _calculate_index_shift(stats:dict, var_to_next_insert_index_list:list, indices_to_increase:list, indices_to_replace:list, first_index:int, second_index:int|None=None,
        first_shift:int=1, second_shift:int=1) -> tuple:
    """
        A helping function to calculate the new indicies based on an insersion at first (and second) index.

        Args:
            stats: Dictionary for statistical evaluation.

            var_to_next_insert_index_list: Conatins the lowst index for each var, where it could be inserted.

            indices_to_increase: Contains all indicies of var_to_next_insert_index_list that have to be increased.

            indices_to_replace: Contains all indicies of var_to_next_insert_index_list that have to be replaced i.e. not only ajusted but be set much further.

            first_index: Represents the position where the ajustment occures.

            second_index [= None]: In case of an insersion of two variables a second index can be given as well.

            first_shift [= 1]: In most cases the indicies will be ajusted by +1. But it might be of interest to increase by more or less.

            second_shift [= 1]: In most cases the indicies will be ajusted by +1. But it might be of interest to increase by more or less.

            param_smart_matching [= None]: requires a tuple (Dictionary, Variable set) to evaluate the queries on the sample

        Returns:
            tuple: The 2-tuple returns the adjusted var_to_next_insert_index_list in the first argument and the adjusted indices_to_increase in second argument.

        Raises:
            None

        Passes:
            None
    """
    if second_index is None:
        new_indices_to_increase = [index for index in indices_to_increase if var_to_next_insert_index_list[index] > first_index] # for growing i this set will decrease
        indices_to_replace.extend(set(indices_to_increase) - set(new_indices_to_increase))
        indices_to_increase = new_indices_to_increase
        new_var_to_next_insert_index_list = var_to_next_insert_index_list.copy()
        new_var_to_next_insert_index_list[indices_to_increase] = var_to_next_insert_index_list[indices_to_increase] + first_shift
        new_var_to_next_insert_index_list[indices_to_replace]  = first_index                                                     # by definition var_num is here
    else:
        new_indices_to_increase = [index for index in indices_to_increase if var_to_next_insert_index_list[index] > first_index] # for growing i this set will decrease
        indices_to_replace.extend(set(indices_to_increase) - set(new_indices_to_increase))
        indices_to_increase_2 = [index for index in new_indices_to_increase if var_to_next_insert_index_list[index] > second_index]
        indices_to_increase_1 = new_indices_to_increase

        new_var_to_next_insert_index_list = var_to_next_insert_index_list.copy()
        new_var_to_next_insert_index_list[indices_to_increase_1] = new_var_to_next_insert_index_list[indices_to_increase_1] + first_shift
        new_var_to_next_insert_index_list[indices_to_increase_2] = new_var_to_next_insert_index_list[indices_to_increase_2] + second_shift
        new_var_to_next_insert_index_list[indices_to_replace]  = first_index                                                # by definition var_num is here
    return (new_var_to_next_insert_index_list, new_indices_to_increase)

def _search_var_smart_multidim(stats:dict, sample:MultidimSample, supp:float, already_existing_tree:HyperLinkedTree, param_smart_matching:tuple|None=None) -> HyperLinkedTree:
    """
        Non-recursive iteration to add further variables, checking if they are still frequent and iterating the process until support condition can no longer be achieved.
        This approach uses a smart version and only mines pattern in normal form.
        In the case of hyperlinked trees, a current_vertex is required to store the found results.

        Args:
            stats: Dictionary for statistical evaluation.

            sample: Sample instance.

            supp: Float between 0 and 1 which describes the requested support.

            already_existing_tree: Optional parameter, which should be a hyperlinked tree. It to decide whether the result is stored there or in a list.

            param_smart_matching [= None]: requires a tuple (Dictionary, Variable set) to evaluate the queries on the sample

        Returns:
            HyperLinkedTree: Set of pattern-queries containing only variables and fitting support.

        Raises:
            None

        Passes:
            ValueError: from _check_frequency_of_query
    """
    # s-Extensions by temp_new_var_allowed from s-Extension
    event_dimension = sample._sample_event_dimension

    # setup
    next_element_queue = []

    # for dim in range(0, event_dimension):
    for dim in param_smart_matching[1]:
        single_event = dim*";" + "$x0" + (event_dimension-dim)*";"
        query_string = single_event + " " + single_event

        event_array = dim*[""] + ["$x0"] + (event_dimension-dim-1)*[""]
        query_array = [event_array, event_array.copy()]

        current_var_num  = 0
        s_0 = {dim: set() for dim in range(0, event_dimension)}
        s_0[dim].add(current_var_num)

        query = MultidimQuery(query_string)
        if not _check_frequency_of_query(stats, sample, supp, query, param_smart_matching):
            continue

        next_vertex = already_existing_tree.insert_query_string(already_existing_tree.get_root(), query_string, query_array, query, search_for_parents=True)
        next_vertex.query_next_insert_index = np.zeros(event_dimension, dtype=np.int8)
        next_vertex.query_next_insert_index[:dim+1] += np.ones(dim+1, dtype=np.int8)
        already_existing_tree.set_match_results(next_vertex, query._query_matched_traces)

        var_to_next_insert_index_list = np.array([2])

        next_element_queue.append((next_vertex, query_array, var_to_next_insert_index_list, s_0, s_0, current_var_num, True, True))

    while len(next_element_queue)>0:
        next_element = next_element_queue.pop(0)
        (current_vertex, query_array, var_to_next_insert_index_list, s_n_dict, s_same_dict, current_var_num, new_var_allowed_for_s_ext, new_var_allowed_for_i_ext) = next_element
        query_length = len(query_array)

    # s-extension
        if len(query_array) < MAX_QUERY_LENGTH:
            #insert already inserted variable again

            s_temp = {dim: set() for dim in range(0, event_dimension)}
            ready_to_queue_list = []
            for dim, s_n in enumerate(s_n_dict.values()):
                for var_num in s_n:
                    var_num_exists = False
                    indices_to_increase = list(range(0,len(var_to_next_insert_index_list)))      # s_n should be enough as well
                    indices_to_replace  = []
                    for i in range(max(current_vertex.query_next_insert_index[0],var_to_next_insert_index_list[var_num]),query_length+1):
                        single_event = dim*[""] + ['$x'+str(var_num)] + (event_dimension-dim-1)*[""]
                        new_query_array  = query_array[:i] + [single_event] + query_array[i:]
                        new_query_string = ' '.join([';'.join(event)+';' for event in new_query_array])

                        new_query = MultidimQuery(new_query_string)

                        if _check_frequency_of_query(stats, sample, supp, new_query, param_smart_matching):
                            var_num_exists = True

                            next_vertex = already_existing_tree.insert_query_string(current_vertex, new_query_string, new_query_array, new_query, search_for_parents=False)
                            next_vertex.query_next_insert_index = i*np.ones(event_dimension, dtype=np.int8)
                            next_vertex.query_next_insert_index[:dim+1] += np.ones(dim+1, dtype=np.int8)
                            already_existing_tree.set_match_results(next_vertex, new_query._query_matched_traces)

                            (new_var_to_next_insert_index_list, indicies_to_increase) = _calculate_index_shift(stats, var_to_next_insert_index_list, indices_to_increase,
                                    indices_to_replace, i)

                            ready_to_queue_list.append((next_vertex, new_query_array, new_var_to_next_insert_index_list, current_var_num, dim))
                    if var_num_exists:
                        s_temp[dim].add(var_num)
            #insert new variables
            if new_var_allowed_for_s_ext:
                temp_new_var_allowed_for_s_ext = False
                if len(query_array) < MAX_QUERY_LENGTH-1:
                    insert_symbol = '$x' + str(current_var_num + 1)

                    # for dim in range(0, event_dimension):
                    for dim in param_smart_matching[1]:
                        for i in range(current_vertex.query_next_insert_index[0], query_length+1):
                            indices_to_increase = list(range(0,len(var_to_next_insert_index_list)))        # s_n should be enough as well
                            indices_to_replace  = []
                            for j in range(i, query_length+1):
                                single_event = dim*[""] + [insert_symbol] + (event_dimension-dim-1)*[""]
                                new_query_array = query_array[:i] + [single_event] + query_array[i:j] + [single_event] + query_array[j:]
                                new_query_string = ' '.join([';'.join(event)+';' for event in new_query_array])

                                new_query = MultidimQuery(new_query_string)

                                if _check_frequency_of_query(stats, sample, supp, new_query, param_smart_matching):
                                    temp_new_var_allowed_for_s_ext = True

                                    next_vertex = already_existing_tree.insert_query_string(current_vertex, new_query_string, new_query_array, new_query, search_for_parents=True)
                                    next_vertex.query_next_insert_index =  i * np.ones(event_dimension, dtype=np.int8)
                                    next_vertex.query_next_insert_index[:dim+1] += np.ones(dim+1, dtype=np.int8)
                                    already_existing_tree.set_match_results(next_vertex, new_query._query_matched_traces)

                                    (new_var_to_next_insert_index_list, indicies_to_increase) = _calculate_index_shift(stats, var_to_next_insert_index_list, indices_to_increase,
                                            indices_to_replace, i, j)
                                    new_var_to_next_insert_index_list = np.append(new_var_to_next_insert_index_list, j+2)

                                    ready_to_queue_list.append((next_vertex, new_query_array, new_var_to_next_insert_index_list, current_var_num+1, dim))

            s_temp_extend = event_dimension*[False]
            s_temp_multidim = event_dimension*[s_temp]
            s_temp_same = deepcopy(s_temp)
            for dim, s_same_dim in s_same_dict.items():
                s_temp_same[dim] = s_temp_same[dim].union(s_same_dim)
            s_temp_same_multidim = event_dimension*[s_temp_same]
            temp_new_var_allowed_for_i_ext = new_var_allowed_for_i_ext or temp_new_var_allowed_for_s_ext
            for item in ready_to_queue_list:
                (t_next_vertex, t_new_query_array, t_new_var_to_next_insert_index_list, t_new_var_num, t_last_edited_dim) = item
                if current_var_num != t_new_var_num and not s_temp_extend[t_last_edited_dim]:
                    s_temp_multidim[t_last_edited_dim] = deepcopy(s_temp)
                    s_temp_multidim[t_last_edited_dim][t_last_edited_dim].add(current_var_num+1)
                    s_temp_extend[t_last_edited_dim] = True

                    s_temp_same_multidim[t_last_edited_dim] = deepcopy(s_temp_same)
                    s_temp_same_multidim[t_last_edited_dim][t_last_edited_dim].add(current_var_num+1)
                next_element_queue.append((t_next_vertex, t_new_query_array, t_new_var_to_next_insert_index_list.copy(), deepcopy(s_temp_multidim[t_last_edited_dim]),
                    deepcopy(s_temp_same_multidim[t_last_edited_dim]), t_new_var_num, temp_new_var_allowed_for_s_ext, temp_new_var_allowed_for_i_ext))
        else:
            s_temp = {dim: set() for dim in range(0, event_dimension)}
            s_temp_extend = event_dimension*[False]
            s_temp_multidim = event_dimension*[s_temp]
            temp_new_var_allowed_for_s_ext = False

    # i-extension
        #insert already inserted variable again
        s_temp_same = {dim: set() for dim in range(0, event_dimension)}
        ready_to_queue_list = []
        for dim, s_same in enumerate(s_same_dict.values()):
            for var_num in s_same:
                var_num_exists = False
                indices_to_increase = list(range(0,len(var_to_next_insert_index_list)))      # s_n should be enough as well
                indices_to_replace  = []
                for i in range(max(current_vertex.query_next_insert_index[dim],var_to_next_insert_index_list[var_num]),query_length):
                    if not query_array[i][dim] == "":
                        continue
                    single_event = deepcopy(query_array[i])
                    single_event[dim] = '$x'+str(var_num)
                    new_query_array  = query_array[:i] + [single_event] + query_array[i+1:]
                    new_query_string = ' '.join([';'.join(event)+';' for event in new_query_array])

                    new_query = MultidimQuery(new_query_string)
                    if _check_frequency_of_query(stats, sample, supp, new_query, param_smart_matching):
                        var_num_exists = True

                        next_vertex = already_existing_tree.insert_query_string(current_vertex, new_query_string, new_query_array, new_query, search_for_parents=False)
                        next_vertex.query_next_insert_index = i*np.ones(event_dimension, dtype=np.int8)
                        next_vertex.query_next_insert_index[:dim+1] += np.ones(dim+1, dtype=np.int8)
                        already_existing_tree.set_match_results(next_vertex, new_query._query_matched_traces)

                        (new_var_to_next_insert_index_list, indicies_to_increase) = _calculate_index_shift(stats, var_to_next_insert_index_list, indices_to_increase,
                                indices_to_replace, i, None, 0)

                        ready_to_queue_list.append((next_vertex, new_query_array, new_var_to_next_insert_index_list, current_var_num, dim))
                if var_num_exists:
                    s_temp_same[dim].add(var_num)

        #insert new variables
        if new_var_allowed_for_i_ext:
            temp_new_var_allowed = False
            insert_symbol = '$x' + str(current_var_num + 1)

            # complete i-extention
            # for dim in range(0, event_dimension):
            for dim in param_smart_matching[1]:
                for i in range(current_vertex.query_next_insert_index[dim], query_length):                       # always take NVIIL[dim] for i-extension
                    if not query_array[i][dim] == "":
                        continue
                    indices_to_increase_1 = list(range(0,len(var_to_next_insert_index_list)))        # s_n should be enough as well
                    indices_to_replace  = []
                    for j in range(i+1, query_length):
                        if not query_array[j][dim] == "":
                            continue
                        single_event_i = deepcopy(query_array[i])
                        single_event_j = deepcopy(query_array[j])
                        single_event_i[dim] = insert_symbol
                        single_event_j[dim] = insert_symbol

                        new_query_array = query_array[:i] + [single_event_i] + query_array[i+1:j] + [single_event_j] + query_array[j+1:]
                        new_query_string = ' '.join([';'.join(event)+';' for event in new_query_array])

                        new_query = MultidimQuery(new_query_string)

                        if _check_frequency_of_query(stats, sample, supp, new_query, param_smart_matching):
                            temp_new_var_allowed = True

                            next_vertex = already_existing_tree.insert_query_string(current_vertex, new_query_string, new_query_array, new_query, search_for_parents=False)
                            next_vertex.query_next_insert_index =  i * np.ones(event_dimension, dtype=np.int8)
                            next_vertex.query_next_insert_index[:dim+1] += np.ones(dim+1, dtype=np.int8)
                            already_existing_tree.set_match_results(next_vertex, new_query._query_matched_traces)

                            (new_var_to_next_insert_index_list, indicies_to_increase) = _calculate_index_shift(stats, var_to_next_insert_index_list, indices_to_increase,
                                    indices_to_replace, i, None, 0)
                            new_var_to_next_insert_index_list = np.append(new_var_to_next_insert_index_list, j+1)

                            ready_to_queue_list.append((next_vertex, new_query_array, new_var_to_next_insert_index_list, current_var_num+1, dim))

            # mixed i-extention - first index
            if len(query_array) < MAX_QUERY_LENGTH:
                # for dim in range(0, event_dimension):
                for dim in param_smart_matching[1]:
                    for i in range(current_vertex.query_next_insert_index[dim],query_length):
                        if not query_array[i][dim] == "":
                            continue
                        indices_to_increase = list(range(0,len(var_to_next_insert_index_list)))        # s_n should be enough as well
                        indices_to_replace  = []
                        for j in range(i+1, query_length+1):
                            single_event_i = deepcopy(query_array[i])
                            single_event_j = event_dimension*[""]
                            single_event_i[dim] = insert_symbol
                            single_event_j[dim] = insert_symbol

                            new_query_array = query_array[:i] + [single_event_i] + query_array[i+1:j] + [single_event_j] + query_array[j:]
                            new_query_string = ' '.join([';'.join(event)+';' for event in new_query_array])

                            new_query = MultidimQuery(new_query_string)

                            if _check_frequency_of_query(stats, sample, supp, new_query, param_smart_matching):
                                temp_new_var_allowed = True

                                next_vertex = already_existing_tree.insert_query_string(current_vertex, new_query_string, new_query_array, new_query, search_for_parents=False)
                                next_vertex.query_next_insert_index =  i * np.ones(event_dimension, dtype=np.int8)
                                next_vertex.query_next_insert_index[:dim+1] += np.ones(dim+1, dtype=np.int8)
                                already_existing_tree.set_match_results(next_vertex, new_query._query_matched_traces)

                                (new_var_to_next_insert_index_list, indicies_to_increase) = _calculate_index_shift(stats, var_to_next_insert_index_list, indices_to_increase,
                                        indices_to_replace, i, j, 0, 1)
                                new_var_to_next_insert_index_list = np.append(new_var_to_next_insert_index_list, j+1)

                                ready_to_queue_list.append((next_vertex, new_query_array, new_var_to_next_insert_index_list, current_var_num+1, dim))

            # mixed i-extention - second index
                # for dim in range(0, event_dimension):
                for dim in param_smart_matching[1]:
                    for i in range(current_vertex.query_next_insert_index[0], query_length):                         # always take NVIIL[0] for i-extension
                        indices_to_increase = list(range(0,len(var_to_next_insert_index_list)))        # s_n should be enough as well
                        indices_to_replace  = []
                        for j in range(i, query_length):
                            if not query_array[j][dim] == "":
                                continue
                            single_event_i = event_dimension*[""]
                            single_event_j = deepcopy(query_array[j])
                            single_event_i[dim] = insert_symbol
                            single_event_j[dim] = insert_symbol

                            new_query_array = query_array[:i] + [single_event_i] + query_array[i:j] + [single_event_j] + query_array[j+1:]
                            new_query_string = ' '.join([';'.join(event)+';' for event in new_query_array])

                            new_query = MultidimQuery(new_query_string)

                            if _check_frequency_of_query(stats, sample, supp, new_query, param_smart_matching):
                                temp_new_var_allowed = True

                                next_vertex = already_existing_tree.insert_query_string(current_vertex, new_query_string, new_query_array, new_query, search_for_parents=False)
                                next_vertex.query_next_insert_index =  i * np.ones(event_dimension, dtype=np.int8)
                                next_vertex.query_next_insert_index[:dim+1] += np.ones(dim+1, dtype=np.int8)
                                already_existing_tree.set_match_results(next_vertex, new_query._query_matched_traces)

                                (new_var_to_next_insert_index_list, indicies_to_increase) = _calculate_index_shift(stats, var_to_next_insert_index_list, indices_to_increase,
                                        indices_to_replace, i, j, 1, 0)
                                new_var_to_next_insert_index_list = np.append(new_var_to_next_insert_index_list, j+2)

                                ready_to_queue_list.append((next_vertex, new_query_array, new_var_to_next_insert_index_list, current_var_num+1, dim))

        s_temp_extend = event_dimension*[False]
        s_temp_multidim = event_dimension*[s_temp]
        s_temp_same_multidim = event_dimension*[s_temp_same]
        s_and_i_temp = deepcopy(s_temp)
        for dim in s_temp_same:
            s_and_i_temp[dim].update(s_temp_same[dim])
        s_and_i_temp_multidim = event_dimension*[s_and_i_temp]
        for item in ready_to_queue_list:
            (next_vertex, new_query_array, new_var_to_next_insert_index_list, new_var_num, last_edited_dim) = item
            if current_var_num != new_var_num:
                if not s_temp_extend[last_edited_dim]:
                    s_temp_multidim[last_edited_dim] = deepcopy(s_temp)
                    s_temp_multidim[last_edited_dim][last_edited_dim].add(current_var_num+1)
                    s_temp_extend[last_edited_dim] = True

                    s_temp_same_multidim[last_edited_dim] = deepcopy(s_temp_same)
                    s_temp_same_multidim[last_edited_dim][last_edited_dim].add(current_var_num+1)

                    s_and_i_temp_multidim[last_edited_dim] = deepcopy(s_and_i_temp)
                    s_and_i_temp_multidim[last_edited_dim][last_edited_dim].add(current_var_num+1)
                next_element_queue.append((next_vertex, new_query_array, new_var_to_next_insert_index_list.copy(), deepcopy(s_temp_multidim[last_edited_dim]),
                    deepcopy(s_and_i_temp_multidim[last_edited_dim]), new_var_num, temp_new_var_allowed_for_s_ext, temp_new_var_allowed))
            else:
                next_element_queue.append((next_vertex, new_query_array, new_var_to_next_insert_index_list.copy(), deepcopy(s_temp_multidim[last_edited_dim]),
                    deepcopy(s_temp_same_multidim[last_edited_dim]), new_var_num, temp_new_var_allowed_for_s_ext, temp_new_var_allowed))

def _build_pattern_tree_multidim(stats:dict, sample:MultidimSample, supp:float, use_tree_structure:bool=False, param_smart_matching:tuple|None=None) -> HyperLinkedTree|set:
    """
        Builds a set with all queries containing only variables and fitting the sample with the given support.

        Args:
            stats: Dictionary for statistical evaluation.

            sample: Sample instance.

            supp: Float between 0 and 1 which describes the requested support.

            use_tree_structure [= False]: Optional parameter to decide whether the result should be stored in a hyperlinked tree or not.

            param_smart_matching [= None]: requires a tuple (Dictionary, Variable set) to evaluate the queries on the sample

        Returns:
            Set of all pattern-queries fitting the support.

        Raises:
            None

        Passes:
            ValueError: from _search_var
            ValueError: from _search_var_smart
    """
    complete_vsdb = sample.get_att_vertical_sequence_database()
    if param_smart_matching:
        if not param_smart_matching[1]:
            # all_patternset = {}
            for (i, single_dim_vsdb) in complete_vsdb.items():
                # param_smart_matching[1][i] = {key for key in single_dim_vsdb 
                #                               for item in single_dim_vsdb[key].keys() 
                #                               if len(single_dim_vsdb[key][item]) >= 2}
                param_smart_matching[1][i] = {trace_id: set() for trace_id in range(len(sample._sample))}
                for letter, pos_dict in single_dim_vsdb.items():
                    for trace_id, positions in pos_dict.items():
                        # if len(value[item]) >= 2:
                        if len(positions) >=2:
                            param_smart_matching[1][i][trace_id].add(letter)
            # param_smart_matching = (param_smart_matching[0], all_patternset, param_smart_matching[2])

    pattern_queries = None
    if use_tree_structure:
        pattern_queries = HyperLinkedTree(ceil(supp*sample._sample_size), sample._sample_event_dimension)
        _search_var_smart_multidim(stats, sample, supp, pattern_queries, param_smart_matching=param_smart_matching)
    else:
        pattern_queries = {''}
        if len([trace for trace in sample._sample if len(trace) == 0]) < len(sample._sample)*supp:
            allow_new_variables = [True]*sample._sample_event_dimension
            s_0 = {dim : set() for dim in range(0,sample._sample_event_dimension)}
            for dim in range(0,sample._sample_event_dimension):
                new_event = [""]*sample._sample_event_dimension
                new_event[dim] = "$x0"
                new_pattern = ";".join(new_event)+";"
                pattern_queries.add(new_pattern)
                s_0_copy = deepcopy(s_0)
                s_0_copy[dim].add("$x0")
                pattern_queries |= _search_var_multidim(sample, 1, "", new_event, dim+1, s_0_copy, s_0_copy, supp, stats, allow_new_variables.copy(),
                        param_smart_matching=param_smart_matching)

        # normalize
        pattern_queries_normalform = set()
        for query_string in pattern_queries:
            new_query = MultidimQuery()
            new_query.set_query_string(query_string)
            new_query.query_string_to_normalform()

            pattern_queries_normalform.add(new_query._query_string)
        pattern_queries = pattern_queries_normalform

    return pattern_queries

def _empty_merge(mixed_query_tree:HyperLinkedTree, given_vertex:Vertex, evolved_vertex:Vertex) -> Vertex:
    """
        Helper function to fastly move a given vertex to the new mixed_query_tree, if its merge partner is empty.

        Args:
            mixed_query_tree: The HyperLinkedTree containing all merged queries. It stores the results.

            given_vertex: Is the parent vertex to the given evolved_vertex. It gets one end of the link.

            evolved_vertex: Is the new vertex, which shall be inserted.

        Returns:
            Vertex: The new vertex created in the mixed_query_tree by inserting evolved_vertex in it.

        Raises:
            None

        Passes:
            None
    """
    new_vertex = mixed_query_tree.insert_query_string(given_vertex, deepcopy(evolved_vertex.query_string), deepcopy(evolved_vertex.query_array), deepcopy(evolved_vertex.query), search_for_parents=True,
            break_when_missing_parent=True, break_when_non_matching_parent=True)
    new_vertex.query_next_insert_index = deepcopy(evolved_vertex.query_next_insert_index)
    mixed_query_tree.set_match_results(new_vertex, evolved_vertex.matched_traces)
    return new_vertex

def _merge_event_arrays(given_event_array:list, to_insert_event_array:list, to_insert_event_is_type_event:bool, merge_from_index:int=0, allow_mixed_merge:bool=False,
        allow_update:bool=True, obey_construction_order:bool=True, force_mixed_merge:bool=False) -> list|None:
    """
        Determines whether to_insert_event_array can be merged with given_event_array by fulfilling the constrains or not.

        Args:
            given_event_array: A list-represention of an arbitrary event.

            to_insert_event_array: A list-represention of an event. The function evaluates whether it can be merged in to given_event_array or not, and if, how the merged event
                looks like.

            to_insert_event_is_type_event: Boolean representing whether to_insert_event_array is a singleton type (True) or not (False).

            merge_from_index [= 0]: Normally merges the event from first (0) to last index. But if already know, that nothing of interest happens in the first m-1 positions, the
                merge can begin at index m.

            allow_mixed_merge: [= False]: In most cases the merge violates the discovery rules, if types and variables are merged into the given event, but there might special
                cases, where it is a desirable behavior.

            allow_update [= False]: The function only merges types and variables into empty entries (= False). But especially with later insersions it might be desirable to merge
                only the empty slots whereas occupied entries must have the same entry as in both lists (= True).

            obey_construction_order [= True]: If the merge violates the construction order, the merge will be rejected (= True). In special cases this might be a desired behavior
                and the merge will be proceeded as excpected (= False).

            force_mixed_merge [= False]: Rejects the merge, if no mixed merge occurs (= True).

        Returns:
            list: The merged event arrays if they satisfy the given constrains.
            None: If the event arrays don't satisfy the given constrains.

        Raises:
            AssertionError: Raised, if given_event_array and to_insert_event_array don't have the same length.

        Passes:
            None
    """
    assert len(given_event_array) == len(to_insert_event_array)

    new_array = []
    single_diviation = False
    in_construction_order = True
    found_type_entry = False
    #found_variable_entry = False

    for dim, value in enumerate(given_event_array):
        if not allow_mixed_merge and value != "":
            if (value[0] == "$") != (to_insert_event_is_type_event):
                return None

        if value == "":
            if to_insert_event_array[dim] == "":
                new_array.append(value)
            elif dim >= merge_from_index:
                if single_diviation is False:
                    single_diviation = True
                else:
                    return None
                new_array.append(to_insert_event_array[dim])
                in_construction_order = False
            else:
                return None
        else:
            if force_mixed_merge:
                if value[0] != '$':
                    found_type_entry = True
                #else:
                #    found_variable_entry = True
            if allow_update and value == to_insert_event_array[dim]:
                new_array.append(value)
            elif (not obey_construction_order or in_construction_order):
                if to_insert_event_array[dim] == "":
                    new_array.append(value)
                else:
                    return None
            else:
                return None

    if force_mixed_merge:
        if not found_type_entry:# or (not found_variable_entry):
            return None
    return new_array

def _merge_query_vertices_multidim(mixed_query_tree:HyperLinkedTree, t_vertex:Vertex, p_vertex:Vertex, m_vertex:Vertex, evolved_vertex:Vertex,
        evolved_vertex_contains_type_query:bool) -> list:
    """
        The method creates a list of new vertices for mixed-query-vertex, when either the t(ype)-vertex or the p(attern)-vertex evolved (was increased).
        The goal is to create a spanning tree by creating each possible mixed-query only once.

        Args:
            mixed_query_tree: A hyperlinked tree to store the results to.

            t_vertex: the current type_vertex.

            p_vertex: the current pattern_vertex.

            m_vertex: the vertex contains the current mixed query of the t_vertex and the p_vertex in a specific manner.

            evolved_vertex: the vertex, which got one or two more symbols than either the t_vertex or p_vertex. The vertex will be used to create a child from m_vertex and store
                it in mixed_query_tree.

            evolved_vertex_contains_type_query: Provides the information whether the evolved_vertex contains a type query (= True) or a pattern query (= False).

        Returns:
            list: A list of newly found queries

        Raises:
            IndexError: If t_index and p_index run out of bound, while iterating and comparing the type_string and the pattern_string with the mixed_query_string, and an
                "Undefined behavior" is raised, since this should normally never happen.

        Passes:
            AssertionError from _merge_event_arrays(...)
            TypeError from _syntactically_contained(...)
            ValueError from _syntactically_contained(...)
    """
    t_array = t_vertex.query_array
    t_index = 0
    t_length = len(t_array)
    if t_length == 0 and not evolved_vertex_contains_type_query:
        return [_empty_merge(mixed_query_tree, m_vertex, evolved_vertex)]
    p_array = p_vertex.query_array
    p_index = 0
    p_length = len(p_array)
    if p_length == 0 and evolved_vertex_contains_type_query:
        return [_empty_merge(mixed_query_tree, m_vertex, evolved_vertex)]
    m_array = m_vertex.query_array
    m_index = 0
    m_next_insert_index = m_vertex.query_next_insert_index
    e_array = evolved_vertex.query_array
    e_next_insert_index = evolved_vertex.query_next_insert_index
    e_length = len(e_array)
    mixed_indicies = 0

    while m_index < m_next_insert_index[0]:
        t_inc_possible = False
        if t_index < t_length:
            t_inc_possible = True
            if m_array[m_index] == t_array[t_index]:
                if evolved_vertex_contains_type_query and t_array[t_index] != e_array[t_index]:
                    if m_index < m_next_insert_index[-1]: # diviation before m_next_insert_index will result in no possible queries
                        return []
                    break
                t_index += 1
                m_index += 1
                continue
        if p_index < p_length:
            if m_array[m_index] == p_array[p_index]:
                if not evolved_vertex_contains_type_query and p_array[p_index] != e_array[p_index]:
                    if m_index < m_next_insert_index[-1]: # diviation before m_next_insert_index will result in no possible queries
                        return []
                    break
                p_index += 1
                m_index += 1
                continue
            if t_inc_possible:
                if not evolved_vertex_contains_type_query and p_array[p_index] != e_array[p_index]:
                    if m_index < m_next_insert_index[-1]: # diviation before m_next_insert_index will result in no possible queries
                        return []
                    if m_index == m_next_insert_index[-1]: # diviation in m_next_insert_index will result in no possible queries
                        if not _syntactically_contained([p_array[p_index]], [e_array[p_index]]):
                            return []
                    break
                for dim, item in enumerate(m_array[m_index]):
                    if item == "":
                        if t_array[t_index][dim] == "" and p_array[p_index][dim] == "":
                            continue
                        raise IndexError("Undefined behavior 1")
                    if item == t_array[t_index][dim] and p_array[p_index][dim] == "":
                        pass
                    elif item == p_array[p_index][dim] and t_array[t_index][dim] == "":
                        pass
                    else:
                        # LOGGER.debug("t_vertex: %s", t_vertex.query_string)
                        # LOGGER.debug("p_vertex: %s", p_vertex.query_string)
                        # LOGGER.debug("m_vertex: %s", m_vertex.query_string)
                        # LOGGER.debug("e_vertex: %s", evolved_vertex.query_string)
                        # LOGGER.debug("t: %s / %s", t_index, t_length)
                        # LOGGER.debug("p: %s / %s", p_index, p_length)
                        # LOGGER.debug("m: %s / ?", m_index)
                        raise IndexError("Undefined behavior 2")
                p_index += 1
                t_index += 1
                mixed_indicies += 1
                m_index += 1
            else:
                raise IndexError("Undefined behavior 3")
        else:
            raise IndexError("Undefined behavior 4")

    new_vertices = []
    if evolved_vertex_contains_type_query:
        if e_length != t_length: # true: additional event in e_array
            # insert type
            if len(m_array) < MAX_QUERY_LENGTH:
                for i in range(m_next_insert_index[0], len(m_array) + 1):
                    new_query_array = m_array[:i]
                    new_query_array += [evolved_vertex.query_array[-1]]
                    new_query_array += m_array[i:]
                    new_query_event_list = [';'.join(event_array)+";" for event_array in new_query_array]
                    new_query_string = ' '.join(new_query_event_list)
                    new_vertex = mixed_query_tree.insert_query_string(m_vertex, new_query_string, new_query_array, search_for_parents=True, break_when_missing_parent=True,
                            break_when_non_matching_parent=True)
                    new_vertex.query_next_insert_index = e_next_insert_index + (p_index - mixed_indicies + (i - m_next_insert_index[0]))
                    new_vertices.append(new_vertex)
            # insert type + merge
            if m_next_insert_index[0] == m_next_insert_index[-1]+1:
                new_query_array = m_array[:m_next_insert_index[-1]]
                merged_event = _merge_event_arrays(m_array[m_next_insert_index[-1]], evolved_vertex.query_array[-1], evolved_vertex_contains_type_query,
                        merge_from_index=np.where(m_next_insert_index[:-1] != m_next_insert_index[1:])[0], allow_mixed_merge=False, allow_update=True,
                        obey_construction_order=False)
                if merged_event is not None:
                    new_query_array += [merged_event]
                    new_query_array += m_array[m_next_insert_index[0]:]
                    new_query_event_list = [';'.join(event_array)+";" for event_array in new_query_array]
                    new_query_string = ' '.join(new_query_event_list)

                    new_vertex = mixed_query_tree.insert_query_string(m_vertex, new_query_string, new_query_array, search_for_parents=True, break_when_missing_parent=True,
                            break_when_non_matching_parent=True)
                    difference = (e_next_insert_index - e_next_insert_index[0]+1) - (m_next_insert_index - m_next_insert_index[0]+1)
                    if (difference > 0).sum() > 0:
                        new_vertex.query_next_insert_index = e_next_insert_index + (p_index - mixed_indicies - 1)#+ (m_next_insert_index[-1] - m_next_insert_index[0] + 1)) = ...+0
                    else:
                        new_vertex.query_next_insert_index = m_next_insert_index #+ (m_next_insert_index[-1] - m_next_insert_index[0] + 1) = ... + 0
                    new_vertices.append(new_vertex)
            for i in range(m_next_insert_index[0], len(m_array)):
                new_query_array = m_array[:i]
                merged_event = _merge_event_arrays(m_array[i], evolved_vertex.query_array[-1], evolved_vertex_contains_type_query,
                        merge_from_index=0, allow_mixed_merge=False, allow_update=True, obey_construction_order=False)
                if merged_event is None:
                    continue
                new_query_array += [merged_event]
                new_query_array += m_array[i+1:]
                new_query_event_list = [';'.join(event_array)+";" for event_array in new_query_array]
                new_query_string = ' '.join(new_query_event_list)

                new_vertex = mixed_query_tree.insert_query_string(m_vertex, new_query_string, new_query_array, search_for_parents=True, break_when_missing_parent=True,
                        break_when_non_matching_parent=True)
                new_vertex.query_next_insert_index = e_next_insert_index - e_next_insert_index[0] + i+1
                new_vertices.append(new_vertex)
        else:
            difference = (e_next_insert_index - e_next_insert_index[0]+1) - (m_next_insert_index - m_next_insert_index[0]+1)
            if (difference < 0).sum() > 0:
                return []
            new_query_array = m_array[:m_next_insert_index[0]-1]
            merged_event = _merge_event_arrays(m_array[m_next_insert_index[0]-1], e_array[-1], evolved_vertex_contains_type_query, merge_from_index=0, allow_mixed_merge=True,
                     allow_update=True, obey_construction_order=False)
            if merged_event is None:
                return []
            new_query_array += [merged_event]
            new_query_array += m_array[m_next_insert_index[0]:]
            new_query_event_list = [';'.join(event_array)+";" for event_array in new_query_array]
            new_query_string = ' '.join(new_query_event_list)
            new_vertex = mixed_query_tree.insert_query_string(m_vertex, new_query_string, new_query_array, search_for_parents=True, break_when_missing_parent=True,
                    break_when_non_matching_parent=True)
            new_vertex.query_next_insert_index = e_next_insert_index + (p_index - mixed_indicies)
            new_vertices.append(new_vertex)

    else:
        if p_index == p_length:
            # insert pattern
            if len(m_array) < MAX_QUERY_LENGTH:
                new_query_array = m_array[:m_index] #m_next_inser_index might be not big enough and must be replaced by m_index
                new_query_array += e_array[p_index:]
                new_query_event_list = [';'.join(event_array)+";" for event_array in new_query_array]
                new_query_string = ' '.join(new_query_event_list)
                new_vertex = mixed_query_tree.insert_query_string(m_vertex, new_query_string, new_query_array, search_for_parents=True, break_when_missing_parent=True,
                        break_when_non_matching_parent=True)
                new_vertex.query_next_insert_index = e_next_insert_index + (t_index - mixed_indicies)
                new_vertices.append(new_vertex)
            if m_next_insert_index[0] == m_next_insert_index[-1]+1:
                new_query_array = m_array[:m_next_insert_index[-1]]
                merged_event = _merge_event_arrays(m_array[m_next_insert_index[-1]], evolved_vertex.query_array[-1], evolved_vertex_contains_type_query,
                        merge_from_index=np.where(m_next_insert_index[:-1] != m_next_insert_index[1:])[0], allow_mixed_merge=False,
                        allow_update=False, obey_construction_order=True)
                if merged_event is not None:
                    new_query_array += [merged_event]
                    new_query_array += e_array[p_index+1:]
                    if len(new_query_array) <= MAX_QUERY_LENGTH:
                        new_query_event_list = [';'.join(event_array)+";" for event_array in new_query_array]
                        new_query_string = ' '.join(new_query_event_list)
                        new_vertex = mixed_query_tree.insert_query_string(m_vertex, new_query_string, new_query_array, search_for_parents=True, break_when_missing_parent=False,
                                break_when_non_matching_parent=False)
                        difference = (e_next_insert_index - e_next_insert_index[0]+1) - (m_next_insert_index - m_next_insert_index[0]+1)
                        if (difference > 0).sum() > 0:
                            new_vertex.query_next_insert_index = e_next_insert_index + (t_index - mixed_indicies - 1)# + (m_next_insert_index[-1]-m_next_insert_index[0]+1))=...+0
                        else:
                            new_vertex.query_next_insert_index = m_next_insert_index #+ (m_next_insert_index[-1] - m_next_insert_index[0] + 1) = ... + 0
                        new_vertices.append(new_vertex)
        else:
            new_query_array = m_array[:m_index] #m_next_insert_index might be not big enough and must be replaced by m_index
            if t_index == t_length: # only if all entries from t_query had been found -> otherwise some overwriting happens
                new_query_array += e_array[p_index:]
                if len(new_query_array) <= MAX_QUERY_LENGTH:
                    new_query_event_list = [';'.join(event_array)+";" for event_array in new_query_array]
                    new_query_string = ' '.join(new_query_event_list)
                    new_vertex = mixed_query_tree.insert_query_string(m_vertex, new_query_string, new_query_array, search_for_parents=True, break_when_missing_parent=True,
                            break_when_non_matching_parent=True)
                    new_vertex.query_next_insert_index = e_next_insert_index + (t_index - mixed_indicies)
                    new_vertices.append(new_vertex)

                p_containted_in_m = _syntactically_contained_event(p_array[p_index-1], m_array[m_index-1],{})[0]
                merge_is_reasonable = False
                if not p_containted_in_m:
                    merge_is_reasonable = True
                else:
                    p_containted_in_e = _syntactically_contained_event(p_array[p_index-1], e_array[p_index-1],{})[0]
                    if p_containted_in_e :
                        if not p_array[p_index-1] == e_array[p_index-1] or e_array[p_index-1] == e_array[p_index]:
                            merge_is_reasonable = True
                if merge_is_reasonable:
                    if not (m_array[m_index-1] == t_array[t_index-1] and m_array[m_index] == p_array[p_index] and p_array[p_index] == e_array[p_index]):
                        merged_event = _merge_event_arrays(m_array[m_index-1], e_array[p_index], evolved_vertex_contains_type_query, merge_from_index=0, allow_mixed_merge=True,
                                allow_update=False, obey_construction_order=True, force_mixed_merge=True)
                        if merged_event is not None:
                            new_query_array = m_array[:m_index-1] #m_next_insert_index might be not big enough and must be replaced by m_index
                            new_query_array += [merged_event]
                            new_query_array += e_array[p_index+1:]
                            if len(new_query_array) <= MAX_QUERY_LENGTH:
                                new_query_event_list = [';'.join(event_array)+";" for event_array in new_query_array]
                                new_query_string = ' '.join(new_query_event_list)
                                new_vertex = mixed_query_tree.insert_query_string(m_vertex, new_query_string, new_query_array, search_for_parents=True,
                                        break_when_missing_parent=True, break_when_non_matching_parent=True)
                                new_vertex.query_next_insert_index = e_next_insert_index + (t_index - mixed_indicies - 1)
                                new_vertices.append(new_vertex)

            else:
                new_query_array = m_array[:m_index] #m_next_insert_index might be not big enough and must be replaced by m_index
                if m_index >= m_next_insert_index[0]:
                    merged_event = _merge_event_arrays(m_array[m_index], e_array[p_index], evolved_vertex_contains_type_query, merge_from_index=0, allow_mixed_merge=True,
                        allow_update=False, obey_construction_order=True)
                else: # m_index == m_next_insert_index (by break condition in while loop)
                    merge_begin = len(m_next_insert_index) - 1
                    while m_next_insert_index[merge_begin] == m_next_insert_index[merge_begin - 1]: #can't be -1 by if condition
                        merge_begin -= 1
                    merged_event = _merge_event_arrays(m_array[m_index], e_array[p_index], evolved_vertex_contains_type_query, merge_from_index=merge_begin, allow_mixed_merge=True,
                        allow_update=True, obey_construction_order=True)
                if merged_event is not None:
                    new_query_array += [merged_event]
                    new_query_array += e_array[p_index+1:]
                    if len(new_query_array) <= MAX_QUERY_LENGTH:
                        new_query_event_list = [';'.join(event_array)+";" for event_array in new_query_array]
                        new_query_string = ' '.join(new_query_event_list)
                        new_vertex = mixed_query_tree.insert_query_string(m_vertex, new_query_string, new_query_array, search_for_parents=True, break_when_missing_parent=True,
                                break_when_non_matching_parent=True)
                        new_vertex.query_next_insert_index = e_next_insert_index + (t_index - mixed_indicies)
                        new_vertices.append(new_vertex)

    return new_vertices

def _build_mixed_query_tree_multidim(stats:dict, sample:MultidimSample, supp:float, type_tree:HyperLinkedTree|set, pattern_tree:HyperLinkedTree|set, param_smart_matching:tuple|None=None,
        find_descriptive_only:bool=True) -> HyperLinkedTree|set:
    """
        Construct all mixed queries from the set of all type-queries and all pattern-queries fitting the support requirements.

        Args:
            stats: Dictionary for statistical evaluation.

            sample: Sample instance.

            supp: Float between 0 and 1 which describes the requested support.

            type_tree: list of all type-queries.

            pattern_tree: list of all pattern-queries.

            param_smart_matching [= None]: requires a tuple (Dictionary, Variable set) to evaluate the queries on the sample

            find_descriptive_only [= True]: If the optional parameter is "True", all matching queries are internally equipped with the "descriptive" attribute.

        Returns:
            set: A set of all queries fitting the sample.
            HyperLinkedTree: A tree_structure is used, when both the type_tree and pattern_tree are HyperLinkedTrees.

        Raises:
            AssertionError: if type_tree and pattern_tree are not of the same type.

        Passes:
            IndexError from _merge_query_vertices(...)
            TypeError from _merge_query_vertices(...)
            ValueError from matching_smarter(...)
            ValueError from _merge_query_vertices(...)
    """
    mixed_queries = {''}

    assert isinstance(type_tree, type(pattern_tree))
    if not isinstance(type_tree, HyperLinkedTree):
        mixed_queries.update(type_tree)
        mixed_queries.update(pattern_tree)

        splitted_type_queries = []
        splitted_pattern_queries = []
        empty_query_string = ''
        for query_string in type_tree:
            if not query_string == empty_query_string:
                splitted_type_queries.append([event.split(";") for event in query_string.split()])
        for query_string in pattern_tree:
            if not query_string == empty_query_string:
                splitted_pattern_queries.append([event.split(";") for event in query_string.split()])

        for splitted_type_query in splitted_type_queries:
            for splitted_pattern_query in splitted_pattern_queries:
                combination = combine_all(splitted_type_query, splitted_pattern_query)

                for item in combination:
                    query_string = str(";".join(item[0]))

                    for i in range(1, len(item)):
                        query_string += ' ' + str(";".join(item[i]))

                    new_query = MultidimQuery()
                    new_query.set_query_string(query_string)

                    #query_is_frequent = new_query.match_sample(sample, supp)
                    if not param_smart_matching:
                        query_is_frequent = new_query.match_sample(sample, supp)
                    else:
                        new_query.set_query_matchtest('smarter')
                        query_is_frequent = new_query.match_sample(sample, supp, dict_iter= param_smart_matching[0], patternset=param_smart_matching[1],
                                parent_dict=param_smart_matching[2], max_query_length=MAX_QUERY_LENGTH)

                    if query_is_frequent:
                        new_query.query_string_to_normalform()
                        mixed_queries.add(new_query._query_string)

                    event1_is_type = None
                    event2_is_type = None
                    for value in item[0]:
                        if value == '':
                            continue
                        if value[0] == '$':
                            event1_is_type = False
                        else:
                            event1_is_type = True
                        break

                    new_mixed_query = [item[0]]
                    anything_merged = False
                    for i in range(1, len(item)):
                        for value in item[i]:
                            if value == '':
                                continue
                            if value[0] == '$':
                                event2_is_type = False
                            else:
                                event2_is_type = True
                            break
                        if event1_is_type and not event2_is_type:
                            merged_event = merge_event_arrays(item[i-1],item[i])
                            event1_is_type = False

                            if not merged_event:
                                new_mixed_query.append(item[i])
                                continue
                            new_mixed_query[-1] = merged_event
                            anything_merged = True
                        else:
                            new_mixed_query.append(item[i])
                            event1_is_type = event2_is_type
                    if anything_merged:
                        query_string = str(";".join(new_mixed_query[0]))

                        for i in range(1, len(new_mixed_query)):
                            query_string += ' ' + str(";".join(new_mixed_query[i]))

                        new_query = MultidimQuery()
                        new_query.set_query_string(query_string)

                        #query_is_frequent = new_query.match_sample(sample, supp)
                        if not param_smart_matching:
                            query_is_frequent = new_query.match_sample(sample, supp)
                        else:
                            new_query.set_query_matchtest('smarter')
                            query_is_frequent = new_query.match_sample(sample, supp, dict_iter= param_smart_matching[0], patternset=param_smart_matching[1],
                                    parent_dict=param_smart_matching[2], max_query_length=MAX_QUERY_LENGTH)

                        if query_is_frequent:
                            new_query.query_string_to_normalform()
                            mixed_queries.add(new_query._query_string)
    else:
        support = ceil(sample._sample_size*supp)

        mixed_query_tree = HyperLinkedTree(support, event_dimension=sample._sample_event_dimension)

        t_root = type_tree.get_root()
        p_root = pattern_tree.get_root()
        m_root = mixed_query_tree.get_root()
        m_root.query_next_insert_index = np.zeros(len(t_root.query_next_insert_index), dtype=np.int8)

        if len(t_root.child_vertices) == 0:
            if find_descriptive_only:
                vertex_list = sorted(pattern_tree.vertices_to_list(), key=lambda item: (item.query_string.count(' '),item.query_string),reverse=True)
                for vertex in vertex_list:
                    parent_set = pattern_tree.find_parent_vertices(vertex)
                    for parent in parent_set:
                        parent.descriptive=False
            return pattern_tree
        if len(p_root.child_vertices) == 0:
            if find_descriptive_only:
                vertex_list = sorted(type_tree.vertices_to_list(), key=lambda item: (item.query_string.count(' '),item.query_string),reverse=True)
                for vertex in vertex_list:
                    parent_set = type_tree.find_parent_vertices(vertex)
                    for parent in parent_set:
                        parent.descriptive=False
            return type_tree

        new_vertex_queue = []
        for vertex in t_root.child_vertices:
            new_vertex_queue.append((t_root,p_root,m_root,vertex, True))
        for vertex in p_root.child_vertices:
            new_vertex_queue.append((t_root,p_root,m_root,vertex, False))

        while new_vertex_queue:
            (t_vertex, p_vertex, m_vertex, evolved_vertex, evolved_vertex_contains_type_query) = new_vertex_queue.pop(0)

            new_vertices = _merge_query_vertices_multidim(mixed_query_tree, t_vertex, p_vertex, m_vertex, evolved_vertex, evolved_vertex_contains_type_query)

            for new_vertex in new_vertices:
                new_vertex_is_frequent = new_vertex.is_frequent(support)
                if new_vertex_is_frequent is None:
                    new_query = MultidimQuery()
                    new_query.set_query_string(new_vertex.query_string)
                    if not param_smart_matching:
                        new_query_is_frequent = new_query.match_sample(sample, supp)
                    else:
                        new_query.set_query_matchtest('smarter')
                        new_query_is_frequent = new_query.match_sample(sample, supp, dict_iter= param_smart_matching[0], patternset=param_smart_matching[1],
                                parent_dict=param_smart_matching[2], max_query_length=MAX_QUERY_LENGTH)
                    mixed_query_tree.set_match_results(new_vertex, new_query._query_matched_traces)
                    new_vertex.query = new_query
                    if not new_query_is_frequent:
                        continue
                elif new_vertex_is_frequent is False:
                    continue

                for vertex in new_vertex.parent_vertices:
                    vertex.descriptive = False

                if evolved_vertex_contains_type_query:
                    for child_vertex in evolved_vertex.child_vertices:
                        new_vertex_queue.append((evolved_vertex,p_vertex,new_vertex,child_vertex, True))
                    for child_vertex in p_vertex.child_vertices:
                        new_vertex_queue.append((evolved_vertex,p_vertex,new_vertex,child_vertex, False))
                else:
                    for child_vertex in t_vertex.child_vertices:
                        new_vertex_queue.append((t_vertex,evolved_vertex,new_vertex,child_vertex,True))
                    for child_vertex in evolved_vertex.child_vertices:
                        new_vertex_queue.append((t_vertex,evolved_vertex,new_vertex,child_vertex,False))

        mixed_queries = mixed_query_tree
    return mixed_queries

def _syntactically_contained_event(ev_array_1:list, ev_array_2:list, assignments:dict|None=None) -> tuple:
    """
        Decides whether ev_array_1 is contained in ev_array_2 or not.
        The relation checks only one way.
        Example:
            _syntactically_contained(['a','b'], ['a','']) === False
            _syntactically_contained(['a',''], ['a','b']) === True

        Args:
            ev_array_1: an array with attributes or variables (i.e. representing an event)

            ev_array_2: an array with attributes or variables (i.e. representing an event)

            assignments [=None]: optional parameter to give an already defined dictionary as varibale mapping.

        Returns:
            tuple: (value_1, value_2)
                value_1 = True, if ev_array_2 contains ev_array_1
                value_1 = False, else
                value_2 = True, if the assignment has be changed
                value_2 = False, else

        Raises:
            ValueError: if the dimension of both ev_arrays does not match

        Passes:
            None
    """
    if not len(ev_array_1) == len(ev_array_2):
        raise ValueError("Dimension of events does not match!")
    changed_assignments = False
    for dim, value in enumerate(ev_array_1):
        if value == "":
            continue
        if ev_array_2[dim] == "":
            return False, changed_assignments
        if not value[0] == "$":
            if value == ev_array_2[dim]:
                continue
            return False, changed_assignments
        else:
            if value in assignments:
                if assignments[value] == ev_array_2[dim]:
                    continue
                return False, changed_assignments
            else:
                assignments[value] = ev_array_2[dim]
                changed_assignments = True
    return True, changed_assignments

def _syntactically_contained(qs_array_1:list, qs_array_2:list, assignments:dict|None=None) -> bool:
    """
        Decides whether on of the arrays of arrays of events is contained in the other following a given varibale mapping.
        The relation is symmetric.
        Example:
            _syntactically_contained([['a','b']], [['a','']]) === True
            _syntacitcally_contained([['$x0',''],['$x0','']], [['a',''], ['a','']]) === True
            _syntacitcally_contained([['$x0',''],['$x0','']], [['a',''], ['a','']], {'$x0' : 'a'}) === True
            _syntacitcally_contained([['$x0',''],['$x0','']], [['a',''], ['a','']], {'$x0' : 'b'}) === False

        Args:
            qs_array_1: an array of array containg attributes
                e.g. [event.split(';') for event in query_string.split()]

            qs_array_2: an array of array containg attributes

            assignments [=None]: optional parameter to give an already defined dictionary as varibale mapping.

        Returns:
            bool:
                True, if one qs_array contains the other
                False, else.

        Raises:
            TypeError: if a given assignment dictionary is not of type <dict>.
            TypeError if a given assignment dictionary is not of type <dict>.

        Passes:
            ValueError from _syntactically_contained_event(...)
    """
    if len(qs_array_1) == 0:
        return True
    if len(qs_array_2) == 0:
        return False
    if assignments is None:
        assignments = {}
        assignments_cp = {}
    else:
        if not isinstance(assignments, dict):
            raise TypeError("A given assignment dictionary must be of type <dict>!")
        assignments_cp = deepcopy(assignments)
    event_counter = 0
    for i, ev_array_2 in enumerate(qs_array_2):
        ev_array_1 = qs_array_1[event_counter]
        equals, changed = _syntactically_contained_event(ev_array_1, ev_array_2, assignments_cp)
        if not equals:
            continue
        if not changed:
            event_counter += 1
        else:
            if _syntactically_contained(qs_array_1[event_counter+1:],qs_array_2[i+1:],assignments_cp):
                return True
            assignments_cp = deepcopy(assignments)
        if event_counter == len(qs_array_1):
            return True
    return False

def _find_descriptive_querystrings(stats:dict, querystring_set:set) -> set:
    """
        Selects all descriptive querystrings from the querystring_set

        Args:
            stats: Dictionary for statistical evaluation.

            querystring_set: an iterable containing the querystring, which shall be filtered

        Returns:
            set: A set of all descriptive querystrings

        Raises:
            None

        Passes:
            TypeError from _syntactically_contained(...)
            ValueError from _syntactically_contained(...)
    """
    splitted_event_qs_set = [[event.split(";") for event in qs.split()] for qs in querystring_set]
    descriptive_query_set = set()
    while len(splitted_event_qs_set) > 0:
        curr_qs = splitted_event_qs_set.pop()
        curr_qs_is_descriptive = True
        idx = 0
        remove_elements = []
        while idx < len(splitted_event_qs_set):
            splitted_query_string = splitted_event_qs_set[idx]
            if len(curr_qs) < len(splitted_query_string):
                if _syntactically_contained(curr_qs, splitted_query_string):
                    curr_qs_is_descriptive = False
                    break
            elif len(splitted_query_string) < len(curr_qs):
                if _syntactically_contained(splitted_query_string, curr_qs):
                    remove_elements.append(idx)
            else:
                if _syntactically_contained(curr_qs, splitted_query_string):
                    curr_qs_is_descriptive = False
                    break
                if _syntactically_contained(splitted_query_string, curr_qs):
                    remove_elements.append(idx)
            idx += 1
        if curr_qs_is_descriptive:
            descriptive_query_set.add(' '.join([';'.join(event) for event in curr_qs]))
        for idx in reversed(remove_elements):
            del splitted_event_qs_set[idx]

    return descriptive_query_set
