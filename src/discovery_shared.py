#!/usr/bin/python3
"""Shared utilities for all four discovery algorithms."""
import logging
from typing import Iterable
from itertools import combinations, product, chain, combinations_with_replacement, permutations
from copy import deepcopy
from math import ceil
import numpy as np

from query import Query
from query_multidim import MultidimQuery
from sample import Sample
from hyper_linked_tree import HyperLinkedTree
from error import ShinoharaInvalidPositionError

LOG_FORMAT = '| %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

# ---------------------------------------------------------------------------
# Low-level position / extraction utilities (from discovery.py)
# ---------------------------------------------------------------------------

def _find_next_position_in_query_string(query_string: str, position_count: int) -> int:
    if position_count <= 0 or position_count > query_string.count(" ") + 1:
        raise ShinoharaInvalidPositionError(
            f"position_count ({position_count}) is less or greater than the number of events in the given query_string!")
    count = 1
    pos = 0
    while pos < len(query_string):
        if count == position_count:
            break
        if query_string[pos] == " ":
            count = count + 1
            pos = pos + 1
        else:
            pos = pos + 1
    return pos


def _find_next_position_in_query_string_multidim(query_string: str, position_count: int) -> int:
    if position_count <= 0 or position_count > query_string.count(" ") + query_string.count(";") + 1:
        raise ShinoharaInvalidPositionError(
            f"position_count ({position_count}) is less or greater than the number of events in the given query_string!")
    count = 1
    pos = 0
    while pos < len(query_string):
        if count == position_count:
            break
        if query_string[pos] == ";":
            count = count + 1
            if pos < len(query_string) - 1 and query_string[pos + 1] == " ":
                pos = pos + 2
            else:
                pos = pos + 1
        else:
            pos = pos + 1
    return pos


def _extract_var_pre_suf(query_string: str, pos: int) -> tuple:
    if pos < 0 or pos >= len(query_string):
        raise ShinoharaInvalidPositionError(
            f"position_count ({pos}) is less or greater than the number of events in the given query_string!")
    if query_string[pos] != '$':
        raise ShinoharaInvalidPositionError(f"No variable starts at position_count ({pos})!")
    current_variable_and_suffix = query_string[pos:]
    current_variable = current_variable_and_suffix.split(' ')[0]
    if pos == 0:
        prefix = ""
    else:
        prefix = query_string[0:pos - 1]
    suffix = query_string[len(prefix) + len(current_variable) + 1:]
    return (current_variable, current_variable_and_suffix, prefix, suffix)


def _extract_var_pre_suf_multidim(query_string: str, pos: int) -> tuple:
    if pos < 0 or pos >= len(query_string):
        raise ShinoharaInvalidPositionError(
            f"pos ({pos}) is less or greater than the number of events in the given query_string!")
    if query_string[pos] != '$':
        raise ShinoharaInvalidPositionError(f"No variable starts at pos ({pos})!")
    current_variable_and_suffix = query_string[pos:]
    current_variable = current_variable_and_suffix.split(';')[0] + ";"
    if pos == 0:
        prefix = ""
    else:
        prefix = query_string[0:pos]
    suffix = query_string[len(prefix) + len(current_variable):]
    return (current_variable, current_variable_and_suffix, prefix, suffix)


def _find_attribute_index(query_string: str, position_count: int) -> int:
    if position_count <= 0 or position_count > query_string.count(" ") + query_string.count(";") + 1:
        raise ShinoharaInvalidPositionError(
            f"pos ({position_count}) is less or greater than the number of events in the given query_string!")
    dimension = query_string.split(' ')[0].count(';')
    if dimension == 0:
        dimension = 1
    attribute_index = (position_count % dimension) - 1
    if attribute_index == -1:
        attribute_index = dimension - 1
    assert attribute_index > -1
    return attribute_index


def combine_all(iteratable1: list, iteratable2: list) -> Iterable:
    if iteratable1 == []:
        return [iteratable2]
    if iteratable2 == []:
        return [iteratable1]
    xs_first, *xs_tail = iteratable1
    ys_first, *ys_tail = iteratable2
    return [[xs_first] + item for item in combine_all(xs_tail, iteratable2)] + \
           [[ys_first] + item for item in combine_all(ys_tail, iteratable1)]


def merge_event_arrays(event1: list, event2: list) -> list | None:
    if not isinstance(event1, type(event2)):
        raise ValueError("Both events have to be of type <list>!")
    if not isinstance(event1, list):
        raise ValueError("Both events have to be of type <list>!")
    if not len(event1) == len(event2):
        raise IndexError("Both events have to have the same length!")
    merged_event = []
    for dim, value in enumerate(event1):
        if value == '':
            merged_event.append(event2[dim])
        elif event2[dim] == '':
            merged_event.append(value)
        else:
            return None
    return merged_event


def matching_smarter(querystring: str, sample: Sample, dict_iter: dict, patternset: set, supp: float, parentstring: str):
    trace_matches = {}
    sample_size = len(sample._sample)

    if querystring.count('$x') == 0:
        return_match_result = True
        num_trace_match = sample_size
        for trace_idx, trace in enumerate(sample._sample):
            idx, dict_iter = smart_trace_match(querystring, trace, trace_idx, dict_iter)
            if trace_idx not in trace_matches:
                trace_matches[trace_idx] = {}
            trace_matches[trace_idx] = idx
            if idx == -1:
                num_trace_match -= 1
            if num_trace_match / sample_size < supp:
                return_match_result = False
                break

        if return_match_result:
            for trace_index, value in trace_matches.items():
                if querystring not in dict_iter:
                    dict_iter[querystring] = {}
                dict_iter[querystring][trace_index] = value
        return return_match_result, trace_matches, list(trace_matches.keys())

    if parentstring.count('$') == 0:
        if not parentstring:
            parent_traces = list(range(sample_size))
        else:
            parent_traces = list(dict_iter[parentstring].keys())
        trace_list = parent_traces + list(range(parent_traces[-1] + 1, sample_size))
        num_trace_match = len(trace_list)
        return_match_result = True
        for trace in trace_list:
            for letter in patternset:
                letter_querystring = querystring.replace('$x0', letter)
                if letter_querystring in dict_iter:
                    if trace in dict_iter[letter_querystring]:
                        if dict_iter[letter_querystring][trace] != -1:
                            if trace not in trace_matches:
                                trace_matches[trace] = {}
                            trace_matches[trace][(letter,)] = dict_iter[letter_querystring][trace]
                    else:
                        idx, dict_iter = smart_trace_match(letter_querystring, sample._sample[trace], trace, dict_iter)
                        if idx != -1:
                            if trace not in trace_matches:
                                trace_matches[trace] = {}
                            trace_matches[trace][(letter,)] = idx
                else:
                    idx, dict_iter = smart_trace_match(letter_querystring, sample._sample[trace], trace, dict_iter)
                    if idx != -1:
                        if trace not in trace_matches:
                            trace_matches[trace] = {}
                        trace_matches[trace][(letter,)] = idx
            if num_trace_match / sample_size < supp:
                return_match_result = False
                break

            if trace not in trace_matches:
                num_trace_match -= 1
        if return_match_result:
            for trace_index, value in trace_matches.items():
                if querystring not in dict_iter:
                    dict_iter[querystring] = {}
                dict_iter[querystring][trace_index] = value
        return return_match_result, trace_matches, list(trace_matches.keys())

    else:
        parent = Query()
        parent.set_query_string(parentstring)
        parent_variables = sorted(list(parent._query_repeated_variables))
        if parentstring in dict_iter:
            parent_traces = list(dict_iter[parentstring].keys())
        else:
            LOGGER.info(querystring)
            LOGGER.info(parentstring)
            LOGGER.info(dict_iter)
            raise ValueError("Yet grand parent is needed!")

        trace_list = parent_traces
        num_trace_match = len(trace_list)
        return_match_result = True
        for trace in trace_list:
            group_list = list(dict_iter[parentstring][trace].keys())
            for group in group_list:
                letter_querystring = querystring
                assert len(group) == len(parent_variables)
                for val, letter in enumerate(group):
                    letter_querystring = letter_querystring.replace(f'${parent_variables[val]}', letter)
                if letter_querystring.count('$') > 0:
                    for letter in patternset:
                        letter_querystring2 = letter_querystring.replace(f'$x{len(group)}', letter)
                        if letter_querystring2 in dict_iter:
                            if trace in dict_iter[letter_querystring2]:
                                if dict_iter[letter_querystring2][trace] != -1:
                                    if trace not in trace_matches:
                                        trace_matches[trace] = {}
                                    trace_matches[trace][group + (letter,)] = dict_iter[letter_querystring2][trace]
                            else:
                                idx, dict_iter = smart_trace_match(letter_querystring2, sample._sample[trace], trace, dict_iter)
                                if idx != -1:
                                    if trace not in trace_matches:
                                        trace_matches[trace] = {}
                                    trace_matches[trace][group + (letter,)] = idx
                        else:
                            idx, dict_iter = smart_trace_match(letter_querystring2, sample._sample[trace], trace, dict_iter)
                            if idx != -1:
                                if trace not in trace_matches:
                                    trace_matches[trace] = {}
                                trace_matches[trace][group + (letter,)] = idx
                else:
                    if letter_querystring in dict_iter:
                        if trace in dict_iter[letter_querystring]:
                            if dict_iter[letter_querystring][trace] != -1:
                                if trace not in trace_matches:
                                    trace_matches[trace] = {}
                                trace_matches[trace][group] = dict_iter[letter_querystring][trace]
                        else:
                            idx, dict_iter = smart_trace_match(letter_querystring, sample._sample[trace], trace, dict_iter)
                            if idx != -1:
                                if trace not in trace_matches:
                                    trace_matches[trace] = {}
                                trace_matches[trace][group] = idx
                    else:
                        idx, dict_iter = smart_trace_match(letter_querystring, sample._sample[trace], trace, dict_iter)
                        if idx != -1:
                            if trace not in trace_matches:
                                trace_matches[trace] = {}
                            trace_matches[trace][group] = idx

            if trace not in trace_matches:
                num_trace_match -= 1
            if num_trace_match / sample_size < supp:
                return_match_result = False
                break
        if return_match_result:
            for trace_index, value in trace_matches.items():
                if querystring not in dict_iter:
                    dict_iter[querystring] = {}
                dict_iter[querystring][trace_index] = value
        return return_match_result, trace_matches, list(trace_matches.keys())


def smart_trace_match(querystring: str, trace: str, trace_idx: int, dict_iter: dict) -> tuple:
    parentstring = ' '.join(querystring.split()[:-1])
    if querystring not in dict_iter:
        dict_iter[querystring] = {}
    if not querystring:
        dict_iter[querystring][trace_idx] = 0
        return 0, dict_iter
    if parentstring not in dict_iter:
        idx, dict_iter = smart_trace_match(parentstring, trace, trace_idx, dict_iter)
    if trace_idx in dict_iter[parentstring]:
        parent_end_pos = dict_iter[parentstring][trace_idx]
    else:
        idx, dict_iter = smart_trace_match(parentstring, trace, trace_idx, dict_iter)
        parent_end_pos = dict_iter[parentstring][trace_idx]
    if not parentstring:
        try:
            end_pos = trace.split().index(querystring)
        except ValueError:
            end_pos = -1
        dict_iter[querystring][trace_idx] = end_pos
        return end_pos, dict_iter
    if parent_end_pos != -1:
        trace_list = trace.split()[parent_end_pos + 1:]
        try:
            idx = trace_list.index(querystring.split()[-1])
        except ValueError:
            idx = -1
        if idx != -1:
            if trace_idx not in dict_iter[querystring]:
                dict_iter[querystring][trace_idx] = {}
            end_pos = dict_iter[parentstring][trace_idx] + idx + 1
            dict_iter[querystring][trace_idx] = end_pos
            return end_pos, dict_iter
        else:
            dict_iter[querystring][trace_idx] = -1
            return -1, dict_iter
    else:
        dict_iter[querystring][trace_idx] = -1
        return -1, dict_iter


# ---------------------------------------------------------------------------
# Tree helpers (from discovery_bu_multidim.py)
# ---------------------------------------------------------------------------

def add_vertex2tree(poss_query, parent, mixed_query_tree, gen_event, parent_dict, matchtest):
    querystring = poss_query._query_string
    if parent._query_string == gen_event and matchtest == 'smarter':
        parentstring = ''
    else:
        parentstring = parent._query_string

    parent_vertex = mixed_query_tree.find_vertex(parentstring)
    while parent_vertex is None:
        mixed_query_tree = add_vertex2tree(parent, parent_dict[parentstring], mixed_query_tree, gen_event, parent_dict, matchtest)
        parent_vertex = mixed_query_tree.find_vertex(parentstring)
    if not mixed_query_tree.find_vertex(querystring):
        mixed_query_tree.insert_query_string(parent_vertex, querystring, query=poss_query, search_for_parents=False)

    return mixed_query_tree


def _next_queries_multidim(query, alphabet, max_query_length, patternset, only_types=False):
    querystring = query._query_string
    querylength = query._query_string_length
    querystring_list = query.get_query_list()
    domain_cnt = query._query_event_dimension
    variables = query._query_repeated_variables
    if variables:
        last_var = sorted(variables)[-1]
    typeset = query._query_typeset
    num_of_vars = len(variables)
    gen_event = ';' * domain_cnt
    gen_event_list = [i for i in gen_event]
    if querystring == '':
        return [MultidimQuery(gen_event)]
    children = []
    children_strings = set()
    pos_last_type = query._pos_last_type_and_variable[0]
    pos_first_var = query._pos_last_type_and_variable[1]
    pos_last_var = query._pos_last_type_and_variable[2]

    if querystring == gen_event:
        if max_query_length >= 2 and not only_types:
            for domain in range(domain_cnt):
                if patternset[domain]:
                    domain_var = ''.join(gen_event_list[:domain] + ['$x0'] + gen_event_list[domain:])
                    child = domain_var + ' ' + domain_var
                    child_query = MultidimQuery()
                    child_query._query_string = child
                    child_query._query_repeated_variables = {'x0'}
                    child_query._query_string_length = 2
                    child_query._pos_last_type_and_variable = np.array([-1, 0, 1])
                    child_query._query_event_dimension = query._query_event_dimension
                    children.append(child_query)

        if max_query_length >= 1:
            for letter in alphabet:
                child = str(letter)
                child_query = MultidimQuery()
                child_query._query_string = child
                child_query._query_typeset = typeset | {letter}
                child_query._query_string_length = 1
                child_query._pos_last_type_and_variable = np.array([0, -1, -1])
                child_query._query_event_dimension = query._query_event_dimension
                children.append(child_query)

    else:
        first_pos = max(pos_last_type, pos_first_var)
        first_pos_event = querystring_list[first_pos]
        first_pos_domains = query.non_empty_domain(first_pos_event)
        for domain in first_pos_domains:
            att = first_pos_event.split(';')[domain]
            if first_pos == pos_last_type and att.count('$') == 0:
                first_pos_domain = domain
            if variables:
                if first_pos == pos_first_var and last_var in att:
                    first_pos_domain = domain

        querystring_split = querystring_list[first_pos:]
        if not only_types:
            for domain in range(domain_cnt):
                if patternset[domain]:
                    domain_var = ''.join(gen_event_list[:domain] + ['$x' + str(num_of_vars)] + gen_event_list[domain:])
                    var_domain = domain_var.find(domain_var.strip(';'))
                    var = 'x' + str(num_of_vars)
                    for idx, event in enumerate(querystring_split, start=first_pos):
                        if querylength + 1 <= max_query_length:
                            if idx != querylength - 1:
                                child = " ".join(querystring_list[:idx + 1]) + ' ' + domain_var + ' ' + " ".join(querystring.split()[idx + 1:])
                            else:
                                child = querystring + ' ' + domain_var

                            for idx2, event2 in enumerate(child.split()[idx + 1:], start=idx + 1):
                                if querylength + 2 <= max_query_length:
                                    if idx2 != querylength:
                                        child2 = " ".join(child.split()[:idx2 + 1]) + ' ' + domain_var + ' ' + " ".join(child.split()[idx2 + 1:])
                                    else:
                                        child2 = child + ' ' + domain_var

                                    child_query = MultidimQuery()
                                    child_query._query_string = child2.strip()
                                    child_query._query_typeset = typeset
                                    child_query._query_repeated_variables = variables | {var}
                                    child_query._query_string_length = querylength + 2
                                    child_query._query_event_dimension = query._query_event_dimension
                                    child_query._pos_last_type_and_variable = np.array([pos_last_type, idx + 1, idx2 + 1])
                                    assert child_query._query_string_length <= max_query_length
                                    if child_query._query_string not in children_strings:
                                        children.append(child_query)
                                        children_strings.add(child_query._query_string)

                                last_non_empty = query.non_empty_domain(event2)[-1]
                                if not event2.split(';')[var_domain] and idx2 <= querylength + 1:
                                    new_event = event2.split(';')
                                    new_event[var_domain] = domain_var.strip(';')
                                    if idx == querylength - 1:
                                        child3 = " ".join(child.split()[:idx2]) + ' ' + ';'.join(new_event)
                                    else:
                                        child3 = " ".join(child.split()[:idx2]) + ' ' + ';'.join(new_event) + ' ' + " ".join(child.split()[idx2 + 1:])
                                    child_query = MultidimQuery()
                                    child_query._query_string = child3.strip()
                                    child_query._query_typeset = typeset
                                    child_query._query_repeated_variables = variables | {var}
                                    child_query._query_string_length = querylength + 1
                                    child_query._query_event_dimension = query._query_event_dimension
                                    child_query._pos_last_type_and_variable = np.array([pos_last_type, idx + 1, idx2])
                                    assert child_query._query_string_length <= max_query_length
                                    if child_query._query_string not in children_strings:
                                        children.append(child_query)
                                        children_strings.add(child_query._query_string)

                        if not event.split(';')[var_domain]:
                            if idx != first_pos or var_domain > first_pos_domain:
                                new_event = event.split(';')
                                new_event[var_domain] = domain_var.strip(';')
                                if idx == querylength - 1:
                                    child = " ".join(querystring_list[:idx]) + ' ' + ';'.join(new_event)
                                else:
                                    child = " ".join(querystring_list[:idx]) + ' ' + ';'.join(new_event) + ' ' + " ".join(querystring_list[idx + 1:])

                                for idx2, event2 in enumerate(child.split()[idx:], start=idx):
                                    if querylength + 1 <= max_query_length:
                                        if idx2 != querylength - 1:
                                            child2 = " ".join(child.split()[:idx2 + 1]) + ' ' + domain_var + ' ' + " ".join(child.split()[idx2 + 1:])
                                        else:
                                            child2 = child + ' ' + domain_var
                                        child_query = MultidimQuery()
                                        child_query._query_string = child2.strip()
                                        child_query._query_typeset = typeset
                                        child_query._query_repeated_variables = variables | {var}
                                        child_query._query_string_length = querylength + 1
                                        child_query._query_event_dimension = query._query_event_dimension
                                        child_query._pos_last_type_and_variable = np.array([pos_last_type, idx, idx2])
                                        assert child_query._query_string_length <= max_query_length

                                        if child_query._query_string not in children_strings:
                                            children.append(child_query)
                                            children_strings.add(child_query._query_string)

                                    last_non_empty = query.non_empty_domain(event2)[-1]
                                    if not event2.split(';')[var_domain]:
                                        if idx2 != first_pos:
                                            new_event = event2.split(';')
                                            new_event[var_domain] = domain_var.strip(';')
                                            if idx == querylength - 1:
                                                child3 = " ".join(child.split()[:idx2]) + ' ' + ';'.join(new_event)
                                            else:
                                                child3 = " ".join(child.split()[:idx2]) + ' ' + ';'.join(new_event) + ' ' + " ".join(child.split()[idx2 + 1:])
                                            child_query = MultidimQuery()
                                            child_query._query_string = child3.strip()
                                            child_query._query_typeset = typeset
                                            child_query._query_repeated_variables = variables | {var}
                                            child_query._query_string_length = querylength
                                            child_query._query_event_dimension = query._query_event_dimension
                                            child_query._pos_last_type_and_variable = np.array([pos_last_type, idx, idx2])
                                            assert child_query._query_string_length <= max_query_length

                                            if child_query._query_string not in children_strings:
                                                children.append(child_query)
                                                children_strings.add(child_query._query_string)

        if pos_first_var >= pos_last_type:
            if pos_first_var != -1:
                var_numb = 0
                for domain, letter in enumerate(querystring_list[pos_first_var].split(';')):
                    if '$' in letter and var_numb <= int(letter.strip('$x;')):
                        last_variable_domain = domain
                        var_numb = int(letter.strip('$x;'))

                last_variable = querystring_list[pos_first_var].split(';')[last_variable_domain]
                num_of_vars = int(last_variable.strip('$x;')) + 1
                domain_var = ''.join(gen_event_list[:last_variable_domain] + [last_variable] + gen_event_list[last_variable_domain:])
            no_letter = True
            if pos_first_var == pos_last_type:
                for event in querystring_list[pos_first_var].split(';')[last_variable_domain + 1:-1]:
                    if event.count('$') == 0 and event:
                        no_letter = False
            if no_letter and not only_types:
                first_pos = max(pos_last_type, pos_last_var)
                querystring_split = querystring_list[first_pos:]
                for idx, event in enumerate(querystring_split, start=first_pos):
                    if querylength + 1 <= max_query_length:
                        if idx != querylength - 1:
                            child = " ".join(querystring_list[:idx + 1]) + ' ' + domain_var + ' ' + " ".join(querystring_list[idx + 1:])
                        else:
                            child = querystring + ' ' + domain_var
                        child_query = MultidimQuery()
                        child_query._query_string = child.strip()
                        child_query._query_typeset = typeset
                        child_query._query_repeated_variables = variables
                        child_query._query_string_length = querylength + 1
                        child_query._query_event_dimension = query._query_event_dimension
                        child_query._pos_last_type_and_variable = np.array([pos_last_type, pos_first_var, idx + 1])
                        assert child_query._query_string_length <= max_query_length

                        if child_query._query_string not in children_strings:
                            children.append(child_query)
                            children_strings.add(child_query._query_string)
                    var_domain = domain_var.find(domain_var.strip(';'))
                    last_non_empty = query.non_empty_domain(event)[-1]
                    if not event.split(';')[var_domain]:
                        new_event = event.split(';')
                        new_event[var_domain] = domain_var.strip(';')
                        if idx == 0:
                            child2 = ';'.join(new_event)
                        elif idx == querylength - 1:
                            child2 = " ".join(querystring_list[:idx]) + ' ' + ';'.join(new_event)
                        else:
                            child2 = " ".join(querystring_list[:idx]) + ' ' + ';'.join(new_event) + ' ' + " ".join(querystring_list[idx + 1:])
                        child_query = MultidimQuery()
                        child_query._query_string = child2.strip()
                        child_query._query_typeset = typeset
                        child_query._query_repeated_variables = variables
                        child_query._query_string_length = querylength
                        child_query._query_event_dimension = query._query_event_dimension
                        child_query._pos_last_type_and_variable = np.array([pos_last_type, pos_first_var, idx])
                        assert child_query._query_string_length <= max_query_length

                        if child_query._query_string not in children_strings:
                            children.append(child_query)
                            children_strings.add(child_query._query_string)

        first_pos = max(pos_last_type, pos_first_var)
        first_pos_event = querystring_list[first_pos]
        if 'last_variable_domain' in locals():
            if pos_first_var != pos_last_type:
                last_symbol_domain = last_variable_domain
            else:
                for domain, letter in enumerate(first_pos_event.split(';')):
                    if letter:
                        last_symbol_domain = domain
        else:
            for domain, letter in enumerate(first_pos_event.split(';')):
                if letter and '$' not in letter:
                    last_symbol_domain = domain

        querystring_split = querystring_list[first_pos:]
        for letter in alphabet:
            for idx, event in enumerate(querystring_split, start=first_pos):
                if querylength + 1 <= max_query_length:
                    if idx != querylength - 1:
                        child = " ".join(querystring_list[:idx + 1]) + ' ' + letter + ' ' + " ".join(querystring_list[idx + 1:])
                    else:
                        child = querystring + ' ' + letter
                    child_query = MultidimQuery()
                    child_query._query_string = child.strip()
                    child_query._query_typeset = typeset | {letter}
                    child_query._query_repeated_variables = variables
                    child_query._query_string_length = querylength + 1
                    child_query._query_event_dimension = query._query_event_dimension
                    if idx < pos_last_var:
                        child_query._pos_last_type_and_variable = np.array([idx + 1, pos_first_var, pos_last_var + 1])
                    else:
                        child_query._pos_last_type_and_variable = np.array([idx + 1, pos_first_var, pos_last_var])
                    assert child_query._query_string_length <= max_query_length
                    if child_query._query_string not in children_strings:
                        children.append(child_query)
                        children_strings.add(child_query._query_string)
                letter_domain = letter.find(letter.strip(';'))
                last_non_empty = query.non_empty_domain(event)[-1]
                if not event.split(';')[letter_domain]:
                    if idx != first_pos or letter_domain > last_non_empty or '$' in event.split(';')[last_non_empty]:
                        if letter_domain < last_symbol_domain and idx == first_pos:
                            continue
                        new_event = event.split(';')
                        new_event[letter_domain] = letter.strip(';')
                        if idx == querylength - 1:
                            child2 = " ".join(querystring_list[:idx]) + ' ' + ';'.join(new_event)
                        else:
                            child2 = " ".join(querystring_list[:idx]) + ' ' + ';'.join(new_event) + ' ' + " ".join(querystring_list[idx + 1:])
                        child_query = MultidimQuery()
                        child_query._query_string = child2.strip()
                        child_query._query_typeset = typeset | {letter}
                        child_query._query_repeated_variables = variables
                        child_query._query_string_length = querylength
                        child_query._query_event_dimension = query._query_event_dimension
                        child_query._pos_last_type_and_variable = np.array([idx, pos_first_var, pos_last_var])
                        assert child_query._query_string_length <= max_query_length

                        if child_query._query_string not in children_strings:
                            children.append(child_query)
                            children_strings.add(child_query._query_string)

    return children


def _merge_domain_queries(querystring_dict, pos_dict, max_query_length, supp=1.0):
    query_list = list(querystring_dict.values())
    domain_indeces = list(querystring_dict.keys())
    querystring1 = query_list[0]
    domain_cnt = querystring1.split()[0].count(';')
    sample_size = len(pos_dict[querystring1]['trace_instances'])
    gen_event = ';' * domain_cnt
    all_positions = {}
    for domain, querystring in querystring_dict.items():
        all_positions[domain] = pos_dict[querystring]['trace_instances']

    if domain_cnt == -1:
        return {}

    min_length = -1
    trace_id = 0
    number_of_traces = ceil(sample_size - supp * sample_size) + 1
    trace_id_list = []
    if supp == 1.0:
        for idx in range(sample_size):
            query_occ_list = [pos_dict[querystring]['occurences'] for querystring in query_list]
            trace_product = np.prod(query_occ_list)
            if trace_product <= 3 * len(query_occ_list):
                trace_id_list = [idx]
                break
            if min_length == -1 or min_length > trace_product:
                min_length = trace_product
                trace_id_list = [idx]
    else:
        all_trace_ids = [set(inner_dict.keys()) for inner_dict in all_positions.values()]
        query_occ_list = set.intersection(*all_trace_ids)
        trace_id_list = list(query_occ_list)

    all_instance_pairs = []
    if len(trace_id_list) < supp * sample_size and supp < 1.0:
        return all_instance_pairs

    for idx in trace_id_list[:number_of_traces]:
        instance_trace_list = [all_positions[domain][idx] for domain in domain_indeces]
        instance_pairs = list(product(*instance_trace_list))
        all_instance_pairs.extend(instance_pairs)

    return all_instance_pairs


def pos2query(querystring_dict, pair, adapted_querystring_dict, max_query_length):
    new_query = ''
    new_query_list = []
    instance_positions = sorted(set(chain(*pair)))
    if len(instance_positions) > max_query_length:
        return new_query
    query_list = list(querystring_dict.values())
    domain_indeces = list(querystring_dict.keys())
    domain_cnt = -1
    for querystring in query_list:
        querystring_list = querystring.split()
        if querystring:
            domain_cnt = querystring_list[0].count(';')
            break

    for pos in instance_positions:
        domain_pos = [idx for idx, p in zip(domain_indeces, pair) if pos in p]
        last_domain = -1
        instance_count = 0
        for domain in range(domain_cnt):
            if domain in domain_indeces:
                if domain in domain_pos:
                    dom_instance = pair[domain_indeces.index(domain)]
                    instance_type = dom_instance.index(pos)
                    instance_count += 1
                    if last_domain >= 0:
                        new_query_list.append(adapted_querystring_dict[domain].split()[instance_type].strip(';'))
                        new_query_list.append(';')
                    elif new_query:
                        new_query_list.append(' ')
                        new_query_list.append(adapted_querystring_dict[domain].split()[instance_type].strip(';'))
                        new_query_list.append(';')
                    else:
                        new_query_list.append(adapted_querystring_dict[domain].split()[instance_type].strip(';'))
                        new_query_list.append(';')
                else:
                    if last_domain >= 0:
                        new_query_list.append(';')
                    elif new_query:
                        new_query_list.append(' ;')
                    else:
                        new_query_list.append(';')
            else:
                if last_domain >= 0:
                    new_query_list.append(';')
                elif new_query:
                    new_query_list.append(' ;')
                else:
                    new_query_list.append(';')

            if domain == domain_cnt - 1 and pos != instance_positions[-1]:
                new_query_list.append(' ')
            last_domain += 1

    if new_query_list:
        new_query = ''.join(new_query_list)
        normal_form = reposition_vars(new_query)
        return normal_form
    else:
        return ''


def adapted_querystring(querystring_dict, query_dict):
    query_list = list(querystring_dict.values())
    domain_indeces = list(querystring_dict.keys())
    var_domains = [domain for domain, querystring in querystring_dict.items() if querystring.count('$') != 0]
    adapted_querystring_dict = deepcopy(querystring_dict)
    if len(var_domains) > 1:
        for idx, var_domain in enumerate(var_domains):
            var_domain_idx = domain_indeces.index(var_domain)
            if idx == 0:
                var_querystring = querystring_dict[var_domain].replace(';', '')
                var_queryslist = var_querystring.split()
                var_list = [int(event[2:]) for event in var_queryslist if event.startswith('$x')]
                max_var = max(var_list)
                new_var = max_var + 1
            else:
                var_querystring2 = querystring_dict[var_domain].replace(';', '')
                var_queryslist2 = var_querystring2.split()
                var_list2 = [int(event[2:]) for event in var_queryslist2 if event.startswith('$x')]
                max_var2 = max(var_list2)
                new_var2 = max_var2 + 1
                querystring = querystring_dict[var_domain]
                for old_var in range(new_var2):
                    var_shift = old_var + new_var
                    querystring = querystring.replace(f'$x{old_var}', f'$x_{var_shift}')
                querystring = querystring.replace('_', '')
                new_var += new_var2
                adapted_querystring_dict[var_domain] = querystring
    return adapted_querystring_dict


def calc_instance_pairs(instance, dom_query_length, max_dom_query_length):
    combs = sorted(combinations(instance, dom_query_length[0]))
    rcombs = sorted(combinations(instance, max_dom_query_length - dom_query_length[0]), reverse=True)

    if len(dom_query_length) == 2:
        inst_pairs = tuple((x, y) for x, y in zip(combs, rcombs) if len(set(x)) == len(x) and len(set(y)) == len(y))
        return set(inst_pairs)
    else:
        inst_pairs = []
        for x, y in zip(combs, rcombs):
            if len(set(x)) == len(x):
                for z in calc_instance_pairs(y, dom_query_length[1:], max_dom_query_length - dom_query_length[0]):
                    inst_pairs.append((x, *z))
        return set(inst_pairs)


def to_normalform(querystring):
    if not querystring:
        return querystring
    normal_query = MultidimQuery()
    normal_query.set_query_string(querystring, recalculate_attributes=False)
    normal_query.query_string_to_normalform()
    return normal_query._query_string


def non_descriptive_queries_multidim(query=None, querystring=None, parent_dict=None):
    if query:
        querystring = query._query_string
        variables = query._query_repeated_variables
        if not variables and querystring.count('$') != 0:
            query.set_query_repeated_variables()
            variables = query._query_repeated_variables
        var_domains = []
    else:
        query = MultidimQuery()
        query.set_query_string(querystring)
        variables = query._query_repeated_variables
        var_domains = []

    non_descriptive_set = set()
    if not querystring:
        return non_descriptive_set
    query_liste = query.get_query_list()
    query_length = len(query_liste)

    num_of_vars = len(variables)
    domain_cnt = query_liste[0].count(';')
    gen_event = ';' * domain_cnt
    gen_event_list = [i for i in gen_event]

    typeset = {}
    for domain in range(domain_cnt):
        domain_types = set()
        domain_query = ' '.join([event.split(';')[domain] for event in query_liste])
        for letter in domain_query.split():
            if '$' not in letter:
                domain_types.add(letter)
        if domain_types:
            typeset[domain] = domain_types

    if variables:
        for idx, variable in enumerate(variables):
            variable_count = querystring.count(variable)
            var_pos = querystring.find(variable) - 1
            if var_domains:
                var_domain = var_domains[idx]
            else:
                var_domain = querystring[:var_pos].count(';') % domain_cnt
            domain_query_liste = [event.split(';')[var_domain] for event in query_liste]
            counter = 2
            if variable_count >= 4:
                pos_list = {i for i in range(len(domain_query_liste)) if domain_query_liste[i] == '$' + variable}
                pos_pairs = set()
                while counter <= variable_count - 2:
                    pos_pairs.update(set(combinations(pos_list, counter)))
                    counter += 1

                for pos_pair in pos_pairs:
                    if var_domain == 0:
                        gen_querystring = " ".join([
                            ''.join(
                                [';'.join(query_liste[i].split(';')[:var_domain])] +
                                [f"$x{num_of_vars};"] +
                                [';'.join(query_liste[i].split(';')[var_domain + 1:])])
                            if i in pos_pair else query_liste[i]
                            for i in range(len(query_liste))])
                    else:
                        gen_querystring = " ".join([
                            ''.join(
                                [';'.join(query_liste[i].split(';')[:var_domain])] +
                                [f";$x{num_of_vars};"] +
                                [';'.join(query_liste[i].split(';')[var_domain + 1:])])
                            if i in pos_pair else query_liste[i]
                            for i in range(len(query_liste))])

                    if gen_querystring not in parent_dict:
                        gen_querystring = reposition_vars(gen_querystring)

                    if gen_querystring != querystring:
                        non_descriptive_set.add(gen_querystring)
    if typeset:
        for domain, letters in typeset.items():
            for letter in letters:
                domain_query_liste = [event.split(';')[domain] for event in query_liste]
                letter_count = domain_query_liste.count(letter)
                counter = 2
                pos_list = {i for i in range(len(domain_query_liste)) if domain_query_liste[i] == letter}
                pos_pairs = set()
                while counter <= letter_count:
                    pos_pairs.update(set(combinations(pos_list, counter)))
                    counter += 1

                    for pos_pair in pos_pairs:
                        if domain == 0:
                            gen_querystring = " ".join([
                                ''.join(
                                    [';'.join(query_liste[i].split(';')[:domain])] +
                                    [f"$x{num_of_vars};"] +
                                    [';'.join(query_liste[i].split(';')[domain + 1:])])
                                if i in pos_pair else query_liste[i]
                                for i in range(len(query_liste))])
                        else:
                            gen_querystring = " ".join([
                                ''.join(
                                    [';'.join(query_liste[i].split(';')[:domain])] +
                                    [f";$x{num_of_vars};"] +
                                    [';'.join(query_liste[i].split(';')[domain + 1:])])
                                if i in pos_pair else query_liste[i]
                                for i in range(len(query_liste))])
                        if gen_querystring not in parent_dict:
                            gen_querystring = to_normalform(gen_querystring)

                        if gen_querystring != querystring:
                            non_descriptive_set.add(gen_querystring)

    for event in query_liste:
        for domain in range(domain_cnt):
            domain_query = " ".join([event.split(';')[domain] for event in query_liste])
            dom_query_list = []
            for item in domain_query.split():
                cur_event_list = gen_event_list[:domain] + [item] + gen_event_list[domain:]
                cur_event = ''.join(cur_event_list)
                dom_query_list.append(cur_event)
            dom_query_string = ' '.join(dom_query_list)
            if dom_query_string:
                if dom_query_string not in parent_dict:
                    dom_query_string = rename_variables(dom_query_string, variables)
                    normal_form = dom_query_string
                else:
                    normal_form = dom_query_string
                if normal_form != querystring:
                    non_descriptive_set.add(normal_form)
    subqueries = combinations(query_liste, query_length - 1)
    subquerystrings = {' '.join(subquery) for subquery in subqueries}
    subsubqueries = []
    for event in query_liste:
        event_set = set()
        for idx, symbol in enumerate(event):
            if symbol == ';' or event[idx - 1] != ';':
                continue
            new_event = event
            while new_event[idx] != ';':
                new_event = new_event[:idx] + new_event[idx + 1:]
            if new_event != gen_event:
                event_set.add(new_event)
        event_set.add(event)
        subsubqueries.append(event_set)
    subsubs = [' '.join(subquery) for subquery in set(product(*subsubqueries))]
    subquerystrings.update(subsubs)
    for gen_querystring in subquerystrings:
        if gen_querystring not in ['', gen_event]:
            if gen_querystring.count('$') >= 1:
                for i in range(len(variables)):
                    var = f'$x{i}'
                    if gen_querystring.count(var) == 1:
                        gen_querystring = gen_querystring.replace(var, '')
                gen_querystring = ' '.join([event for event in gen_querystring.split() if event != gen_event])
                if not gen_querystring:
                    continue
                gen_querystring = rename_variables(gen_querystring, variables)
                gen_querystring = reposition_vars(gen_querystring)
            normal_form = gen_querystring
            if normal_form != querystring:
                non_descriptive_set.add(normal_form)
    if '' in non_descriptive_set:
        non_descriptive_set.remove('')
    if gen_event in non_descriptive_set:
        non_descriptive_set.remove(gen_event)
    return non_descriptive_set


def ht_descriptive_queries(query_tree: HyperLinkedTree, matching_queries: set):
    ht_non_descriptive = set()
    for vertex in query_tree.vertices_to_list(frequent_items_only=False):
        querystring = vertex.query_string
        if querystring not in matching_queries:
            continue
        if vertex.parent_vertices:
            parent_vertices = vertex.parent_vertices
        else:
            parent_vertices = query_tree.find_parent_vertices(vertex)
            vertex.parent_vertices = parent_vertices
        parent_querystrings = [parent_vertex.query_string for parent_vertex in parent_vertices]
        ht_non_descriptive.update(parent_querystrings)
    pos_descriptive = matching_queries - ht_non_descriptive
    splitted_event_qs_set = [[event.split(";") for event in qs.split()] for qs in pos_descriptive]
    non_descriptive_query_set = set()
    qs_set_pairs = list(combinations(splitted_event_qs_set, 2))
    for cur_tuple in qs_set_pairs:
        curr_qs = cur_tuple[0]
        splitted_query_string = cur_tuple[1]

        curr_qs_is_descriptive = True
        splitted_event_is_descriptive = True
        if len(curr_qs) < len(splitted_query_string):
            if _syntactically_contained(curr_qs, splitted_query_string):
                curr_qs_is_descriptive = False
                childstring = ' '.join([';'.join(event) for event in splitted_query_string])
                new_child = query_tree.find_vertex(childstring)
                if new_child:
                    non_desc_string = ' '.join([';'.join(event) for event in curr_qs])
                    non_desc_vertex = query_tree.find_vertex(non_desc_string)
                    new_child.parent_vertices.add(non_desc_vertex)

        elif len(splitted_query_string) < len(curr_qs):
            if _syntactically_contained(splitted_query_string, curr_qs):
                splitted_event_is_descriptive = False
                childstring = ' '.join([';'.join(event) for event in curr_qs])
                non_desc_string = ' '.join([';'.join(event) for event in splitted_query_string])
                new_child = query_tree.find_vertex(childstring)
                non_desc_vertex = query_tree.find_vertex(non_desc_string)
                new_child.parent_vertices.add(non_desc_vertex)
        else:
            if _syntactically_contained(curr_qs, splitted_query_string):
                curr_qs_is_descriptive = False
                childstring = ' '.join([';'.join(event) for event in splitted_query_string])
                new_child = query_tree.find_vertex(childstring)
                if new_child:
                    non_desc_string = ' '.join([';'.join(event) for event in curr_qs])
                    non_desc_vertex = query_tree.find_vertex(non_desc_string)
                    new_child.parent_vertices.add(non_desc_vertex)

            if _syntactically_contained(splitted_query_string, curr_qs):
                splitted_event_is_descriptive = False
                childstring = ' '.join([';'.join(event) for event in curr_qs])
                non_desc_string = ' '.join([';'.join(event) for event in splitted_query_string])
                new_child = query_tree.find_vertex(childstring)
                non_desc_vertex = query_tree.find_vertex(non_desc_string)
                new_child.parent_vertices.add(non_desc_vertex)

        if not curr_qs_is_descriptive:
            non_descriptive_query_set.add(non_desc_string)
        elif not splitted_event_is_descriptive:
            non_descriptive_query_set.add(non_desc_string)

    return pos_descriptive - non_descriptive_query_set, query_tree


def _syntactically_contained(qs_array_1: list, qs_array_2: list, assignments: dict | None = None) -> bool:
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
            if _syntactically_contained(qs_array_1[event_counter + 1:], qs_array_2[i + 1:], assignments_cp):
                return True
            assignments_cp = deepcopy(assignments)
        if event_counter == len(qs_array_1):
            return True
    return False


def reposition_vars(gen_querystring):
    gen_querylist = gen_querystring.split()
    counter2 = 0
    seen_vars = set()
    for event in gen_querylist:
        for domain_letter in event.split(';'):
            if '$x' in domain_letter and domain_letter not in seen_vars:
                if gen_querystring.count(domain_letter) > 1:
                    if int(domain_letter[2:]) == counter2:
                        pass
                    else:
                        gen_querystring = gen_querystring.replace(domain_letter, f"$x_{counter2}")
                    seen_vars.add(domain_letter)
                    counter2 += 1
    gen_querystring = gen_querystring.replace('_', '')
    return gen_querystring


def _syntactically_contained_event(ev_array_1: list, ev_array_2: list, assignments: dict | None = None) -> tuple:
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


def rename_variables(gen_querystring, variables):
    total_count = gen_querystring.count('$')
    cur_count = 0
    for i in range(len(variables) - 1):
        var = f'$x{i}'
        cur_count += gen_querystring.count(var)
        if cur_count == total_count:
            break
        if gen_querystring.count(var) == 0:
            for j in range(i + 1, len(variables)):
                if gen_querystring.count(f'$x{j}') >= 1:
                    old_var = f'$x{j}'
                    break
            gen_querystring = gen_querystring.replace(old_var, var)
    return gen_querystring


def adapt_sample_multidim(sample):
    """Adapts a MultidimSample by adding frequency counters to common types."""
    trace_dict = {}
    sample_set = sample._sample
    sample_list = []
    domain_cnt = sample_set[0].split(' ')[0].count(';')

    for trace_id, trace in enumerate(sample_set):
        domain_list = []
        trace_list = [domain.split(';')[:-1] for domain in trace.split()]

        for i in range(domain_cnt):
            current_domain = []
            for event in trace_list:
                current_domain.append(event[i])
            domain_list.append(current_domain)
        sample_list.append(domain_list)

    for domain in range(len(sample_list[0])):
        domain_sample_list = []
        for trace_id, trace in enumerate(sample_list):
            domain_sample_list.append(sample_list[trace_id][domain])
        domain_sample_set = [' '.join(trace) for trace in domain_sample_list]
        domain_sample = Sample()
        domain_sample.set_sample(domain_sample_set)
        domain_sample.set_sample_typeset()
        domain_sample.adapt_sample()
        domain_sample_set = domain_sample._sample

        for trace_id, trace in enumerate(sample_set):
            if trace_id in trace_dict:
                current_trace = trace_dict[trace_id]
                trace_dict[trace_id] = ' '.join([
                    ';'.join([cur_event] + [new_ev])
                    for cur_event, new_ev in zip(current_trace.split(), domain_sample_set[trace_id].split())
                ])
            else:
                trace_dict[trace_id] = domain_sample_set[trace_id]
    new_sampleset = []
    for trace_id, trace in trace_dict.items():
        new_sampleset.append(trace)
    new_sample = Sample()
    new_sample.set_sample(new_sampleset)
    new_sample.set_sample_typeset()
    return new_sample


# ---------------------------------------------------------------------------
# Shared outer loop for D-U-S and B-S-S
# ---------------------------------------------------------------------------

def _domain_separated_discovery(sample, supp, matchtest, max_query_length, per_domain_fn) -> dict:
    """Outer discovery loop for D-U-S and B-S-S.

    Each domain is discovered independently via per_domain_fn, then results
    are merged across domains.

    Args:
        sample: MultidimSample instance.
        supp: Support threshold in [0, 1].
        matchtest: 'smarter' for D-U-S, 'pattern-split-sep' for B-S-S.
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
                        match = poss_query.match_sample(
                            sample, supp, parent_dict=parent_dict,
                            patternset=all_patternset, dict_iter=dict_iter)
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

    result_dict = {}
    result_dict['queryset'], mixed_query_tree = ht_descriptive_queries(mixed_query_tree, descriptive_query_list)
    result_dict['matching_dict'] = query_dict
    result_dict['domain_queries'] = [len(domain_list) for domain_list in domain_query_list]
    result_dict['merged queries'] = len(descriptive_query_list)
    return result_dict
