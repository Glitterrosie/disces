#!/usr/bin/python3
""" Contains the class for handling multidimensional Queries"""
import logging
from copy import copy
from typing import Pattern, Match
from typing_extensions import Self
from math import ceil
import numpy as np
from query import Query
from sample_multidim import MultidimSample
from error import InconsistentQueryError,QueryRegexError,InvalidQueryStringLengthError,InvalidQueryGapConstraintError,InvalidQueryLocalWindowSizeError,InvalidEventDimensionError

DISCOVERY_ALGORITHM_LIST = [
    'shinohara_icdt',
    'bottom_up',
    'top_down'
]

MATCH_TEST_LIST = [
    'regex',
    'finditer'
]

QUERY_CLASS_LIST = [
    'normal'
]

#Logger Configuration:
LOG_FORMAT = '| %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)


class MultidimQuery(Query):
    """
        Base class for representing and discovering a multidimensional Query.

        Queries consist of a query string, global or local window size(s) and
        optional gap constraints.

        The query string consists of events which are composed of one or more
        attributes. The number of attributes defines the so called dimension of
        the corresponding event. Each attribute is represented by so called
        types or variables. Types are represented by strings over an alphabet
        (ascii, excl. whitespace, semicolon and $).

        Gap constraints describe which types are forbidden between to positions
        of the query string.

        The global window size denotes the range for a match in a trace, while
        the list of local window size tuples describes lower and upper bounds
        for the length of each gap between two consecutive events in the query
        string.
        Note that these bounds range over attributes instead of whole events.
        Given a query a;b; c;d; consisting of two 2-dimensional events we
        assume the local window sizes to be [(0,0),(i,j),(0,0)] and
            i,j mod 2 = 0, i<=j,
        since events are 2-dimensional. The a (0,0)-tuple at a gap within an
        event ensures that a; and b; (or c; and d;) correspond to the same
        event. A query in normalfrom may have up to two additional window size
        tuples, up to one left of the first and up to one right of the last
        event. We use the tuple (-1,-1) as a placeholder for these two.

        A query matches a trace t from a sample iff there exists a mapping from
        the query string to t, i.e. the query string is a subsequence of t, and
        neither the window size(s) nor the gap constraints are violated.

        Given a sample and a support this class offers the functionality to
        discover queries which fullfill the support.

        Attributes:
            _query_string: A query string consists of events represented by a
                number of types and/or variables, each symbolizing an attribute
                of the event. Events are modeled as blocks within
                _query_string, separated by whitespaces. The end of each
                attribute is marked by ";". The beginning of a variable is
                marked by $, every unmarked attribute is a type. Note that
                #; = #events x dimension and #events = #spaces - 1.

            _query_list: List of event strings which represents the query string.

            _query_event_dimension: Integer which represents the max. number of
                attributes per event.

            _query_string_length: Integer which represents the length of the
                query string, i.e. the number of events.

            _query_gap_constraints: List of sets of strings, describing which
                types are not allowed between two consecutive attributes (or
                events) in the query string.

            _query_windowsize_global: Global window size of the query as
                integer. Uses -1 as default, which means no global window size
                is set.

            _query_windowsize_local: List of local window sizes.

            _query_class: A string which describes the query class. E.g.
                "#repeated variables: 0". Uses "normal" as default if no class
                is set and has to be chosen from the global variable
                QUERY_CLASS_LIST.

            _query_repeated_variables: Set of strings which represent the
                repeated variabels in the _query_string. A variable is called
                'repeated' if it occurs more than once in the _query_string.

            _query_is_in_normalform: Bool which indicates whether the current
                query string is in normalform or is not.
                A query string is in normalform iff it does not contain non-
                repreated variables. This function deleted all non-repeated
                variables and merges the corresponding gap contraints, if they
                exists. If local window sizes exist they are aggregated as
                well. In particular the special window size tuples (-1,-1) may
                be set differently if a variable at the beginning or end is
                deleted to reach normalform.

            _query_sample: A Sample on which the query is evaluated or from
                which the query is inferred. Uses None as default if no sample
                is set.

            _query_matched_traces: List of indices of all traces in
                _query_sample, i.e. the positions of the traces within the
                sample, s.t. the a successful match test was performed. Note
                that this list may not contain all traces that match the query!

            _query_not_matched_traces: List of indices of all traces in
                _query_sample, i.e. the positions of the traces within the
                sample, s.t. an unsuccessful match test was performed. Note
                that this list may not contain all traces that do not match the
                query!

            _query_sample_support: Float between 0 and 1 which represents the
                support of the query regarding the given sample. Uses -1.0 as
                default if no support or sample is set.

            _query_typeset: Set of types occuring in the _query_string.

            _query_attribute_typesets: Dict which stores the set of occuring
                types per attribute.

            _query_discovery_algorithm: A string which defines the requested
                discovery algorithm. Uses 'shinohara' as default discovery
                algorithm. Has to be chosen from the global variable
                DISCOVERY_ALGORITHM_LIST.

            _query_matchtest: A string which defines the requested match test.
                Uses 'regex' as default and has to be chosen from the global
                variable MATCH_TEST_LIST.

            _pos_last_type_and_variable: A numpy array containing last type
                position first position of last variable, last position of last
                variable. Default value: np.array([-1,-1,-1]).
    """
    def __init__(self, given_query_string=None, given_query_gap_constraints=None, given_query_windowsize_global=-1, given_query_windowsize_local=None, is_in_normalform=False) -> None:
        LOGGER.debug('Creating an instance of Query')
        self._query_repeated_variables:set = set()
        """Set of variables occuring more than once in _query_string. Default: set()"""
        self._query_typeset:set = set()
        """Set of types occuring in the _query_string. Default: set()"""
        self._query_attribute_typesets:dict = dict()
        """Dict storing the set of occuring types per attribute. Default: dict()"""
        self._query_list:list = []
        """List of event strings which represents the query string. Default: []"""

        if given_query_string is not None:
            assert isinstance(given_query_string, str)
            self._query_string:str = given_query_string
            """String of events consisting of types and variables. Events are separated by whitespaces. Default: ''"""
            self.set_query_string_length()
            self.set_query_repeated_variables()
            self.set_query_typeset()
            self.set_query_event_dimension()
            if given_query_gap_constraints is not None:
                self._query_gap_constraints:list = given_query_gap_constraints
                """List of sets of strings containing forbidden types between attributes in _query_string. Default: []"""
            else:
                self._query_gap_constraints = []
            if given_query_windowsize_global > -1:
                self.set_query_windowsize_global(given_query_windowsize_global)
            else:
                self._query_windowsize_global:int = -1
                """Integer defining the maximal range for a match in a trace. Default: -1"""
            if given_query_windowsize_local is not None:
                self.set_query_windowsize_local(given_query_windowsize_local)
            else:
                self._query_windowsize_local:list = []
                """List of tuples defining lower and upper bounds for the gap length between attributes in _query_string. Default: []"""
        else:
            self._query_string = ""
            self._query_string_length:int = 0
            """Number of events within _query_string, i.e. number of types and variables. Default: 0"""
            self._query_event_dimension:int = 1
            """Max. number of attributes per event. Default: 1"""
            self._query_gap_constraints = []
            self._query_windowsize_global = -1
            self._query_windowsize_local = []
            self._pos_last_type_and_variable = np.array([-1,-1,-1])
            """Np array containing last type pos, 1st pos of last var, last pos of last var. Default: np.array([-1,-1,-1])"""

        self._query_string_regex:str = ""
        """Regex-string of _query_string. Default: ''"""
        self._query_class:str = "normal"
        """String defining the query class. Default: 'normal'"""
        self._query_is_in_normalform:bool = is_in_normalform
        """Boolean indicating whether the query is in normalform. Default: False"""
        self._query_sample:MultidimSample|None = None
        """Associated sample. Default: None"""
        self._query_matched_traces:list = []
        """List of all trace-indices in _query_sample, s.t. a successful match test was performed. Default: []"""
        self._query_not_matched_traces:list = []
        """List of all trace-indices in _query_sample, s.t. an unsuccessful match test was performed. Default: []"""
        self._query_sample_support:float = -1.0
        """Float between 0 and 1 representing the support of the query in _query_sample. Default: -1"""
        self._query_discovery_algorithm:str = 'shinohara'
        """String defining discovery_algorithm that was used or should be used. Default: 'shinohara'"""
        self._query_matchtest:str = 'regex'
        """String defining which matchtest should be used. Default: 'regex'"""

    ##################################################

    ##################################################

    def __eq__(self, other_query:Self) -> bool:
        """
            Defines the magic method '==' with
            _syntactically_equals(self, other_query).

            See "_syntactically_equals(self, other_query)" for further
            information.

            Args:
                other_query: Other query to check with.

            Returns:
                True iff they are syntactically equal.
                False else.
            Passes on:
                TypeError: Argument is no instance of 'query'

                InvalidQueryGapConstraintEror: One query has not the right
                    amount of gap constraints.
        """
        return self._syntactically_equal(other_query)

    def __ne__(self, other_query:Self) -> bool:
        """"
            Defines the magic method '!=' as inversion of '=='.

            Check "__eq__(self, other_query)" for further information.

            Args:
                other_query: Other query to check with.

            Returns:
                True iff they are not syntactically equal.
                False else.
            Passes on:
                TypeError: Argument is no instance of 'query'

                InvalidQueryGapConstraintEror: One query has not the right
                    amount of gap constraints.
        """
        return not self == other_query

    def __str__(self) -> str:
        """
        """
        return self._query_string

    ##################################################

    def match_sample(self, sample, supp, complete_test=False, dict_iter = None,
                     patternset = None, parent_dict = None, max_query_length = -1):
        """
            Checks whether the query matches a given sample with given support.

            Determines and sets the support of the query regarding the sample
            if a full test is performed. Stores indices of tested traces
            depending on the match test result in _query_matched_traces or
            _query_not_matched_traces.

            Args:
                sample: Sample instance.

                supp: Float between 0 and 1 which describes the requested
                    support.

                complete_test: Boolean which indicates whether all traces
                    should be tested or not. In the latter case the loop over
                    traces will stop if either the support is fulfilled or can
                    not be fulfilled anymore.
                
                dict_iter= dict_iter (dictionary): nested dictionary for each query and trace 
                the last matching position is value. Default None

                patternset: set of types occurring twice in at least one trace. Default None

                parent_dict: Dictionary containing parent query to each querystring. Default None

            Returns:
                True iff the query matches the given sample with given supp.

            Raises:
                EmptySampleError: The given sample is empty.
                InvalidQuerySupportError: Supp is <0 or >1.
        """
        if self._query_matchtest == 'regex':
            return Query.match_sample(self, sample=sample, supp=supp, complete_test=complete_test)
        
        if self._query_matchtest == 'smarter':
            sample_size = sample._sample_size
            querystring = self._query_string
            if max_query_length != -1 and self._query_string_length > max_query_length:
                return False
            
            if self._query_matched_traces:
                trace_list = self._query_matched_traces
                for trace in trace_list:
                    if trace < sample_size:
                        if not sample._sample[trace]:
                            self._query_matched_traces.remove(trace)
                            if querystring.count('$')!=0 and dict_iter:
                                dict_iter[querystring][trace]= -1
                            elif dict_iter:
                                dict_iter[querystring].pop(trace)
                if len(self._query_matched_traces)/sample_size >= supp:
                    return True
                elif len(self._query_matched_traces) + sample_size - trace_list[-1] < ceil(supp*sample_size):
                    return False

            matching = self._matching_smarter_multidim(sample=sample, supp =supp, dict_iter=dict_iter,
                                                           patternset=patternset,  parent_dict=parent_dict)
            if querystring.count('$')!=0:
                matchingcount= len(matching)
                self._query_matched_traces = list(matching.keys())
            else:
                matchingcount=0
                matched_traces = []
                for key, value in matching.items():
                    if value != -1:
                        matched_traces.append(key)
                        matchingcount +=1
                self._query_matched_traces = matched_traces
            
            if matching:
                dict_iter[querystring] = matching
            matchsupport= matchingcount/sample_size
            if matchsupport< supp:
                return False
            else:
                # for trace_index, value in matching.items():
                #     if querystring not in dict_iter:
                #         dict_iter[querystring]= {}
                #     dict_iter[querystring][trace_index]= value
                
                return True
            
    def match_sample_distributed(self, sample, supp, complete_test=False, dict_iter = None,
                     patternset = None, parent_dict = None, max_query_length = -1, chunk_id = None):
        """
            Checks whether the query matches a given sample with given support.

            Determines and sets the support of the query regarding the sample
            if a full test is performed. Stores indices of tested traces
            depending on the match test result in _query_matched_traces or
            _query_not_matched_traces.

            Args:
                sample: Sample instance.

                supp: Float between 0 and 1 which describes the requested
                    support.

                complete_test: Boolean which indicates whether all traces
                    should be tested or not. In the latter case the loop over
                    traces will stop if either the support is fulfilled or can
                    not be fulfilled anymore.
                
                dict_iter= dict_iter (dictionary): nested dictionary for each query and trace 
                the last matching position is value. Default None

                patternset: set of types occurring twice in at least one trace. Default None

                parent_dict: Dictionary containing parent query to each querystring. Default None

            Returns:
                True iff the query matches the given sample with given supp.

            Raises:
                EmptySampleError: The given sample is empty.
                InvalidQuerySupportError: Supp is <0 or >1.
        """
        if self._query_matchtest == 'regex':
            return Query.match_sample(self, sample=sample, supp=supp, complete_test=complete_test)
        
        if self._query_matchtest == 'smarter':
            sample_size = sample._sample_size
            querystring = self._query_string
            if max_query_length != -1 and self._query_string_length > max_query_length:
                return False, dict_iter, chunk_id, parent_dict, self._query_matched_traces

            if self._query_matched_traces:
                trace_list = self._query_matched_traces
                for trace in trace_list:
                    if trace < sample_size:
                        if not sample._sample[trace]:
                            self._query_matched_traces.remove(trace)
                            if querystring.count('$')!=0 and dict_iter:
                                dict_iter[querystring][trace]= -1
                            elif dict_iter:
                                dict_iter[querystring].pop(trace)
                if len(self._query_matched_traces)/sample_size >= supp:
                    return True, dict_iter, chunk_id, parent_dict, self._query_matched_traces
                elif len(self._query_matched_traces) + sample_size - trace_list[-1] < ceil(supp*sample_size):
                    return False, dict_iter, chunk_id, parent_dict, self._query_matched_traces

            matching = self._matching_smarter_multidim(sample=sample, supp =supp, dict_iter=dict_iter,
                                                           patternset=patternset,  parent_dict=parent_dict)
            if querystring.count('$')!=0:
                matchingcount= len(matching)
                self._query_matched_traces = list(matching.keys())
            else:
                matchingcount=0
                matched_traces = []
                for key, value in matching.items():
                    if value != -1:
                        matched_traces.append(key)
                        matchingcount +=1
                self._query_matched_traces = matched_traces

            if matching:
                dict_iter[querystring] = matching
            matchsupport= matchingcount/sample_size
            if matchsupport < supp:
                return False, dict_iter, chunk_id, parent_dict, self._query_matched_traces
            else:
                return True, dict_iter, chunk_id, parent_dict, self._query_matched_traces

    def match_trace_regex(self, trace:str, regex:Pattern) -> Match[str]|None:
        """
            Checks whether the query matches a given trace.

            Args:
                trace: String which represents a trace as described in Sample.

                regex: Regex Object of _query_string_regex

            Returns:
                A MatchObject as witness iff the query matches the given trace
                and None otherwise.
        """
        return Query.match_trace_regex(self, trace=trace, regex=regex)

    ##################################################

    
    def _matching_smarter_multidim(self, sample,supp, dict_iter, patternset,  parent_dict):
        """Matches a query against all traces in the sample.

        Args:
            sample: Sample instance.
            supp:Float between 0 and 1 which describes the requested support. Default: 1

            dict_iter (dictionary): nested dictionary for each query and trace 
                the last matching position is value. Default None

            patternset: set of types occurring twice in at least one trace. Default None

            parent_dict: Dictionary containing parent query to each querystring. Default None

        Returns:
            Trace dictionary containing trace index as keys and a dictionary of groups with the matched string and span as values.
        """
        querystring = self._query_string
        query_list = self.get_query_list()
        trace_split_list = sample.get_sample_list_split()
        if querystring in parent_dict:
            parent = parent_dict[querystring]
        else:
            parent = self._parent()
            parent.set_query_matchtest('smarter')
            parent_dict[querystring] = parent
        parentstring = parent._query_string
        parent_list = parent.get_query_list()
        trace_matches={}
        sample_size = sample._sample_size
        # self.set_pos_last_type_and_variable()
        # last_positions = self._pos_last_type_and_variable
        
        if not self._query_matched_traces:
            traces_to_match = list(range(sample_size))
            matched_traces = []
        else:
            matched_traces = self._query_matched_traces
            traces_to_match = list(range(matched_traces[-1]+1, sample_size))
            for trace in self._query_matched_traces:
                if trace in dict_iter[querystring]:
                    trace_matches[trace]= dict_iter[querystring][trace]

        if querystring.count('$x') == 0:
            num_trace_match = len(traces_to_match) + len(matched_traces)
            # for trace_idx, trace in enumerate(sample._sample):
            for trace_idx in traces_to_match:
                if num_trace_match/sample_size < supp:
                    break
                if not sample._sample[trace_idx]:
                    idx = -1
                else:
                    idx = self._smart_trace_match_multidim(querystring ,trace_split_list[trace_idx], trace_idx, dict_iter, query_list)
                if trace_idx not in trace_matches:
                    trace_matches[trace_idx]= {}
                trace_matches[trace_idx]= idx
                if idx == -1:
                    num_trace_match -=1

            return trace_matches

        var_count = querystring.count('$x')
        cur_count = 0
        var_int = -1
        for event in query_list:
            for dom, letter in enumerate(event.split(';')[:-1]):
                if '$x' in letter:
                    if int(letter[2:]) > var_int:
                        var_int = int(letter[2:])
                        var_domain = dom
                    cur_count +=1
                    if cur_count == var_count:
                        break

        if parentstring.count('$')==0:
            if not parentstring:
                parent_traces = list(range(sample_size))
            else:
                if parentstring in dict_iter:
                    parent_traces= list(dict_iter[parentstring].keys())
                    for stream in traces_to_match:
                        if stream not in parent_traces:
                            parent_match = parent._matching_smarter_multidim(sample, supp, dict_iter, patternset, parent_dict)
                            dict_iter[parentstring] = parent_match
                            parent_traces= list(parent_match.keys())
                            break
                else:
                    parent_match = parent._matching_smarter_multidim(sample,supp,  dict_iter, patternset, parent_dict)
                    dict_iter[parentstring] = parent_match
                    parent_traces= list(parent_match.keys())

            trace_list= [trace for trace in traces_to_match if trace in parent_traces]
            num_trace_match = len(trace_list) + len(matched_traces)
            for trace in trace_list:
                if num_trace_match/sample_size < supp:
                    break
                if not sample._sample[trace]:
                    num_trace_match-=1
                    continue
                if var_domain in patternset:
                    for letter in patternset[var_domain][trace]:
                        letter_querystring = querystring.replace('$x0', letter)
                        if letter_querystring in dict_iter:
                            if trace in dict_iter[letter_querystring]:
                                if sample._sample[trace]:
                                    if dict_iter[letter_querystring][trace] !=-1:
                                        if trace not in trace_matches:
                                            trace_matches[trace]= {}
                                        trace_matches[trace][(letter,)]= dict_iter[letter_querystring][trace]
                            else:
                                idx= self._smart_trace_match_multidim(letter_querystring, trace_split_list[trace], trace, dict_iter)
                                if idx != -1:
                                    if trace not in trace_matches:
                                        trace_matches[trace]= {}
                                    trace_matches[trace][(letter,)]= idx
                        else:
                            idx= self._smart_trace_match_multidim(letter_querystring, trace_split_list[trace], trace, dict_iter)
                            if idx != -1:
                                if trace not in trace_matches:
                                    trace_matches[trace]= {}
                                trace_matches[trace][(letter,)]= idx

                if trace not in trace_matches:
                    num_trace_match-=1
            return trace_matches

        else:
            #parent = MultidimQuery()
            #parent.set_query_string(parentstring, recalculate_attributes=False)
            parent_set = set()
            for event in parent_list:
                for symbol in event.split(';'):
                    if symbol.count('$') !=0:
                        parent_set.add(symbol[1:])
            parent_variables=sorted(list(parent_set))
            if parentstring in dict_iter:
                parent_traces= list(dict_iter[parentstring].keys())
                for stream in traces_to_match:
                    if stream not in parent_traces:
                        parent_match = parent._matching_smarter_multidim(sample, supp, dict_iter, patternset, parent_dict)
                        dict_iter[parentstring] = parent_match
                        parent_traces= list(parent_match.keys())
                        break
                
            else:
                parent_match = parent._matching_smarter_multidim(sample, supp, dict_iter, patternset, parent_dict)
                dict_iter[parentstring] = parent_match
                parent_traces= list(parent_match.keys())
            
            # trace_list= [trace for trace in traces_to_match if trace in parent_traces]
            # trace_list= [trace for trace in traces_to_match if trace in parent_traces]
            trace_list = set(traces_to_match) & set(parent_traces)
            num_trace_match = len(trace_list) + len(matched_traces)
            # trace_list= parent_traces
            # num_trace_match = len(trace_list)
            for trace in trace_list:
                if num_trace_match/sample_size < supp:
                    break
                if not sample._sample[trace]:
                    num_trace_match-=1
                    continue
                group_list= list(dict_iter[parentstring][trace].keys())
                for group in group_list:
                    letter_querystring=querystring
                    assert len(group) == len(parent_variables)
                    for val, letter in enumerate(group):
                        letter_querystring = letter_querystring.replace(f'${parent_variables[val]}', letter)
                    if letter_querystring.count('$')>0 and var_domain in patternset:

                        for letter in patternset[var_domain][trace]:
                            letter_querystring2 = letter_querystring.replace(f'$x{len(group)}', letter)
                            if letter_querystring2 in dict_iter:
                                if trace in dict_iter[letter_querystring2]:
                                    if dict_iter[letter_querystring2][trace] !=-1:
                                        if trace not in trace_matches:
                                            trace_matches[trace]= {}
                                        trace_matches[trace][group + (letter,)]= dict_iter[letter_querystring2][trace]

                                else:
                                    idx = self._smart_trace_match_multidim(letter_querystring2, trace_split_list[trace], trace, dict_iter)
                                    if idx != -1:
                                        if trace not in trace_matches:
                                            trace_matches[trace]= {}
                                        trace_matches[trace][group + (letter,)]= idx

                            else:
                                idx = self._smart_trace_match_multidim(letter_querystring2, trace_split_list[trace], trace, dict_iter)
                                if idx != -1:
                                    if trace not in trace_matches:
                                        trace_matches[trace]= {}
                                    trace_matches[trace][group + (letter,)]= idx

                    else:
                        if letter_querystring in dict_iter:
                            if trace in dict_iter[letter_querystring]:
                                if dict_iter[letter_querystring][trace] !=-1 and sample._sample[trace]:
                                    if trace not in trace_matches:
                                        trace_matches[trace]= {}
                                    trace_matches[trace][group]= dict_iter[letter_querystring][trace]
                            else:
                                idx= self._smart_trace_match_multidim(letter_querystring, trace_split_list[trace], trace, dict_iter)
                                if idx != -1:
                                    if trace not in trace_matches:
                                        trace_matches[trace]= {}
                                    trace_matches[trace][group]= idx
                        else:
                            idx = self._smart_trace_match_multidim(letter_querystring, trace_split_list[trace], trace, dict_iter)
                            if idx != -1:
                                if trace not in trace_matches:
                                    trace_matches[trace]= {}
                                trace_matches[trace][group]= idx

                if trace not in trace_matches:
                    num_trace_match-=1
            return trace_matches

    def _smart_trace_match_multidim(self, querystring, trace_split, trace_idx, dict_iter, query_split=None):
        """Given a trace and a querystring the matching position is calculated and in case of a match
        the dict_iter is updated.

        Args:
            querystring (String): querystring for query
            trace (List): trace list from the sample
            trace_idx (int): index of the given trace
            dict_iter (dictionary): nested dictionary for each query and trace the last matching position is value.

        Returns:
            Last matching position as integer. -1 if there
            is no match.
        """
        if not query_split:
            query_split = querystring.split()
        # trace_split = trace.split()
        domain_cnt= query_split[0].count(';')
        gen_event= ';' * domain_cnt
        last_event= query_split[-1]
        non_empty_domains = self.non_empty_domain(last_event)
        parentstring = ' '.join(query_split[:-1])
        last_event_split = last_event.split(';')
        if non_empty_domains:
            last_non_empty = non_empty_domains[-1]
        if not parentstring:
            if len(non_empty_domains) <=1:
                parentstring = gen_event
            else:
                parentstring= ';'.join(last_event_split[:last_non_empty])+ ';' + ';'.join(last_event_split[last_non_empty+1:]) +';'

        else:
            if len(non_empty_domains)> 1:
                parentstring = parentstring + ' ' + ';'.join(last_event_split[:last_non_empty])+ ';' + ';'.join(last_event_split[last_non_empty+1:]) +';'
        if querystring not in dict_iter:
            dict_iter[querystring]= {}
        if querystring == gen_event:
            dict_iter[querystring][trace_idx]=0
            return 0
        if parentstring not in dict_iter:
            idx = self._smart_trace_match_multidim(parentstring,trace_split, trace_idx, dict_iter)
        if trace_idx in dict_iter[parentstring]:
            parent_end_pos = dict_iter[parentstring][trace_idx]
        else:
            idx= self._smart_trace_match_multidim(parentstring, trace_split, trace_idx, dict_iter)
            parent_end_pos = dict_iter[parentstring][trace_idx]
        if parentstring == gen_event:
            # domain_trace = " ".join([event.split(';')[last_non_empty] for event in trace_split])
            domain_trace_split = [event.split(';')[last_non_empty] for event in trace_split]
            domain_type = last_event_split[last_non_empty]
            # new_domain_trace = domain_trace.replace('  ', ' ; ')
            # while new_domain_trace!= domain_trace:
            #     domain_trace = new_domain_trace
            #     new_domain_trace = domain_trace.replace('  ', ' ; ')
            # domain_trace = new_domain_trace
            # if len(domain_trace.lstrip()) != len(domain_trace):
            #     domain_trace = '; ' + domain_trace
            # if len(domain_trace.rstrip()) != len(domain_trace):
            #     domain_trace = domain_trace + ' ;'
            # domain_trace_split = domain_trace.split()
            if domain_type in domain_trace_split:
                end_pos = domain_trace_split.index(domain_type)
            else:
                end_pos = -1
            dict_iter[querystring][trace_idx]= end_pos
            return end_pos
        if parent_end_pos !=-1:
            if len(non_empty_domains) == 1:
                # domain_trace = " ".join([event.split(';')[last_non_empty] for event in trace_split])
                domain_trace_split = [event.split(';')[last_non_empty] for event in trace_split]
                domain_type= last_event_split[last_non_empty]

                # new_domain_trace = domain_trace.replace('  ', ' ; ')
                # while new_domain_trace!= domain_trace:
                #     domain_trace = new_domain_trace
                #     new_domain_trace = domain_trace.replace('  ', ' ; ')
                # domain_trace = new_domain_trace
                # if len(domain_trace.lstrip()) != len(domain_trace):
                #     domain_trace = '; ' + domain_trace
                # if len(domain_trace.rstrip()) != len(domain_trace):
                #     domain_trace = domain_trace + ' ;'
                # domain_trace_split = domain_trace.split()
                domain_trace_list = domain_trace_split[parent_end_pos+1:]

                if domain_type and domain_type in domain_trace_list:
                    idx = domain_trace_list.index(domain_type)
                else:
                    idx = -1
            else:
                if trace_split[parent_end_pos].split(';')[last_non_empty] and trace_split[parent_end_pos].split(';')[last_non_empty] == last_event_split[last_non_empty]:
                    end_pos = parent_end_pos
                    dict_iter[querystring][trace_idx]= end_pos
                    return end_pos
                else:
                    remaining_trace = ' '.join(trace_split[parent_end_pos+1:])
                    remaining_trace_split = remaining_trace.split()
                    for i, dom in enumerate(non_empty_domains):
                        # domain_trace = " ".join([event.split(';')[dom] for event in remaining_trace_split])
                        domain_trace_split = [event.split(';')[dom] for event in remaining_trace_split]
                        # new_domain_trace = domain_trace.replace('  ', ' ; ')
                        # while new_domain_trace!= domain_trace:
                        #     domain_trace = new_domain_trace
                        #     new_domain_trace = domain_trace.replace('  ', ' ; ')
                        # domain_trace = new_domain_trace
                        # if len(domain_trace.lstrip()) != len(domain_trace):
                        #     domain_trace = '; ' + domain_trace
                        # if len(domain_trace.rstrip()) != len(domain_trace):
                        #     domain_trace = domain_trace + ' ;'
                        # domain_trace_split = domain_trace.split()
                        domain_type= last_event_split[dom]
                        domain_trace_list = domain_trace_split
                        if domain_type in domain_trace_list:
                            idx_list = {i for i, ltr in enumerate(domain_trace_list) if ltr == domain_type}
                        else:
                            idx = -1
                            break
                        if i == 0:
                            index_set = idx_list
                        else:
                            index_set = index_set & idx_list
                            if not index_set:
                                idx = -1
                                break
                        if dom == last_non_empty:
                            idx = min(index_set)

            if idx != -1:
                if trace_idx not in dict_iter[querystring]:
                    dict_iter[querystring][trace_idx]= {}
                end_pos = dict_iter[parentstring][trace_idx] + idx +1
                dict_iter[querystring][trace_idx]= end_pos
                return end_pos
            else:
                dict_iter[querystring][trace_idx]=-1
                return -1
        else:
            dict_iter[querystring][trace_idx]=-1
            return -1
    
    ##################################################

    def query_string_to_normalform(self, check_consistency=False):
        """
            Converts the _query_string into normalform.

            A query string with neither gap constraints nor a global window
            size is in normalform iff it does not contain non-repreated
            variables. This function deletes all non-repeated variables and
            merges the local window sizes.

            Example for 2-dimensional query:
                input query string: $x;a; $y;$z;
                imput local window sizes [(-1,-1),(0,0),(0,2),(0,0),(-1,-1)]
                deleting $x leads to: ;a; $y;$z; and [(1,1),(0,2),(0,0),(-1,-1)]
                deleting $y leads to: ;a; ;$z; and [(1,1),(1,3),(-1,-1)]
                deleting $z leads to: ;a; and [(1,1),(2,4)]

            Note that deleting a non-repeated variable not necessarily leads to
            the deletion of the corresponding semicolon due to readability.
            Semicolons are deleted iff they are part of an empty event, e.g.
            "$x;a; $y;$z;" leads to ";a;" instead of ";a; ;;".

            Raises:
                InconsistentQueryError: If the query has gap constraints or a
                    global window size is given.
        """
        LOGGER.debug('query_string_to_normalform - Started')
        if len(self._query_gap_constraints)>0 or self._query_windowsize_global>-1:
            raise InconsistentQueryError('Inconsistens Query: Query can not be in normalform while having gap constraints or a global window size.')
        if self._query_is_in_normalform is True:
            return

        self.set_query_repeated_variables()
        query_string_nf = copy(self._query_string)
        missing_attributes_count = 0
        pos = 0
        while True:
            if pos >= len(query_string_nf) -1:
                break
            if query_string_nf[pos] == "$":
                variable = query_string_nf[pos+1:]
                variable = variable.split(';')[0]

                LOGGER.debug('query_string_to_normalform - Current variable %s', variable)
                # Check whether the variable occurs more than once in the query
                # string. If not, it has to be deleted and local window sizes
                # have to be edited, if they are given.
                if variable not in self._query_repeated_variables:
                    LOGGER.debug('query_string_to_normalform - Current variable %s is not repeated and has to be deleted', variable)
                    # Delete the repeated variable in the query string and
                    # build the new query string
                    query_string_nf_prefix = query_string_nf[0:pos-1]
                    query_string_nf_suffix = query_string_nf[pos+len(variable)+1:]
                    if pos == 0:
                        query_string_nf = ";" + query_string_nf[len(variable)+2:]
                    else:
                        if query_string_nf[len(query_string_nf_prefix)]==';':
                            query_string_nf = query_string_nf_prefix + ";" + query_string_nf_suffix
                        else:
                            query_string_nf = query_string_nf_prefix + " " + query_string_nf_suffix
                    query_string_nf_string_length = query_string_nf.count(" ")+1
                    pos = pos-1

                    # Update the local window sizes
                    if len(self._query_windowsize_local) > 0:
                        LOGGER.debug('query_string_to_normalform - Local window size tuples have to be updated')
                        if pos <= 0:
                            event_index = -1
                            tuple_index = -1
                            window_size_tuple_at_first_index = list(self._query_windowsize_local[0])
                            window_size_tuple_at_second_index = list(self._query_windowsize_local[1])
                        else:
                            event_index = query_string_nf_prefix.count(" ")+query_string_nf[pos].count(" ")+1
                            tuple_index = query_string_nf_prefix.count(';')+1-missing_attributes_count
                            # Variable at the beginning of an event was deleted
                            if query_string_nf[pos].count(" ")==1:
                                tuple_index = query_string_nf_prefix.count(';')-missing_attributes_count
                            window_size_tuple_at_first_index = list(self._query_windowsize_local[tuple_index])
                            window_size_tuple_at_second_index = list(self._query_windowsize_local[tuple_index+1])

                        # Variable at first attribute in query string is deleted
                        if event_index == -1:
                            LOGGER.debug('query_string_to_normalform - Update the first local window size tuple')
                            # Variable is the last event in the query string
                            if len(query_string_nf)==0:
                                if window_size_tuple_at_second_index[0] == -1:
                                    new_lower_bound = window_size_tuple_at_first_index[0] + 1
                                    new_upper_bound = window_size_tuple_at_first_index[1] + 1
                                else:
                                    new_lower_bound = window_size_tuple_at_first_index[0] + window_size_tuple_at_second_index[0] + 1
                                    new_upper_bound = window_size_tuple_at_first_index[1] + window_size_tuple_at_second_index[1] + 1
                            else:
                                if window_size_tuple_at_first_index[0] == -1:
                                    new_lower_bound = window_size_tuple_at_second_index[0] + 1
                                    new_upper_bound = window_size_tuple_at_second_index[1] + 1
                                else:
                                    new_lower_bound = window_size_tuple_at_first_index[0] + window_size_tuple_at_second_index[0] + 1
                                    new_upper_bound = window_size_tuple_at_first_index[1] + window_size_tuple_at_second_index[1] + 1
                        # Variable at last attribute in query string is deleted
                        elif query_string_nf_string_length==event_index and tuple_index+2 == len(self._query_windowsize_local):
                            LOGGER.debug('query_string_to_normalform - Update the last local window size tuple')
                            if window_size_tuple_at_second_index[0] == -1:
                                new_lower_bound = window_size_tuple_at_first_index[0] + 1
                                new_upper_bound = window_size_tuple_at_first_index[1] + 1
                            else:
                                new_lower_bound = window_size_tuple_at_first_index[0] + window_size_tuple_at_second_index[0] + 1
                                new_upper_bound = window_size_tuple_at_first_index[1] + window_size_tuple_at_second_index[1] + 1
                        else:
                            new_lower_bound = window_size_tuple_at_first_index[0] + window_size_tuple_at_second_index[0] + 1
                            new_upper_bound = window_size_tuple_at_first_index[1] + window_size_tuple_at_second_index[1] + 1
                        window_size_tuple = (new_lower_bound, new_upper_bound)

                        LOGGER.debug('query_string_to_normalform - Build new list of local window size tuples')
                        updated_windowsize_local = []
                        count = 0
                        while len(self._query_windowsize_local) > 0:
                            if count == 0 and tuple_index == -1:
                                LOGGER.debug('query_string_to_normalform - Add new first local window size tuple')
                                updated_windowsize_local.append(window_size_tuple)
                                self._query_windowsize_local.pop(0)
                                self._query_windowsize_local.pop(0)
                                count = count+2
                            elif count == tuple_index:
                                LOGGER.debug('query_string_to_normalform - Add new local window size tuple')
                                updated_windowsize_local.append(window_size_tuple)
                                self._query_windowsize_local.pop(0)
                                self._query_windowsize_local.pop(0)
                                count = count+2
                            else:
                                LOGGER.debug('query_string_to_normalform - Add unchanged local window size tuple')
                                updated_windowsize_local.append(self._query_windowsize_local[0])
                                self._query_windowsize_local.pop(0)
                                count = count+1
                        self.set_query_windowsize_local(updated_windowsize_local, True)
                        missing_attributes_count = missing_attributes_count+1

                    self.set_query_string(query_string_nf, to_regex=False)
                    LOGGER.debug('query_string_to_normalform - Current variable %s is now deleted', variable)
                else:
                    pos = pos+len(variable)
            pos = pos + 1

        # Delete empty events
        event_list = query_string_nf.split(" ")
        query_string_nf_reduced = ""
        for i in range(0,len(event_list)):
            if len(event_list[i])>event_list[i].count(";"):
                query_string_nf_reduced = query_string_nf_reduced + event_list[i] + " "
        if query_string_nf_reduced!="":
            query_string_nf_reduced = query_string_nf_reduced[:len(query_string_nf_reduced)-1]

        # Variables are renamed in order of first appearance to '$x0;','$x1;'..
        gen_variables=[]
        query_string_nf_copy = copy(query_string_nf_reduced)
        while len(query_string_nf_copy)>0:
            if query_string_nf_copy[0] == "$":
                variable = query_string_nf_copy.split(';')[0]+";"
                if variable not in gen_variables:
                    gen_variables.append(variable)
                query_string_nf_copy = query_string_nf_copy[len(variable):]
            else:
                query_string_nf_copy = query_string_nf_copy[1:]
        vars_nf= ['$x_' + str(i) + ';' for i in range(len(gen_variables))]
        new_vars = { v:k for k,v in dict(zip( vars_nf, gen_variables)).items()}
        for variable, new_variable in new_vars.items():
            query_string_nf_reduced = query_string_nf_reduced.replace(variable, new_variable)

        self.set_query_string(query_string_nf_reduced.replace('_', ''))
        self.set_query_is_in_normalform(True)

        if check_consistency is True:
            self.check_consistency()

        LOGGER.debug('query_string_to_normalform - Query string in normalform: %s', self._query_string)
        LOGGER.debug('query_string_to_normalform - Finished')

    def query_string_to_regex(self) -> None:
        """
            Sets a regular expression for the current _query_string.

            Encoding:
                type: type
                first occurence of a variable with name var:
                    (?P<var>[^\\s;]+)
                further occurences of var:
                    (?P=var[^\\s;]+)
                whitespace: \\s
                gap constraint gap and local window size (lower, upper):
                (((?:[^\\s; gap]+;){event_dim})\\s){lower,upper}

            Note that local window size tuple range over single attributes
            instead of whole events unless _query_event_dimension equals 1.
            Hence care must be taken during the handling of local window size
            tuple.

            Exp.: Let _query_event_dimension=2, query string in nf be "a;; ;a;"
            and the local window size tuples [(-1,-1),(2,4),(-1,-1)]. This
            leads to:
            a;(?:[^\\s;]+;){1,1}\\s(((?:[^\\s;]+;){2})\\s){0,1}(?:[^\\s;]+;){1,1}a;
            whereby
                - "a;(?:[^\\s;]+;){1,1}" denotes the first event
                - "(((?:[^\\s;]+;){2})\\s){0,1}" denotes the gap consisting of
                0 or 1 occurences of exactly two attributes
                - "(?:[^\\s;]+;){1,1}a;" denotes the last event.

            Raises:
                InconsistentQueryError: If the number of gap constraints or
                    local window sizes do not match the query string length.

                QueryRegexError: Gap constraint or local window size handling
                    went wrong.
        """
        LOGGER.debug('query_string_to_regex - Started')
        regex_string = ""
        query_string = copy(self._query_string)
        pos = 0

        var_dict_regex = {}
        var_dict_regex_backref = {}
        var_dict_count = {}
        for var in self._query_repeated_variables:
            var_dict_regex[var] = "(?P<" + str(var) + ">[^\\s;]+)"
            var_dict_regex_backref[var] = "(?P=" + str(var) + ")"
            var_dict_count[var] = 0

        # Index of current local window size tuple or gap constraint
        index_lws = 0
        index_gc = 0

        while True:
            # Function reached the end of the query string
            if pos > len(query_string)-1:
                # Handling of last local window size tuple if necessary
                if len(self._query_windowsize_local) > 0:
                    LOGGER.debug('query_string_to_regex - Reached the end of query string')
                    index_lws = len(self._query_windowsize_local)-1
                    lower_bound = list(self._query_windowsize_local[index_lws])[0]-1
                    upper_bound = list(self._query_windowsize_local[index_lws])[1]-1
                    if lower_bound > -1 and upper_bound > -1:
                        assert lower_bound<=upper_bound
                        LOGGER.debug('query_string_to_regex - Reached the end of query string: Add last local window size tuple to regex')
                        if pos > 0:
                            regex_string = regex_string + "\\s"
                        if self._query_event_dimension > 1:
                            lower_bound = (lower_bound+1 - self._query_event_dimension)//self._query_event_dimension
                            upper_bound = (upper_bound+1 - self._query_event_dimension)//self._query_event_dimension
                            if lower_bound>-1 and upper_bound>0:
                                regex_string = regex_string + "(((?:[^\\s;]+;){" + str(self._query_event_dimension) + "})\\s){" + str(lower_bound) + "," + str(upper_bound)+ "}" + "(?:[^\\s;]+;){" + str(self._query_event_dimension) + "}"
                        else:
                            regex_string = regex_string + "(?:[^\\s;]+;\\s){" + str(lower_bound) + "," + str(upper_bound) + "}" + "[^\\s;]+;"

                    LOGGER.debug('query_string_to_regex - Reached the end of query string: Final regex: %s', regex_string)
                break
            # Start regex_string with the first local window size tuple if
            # local window sizes are given and their value is greater than -1
            if pos == 0 and query_string[pos]!=";" and len(self._query_windowsize_local) > 0:
                LOGGER.debug('query_string_to_regex - Begin of query string')
                lower_bound = list(self._query_windowsize_local[0])[0]
                upper_bound = list(self._query_windowsize_local[0])[1]
                if lower_bound > -1 and upper_bound > -1:
                    LOGGER.debug('query_string_to_regex - Begin of query string: Start regex with first local window size tuple')
                    if self._query_event_dimension > 1:
                        assert lower_bound%self._query_event_dimension == 0
                        assert upper_bound%self._query_event_dimension == 0
                        if lower_bound > 0:
                            lower_bound = int(lower_bound/self._query_event_dimension)
                        upper_bound = int(upper_bound/self._query_event_dimension)
                    regex_string = "(((?:[^\\s;]+;){" + str(self._query_event_dimension) + "})\\s){" + str(lower_bound) + "," + str(upper_bound)+ "}"
                index_lws = index_lws+1
                LOGGER.debug('query_string_to_regex - Begin of query string: Current regex: %s', regex_string)
            # Handling of gap
            if query_string[pos] == " ":
                LOGGER.debug('query_string_to_regex - Current position %s in %s is gap', str(pos), query_string)
                gap_constraint_bool = False
                window_size_bool = False
                lower_bound = -1
                upper_bound = -1
                gap_constraint_string = ""
                # Include gap constraints and local window sizes in regex_string if they exist
                if len(self._query_gap_constraints) > 0:
                    LOGGER.debug('query_string_to_regex - Current position is gap: Build gap constraint regex string')
                    if len(self._query_gap_constraints) != (self._query_string_length*self._query_event_dimension)-1:
                        raise InconsistentQueryError('Number of gap constraints do not match the query string length.')
                    #gap_constraint = self._query_gap_constraints[index-1]
                    gap_constraint = self._query_gap_constraints[index_gc]
                    gap_constraint_string = ""
                    for gap in gap_constraint:
                        gap_constraint_string = gap_constraint_string + str(gap)
                    gap_constraint_bool = True
                if len(self._query_windowsize_local) > 0:
                    LOGGER.debug('query_string_to_regex - Current position is gap: Build local window size tuple regex string')
                    if len(self._query_windowsize_local) > (self._query_string_length*self._query_event_dimension)+1:
                        raise InconsistentQueryError('Number of local window sizes do not match the query string length.')
                    lower_bound = list(self._query_windowsize_local[index_lws])[0]
                    upper_bound = list(self._query_windowsize_local[index_lws])[1]
                    window_size_bool = True
                if gap_constraint_bool is True and window_size_bool is True:
                    LOGGER.debug('query_string_to_regex - Current position is gap: Add gap constraint and local window size tuple')
                    if self._query_event_dimension > 1:
                        if lower_bound > 0:
                            lower_bound = int(lower_bound/self._query_event_dimension)
                        upper_bound = int(upper_bound/self._query_event_dimension)
                    regex_string = regex_string + "\\s(((?:[^\\s;" + gap_constraint_string + "]+;){" + str(self._query_event_dimension) + "})\\s){" + str(lower_bound) + "," + str(upper_bound)+ "}"
                    pos = pos+1
                elif gap_constraint_bool is True and window_size_bool is not True:
                    LOGGER.debug('query_string_to_regex - Current position is gap: Add gap constraint')
                    # Since no local window sizes are given we add an arbitrary long gap.
                    # By using the question mark we force the match to be the shortest possible match
                    regex_string = regex_string + "\\s(((?:[^\\s;" + gap_constraint_string + "]+;){" + str(self._query_event_dimension) + "})\\s)*?"
                    pos = pos+1
                elif gap_constraint_bool is not True and window_size_bool is True:
                    LOGGER.debug('query_string_to_regex - Current position is gap: Add local window size tuple')
                    if self._query_event_dimension > 1:
                        if lower_bound > 0:
                            lower_bound = int(lower_bound/self._query_event_dimension)
                        upper_bound = int(upper_bound/self._query_event_dimension)
                    regex_string = regex_string + "\\s(((?:[^\\s;]+;){" + str(self._query_event_dimension) + "})\\s){" + str(lower_bound) + "," + str(upper_bound)+ "}"
                    pos = pos+1
                elif gap_constraint_bool is not True and window_size_bool is not True:
                    LOGGER.debug('query_string_to_regex - Current position is gap: Add neither gap constraint nor local window size tuple')
                    # Since no local window sizes are given we add an arbitrary long gap.
                    # By using the question mark we force the match to be the shortest possible match
                    regex_string = regex_string + "\\s(((?:[^\\s;]+;){" + str(self._query_event_dimension) + "})\\s)*?"
                    pos = pos+1
                else:
                    raise QueryRegexError('Gap constraint or local window size handling went wrong.')
                index_lws = index_lws+1
                index_gc = index_gc+1
                LOGGER.debug('query_string_to_regex - Current position is gap: Current regex after gap handling: %s', regex_string)
            # Handling of upcoming trailing ';'
            if query_string[pos]==";":
                # Store whether the current position is the beginning of an event
                event_starts_with_semicolon = False
                if pos == 0 or query_string[pos-1]==" ":
                    event_starts_with_semicolon = True
                # Perform a lookahead to detect trailing semicolons
                # In the end: pos>=len(query_string) or query_string[pos]!=;/space
                pos_copy = pos
                pos_copy_first_trailing_semicolon = pos
                lookahead = ""
                while pos < len(query_string):
                    if query_string[pos] == ";" or query_string[pos] == " ":
                        lookahead = lookahead + query_string[pos]
                        pos = pos+1
                    else:
                        break
                # Perform a backwards search in lookahead to determine the
                # position of the last semicolon within the corresponding event
                pos_last_semicolon_in_event = pos-1
                offset_last_event_in_lookahead = 0
                while pos_last_semicolon_in_event > 0:
                    if query_string[pos_last_semicolon_in_event] == " ":
                        break
                    pos_last_semicolon_in_event = pos_last_semicolon_in_event-1
                    offset_last_event_in_lookahead = offset_last_event_in_lookahead +1
                # Perform a backwards search to determine the position of the
                # first semicolon of lookahead in its event
                pos_first_semicolon_in_current_event = 0
                while pos_copy>=0 and query_string[pos_copy]!=" ":
                    if query_string[pos_copy]==";":
                        pos_first_semicolon_in_current_event = pos_first_semicolon_in_current_event+1
                    pos_copy = pos_copy-1
                if pos_copy <=0 and query_string[0]!=";":
                    pos_copy = 1

                #Handling of upcoming trailing ';' without local window sizes
                if len(self._query_windowsize_local) == 0:
                    # Lookahead detected trailing semicolons or whitespaces
                    if len(lookahead)>1:
                        if lookahead == "; ":
                            if pos_copy_first_trailing_semicolon == 0:
                                # Event consists of only one semicolon
                                regex_string = regex_string + "(?:[^\\s;]+;){" + str(1) + "};\\s(((?:[^\\s;]+;){" + str(self._query_event_dimension) + "})\\s)*?"
                            else:
                                regex_string = regex_string + ";\\s(((?:[^\\s;]+;){" + str(self._query_event_dimension) + "})\\s)*?"
                        else:
                            # Complete the current attribute by adding the semicolon
                            complete_attribute = False
                            if pos_copy_first_trailing_semicolon>0:
                                regex_string = regex_string + ";"
                                complete_attribute = True
                                lookahead = lookahead[1:]

                            # Count number of "missing" or at least incomplete events:
                            num_missing_events = lookahead.count(" ")
                            missing_events = False
                            if num_missing_events > 0:
                                missing_events = True

                            # Complete the current event and add empty event(s) if necessary
                            if pos_first_semicolon_in_current_event >= 1:
                                if missing_events is False and event_starts_with_semicolon is True:
                                    # Event starts with trailing semicolons but
                                    # lookahead did not reach the next gap
                                    regex_string = regex_string + "(?:[^\\s;]+;){" + str(lookahead.count(";")) + "}"
                                elif missing_events is True and event_starts_with_semicolon is True:
                                    # Event starts with trailing semicolons and
                                    # lookahead reached the next gap
                                    regex_string = regex_string + "(?:[^\\s;]+;){" + str(self._query_event_dimension) + "}\\s"
                                    if pos >= len(query_string):
                                        #Lookahead reached end of query string, hence last regex should not end with \\s
                                        num_missing_events = num_missing_events-1
                                        if num_missing_events>0:
                                            regex_string = regex_string + "((?:[^\\s;]+;){" + str(self._query_event_dimension) + "}\\s){" + str(num_missing_events) + "}"
                                        regex_string = regex_string + "(?:[^\\s;]+;){" + str(self._query_event_dimension) + "}"
                                    else:
                                        regex_string = regex_string + "((?:[^\\s;]+;){" + str(self._query_event_dimension) + "}\\s){" + str(num_missing_events) + "}"
                                elif missing_events is False and pos<len(query_string) and query_string[pos]!=";" and query_string[pos]!=" ":
                                    # Event neither starts nor ends with trailing semicolons
                                    trailing_semicolon_count = lookahead.count(";")
                                    regex_string = regex_string + "(?:[^\\s;]+;){" + str(trailing_semicolon_count) + "}"
                                else:
                                    # Event does not start but ends with trailing semicolons, i.e. reaches next gap
                                    lookahead_event_list = lookahead.split(" ")
                                    first_partial_event_in_lookahead = lookahead_event_list[0]
                                    last_partial_event_in_lookahead = lookahead_event_list[len(lookahead_event_list)-1]

                                    if len(lookahead_event_list) == 1:
                                        # Lookahead reached the end of query string.
                                        # Complete the current event without \\s at the end
                                        regex_string = regex_string + "(?:[^\\s;]+;){"+ str(len(first_partial_event_in_lookahead)) + "}"
                                    else:
                                        # Complete current event
                                        if len(first_partial_event_in_lookahead) > 0:
                                            regex_string = regex_string + "((?:[^\\s;]+;){"+ str(len(first_partial_event_in_lookahead)) + "}\\s)"
                                        else:
                                            regex_string = regex_string + "\\s"
                                        # Adapt number of missing events if necessary
                                        if last_partial_event_in_lookahead.count(";")<self._query_event_dimension:
                                            num_missing_events = num_missing_events-1
                                        # Add arbitrary gap and missing event(s) if necessary
                                        while num_missing_events > 0:
                                            regex_string = regex_string + "((?:[^\\s;]+;){"+ str(self._query_event_dimension) + "}\\s)*?"
                                            regex_string = regex_string + "((?:[^\\s;]+;){" + str(self._query_event_dimension) + "})\\s"
                                            num_missing_events = num_missing_events-1
                                        # Add last arbitrary gap
                                        regex_string = regex_string + "((?:[^\\s;]+;){"+ str(self._query_event_dimension) + "}\\s)*?"
                                        # Add missing attributes for last partial event if necessary
                                        if last_partial_event_in_lookahead.count(";")>0 and last_partial_event_in_lookahead.count(";")<self._query_event_dimension:
                                            regex_string = regex_string + "(?:[^\\s;]+;){"+ str(len(last_partial_event_in_lookahead)) + "}"
                    # Lookahead detected no trailing semicolons
                    else:
                        # Query string starts with exactly one ";"
                        if pos == 1:
                            LOGGER.debug('query_string_to_regex - Current position %s in %s is begin of query string without trailing semicolons', str(pos), query_string)
                            regex_string = regex_string + "(?:[^\\s;]+;)"
                        # Semicolon is just the end of an attribute
                        else:
                            LOGGER.debug('query_string_to_regex - Current position %s in %s is end of attribute', str(pos), query_string)
                            regex_string = regex_string + ";"
                #Handling of upcoming trailing ';' with local window sizes
                else:
                    # Lookahead detected trailing semicolons or whitespaces
                    if len(lookahead)>1 or list(self._query_windowsize_local[index_lws])[1]>self._query_event_dimension:
                        if lookahead == "; ":
                            if pos_copy_first_trailing_semicolon == 0:
                                # Event consists of only one semicolon
                                regex_string = regex_string + "(?:[^\\s;]+;){" + str(self._query_event_dimension) + "}"
                            else:
                                # Current event has to be completed by adding the final semicolon
                                regex_string = regex_string + ";"
                            # To delegate the gap handling the position is set to the position of the space character
                            pos = pos-1
                        else:
                            lower_bound = list(self._query_windowsize_local[index_lws])[0]
                            upper_bound = list(self._query_windowsize_local[index_lws])[1]
                            index_lws = index_lws + 1

                            # Complete the current attribute by adding the semicolon
                            complete_attribute = False
                            if pos_copy_first_trailing_semicolon>0:
                                regex_string = regex_string + ";"
                                complete_attribute = True

                            # Count number of "missing" or at least incomplete events:
                            missing_events = False
                            if upper_bound>self._query_event_dimension:
                                missing_events = True

                            if missing_events is False:
                                # No events are missing. Complete the current event if necessary
                                if pos_first_semicolon_in_current_event >= 1:
                                    if pos >= len(query_string):
                                        # Lookahead reached the end of the query string, hence
                                        # handling of last window size tuple is needed.
                                        if lower_bound > -1 and upper_bound > -1:
                                            regex_string = regex_string + "(?:[^\\s;]+;){" + str(lower_bound) + "," + str(upper_bound)+ "}"
                                        break
                                    if event_starts_with_semicolon is True:
                                        # Event starts with trailing semicolons but
                                        # lookahead did not reach the next gap
                                        regex_string = regex_string + "(?:[^\\s;]+;){" + str(lookahead.count(";")) + "}"
                                    elif pos<len(query_string) and query_string[pos]!=";" and query_string[pos]!=" ":
                                        # Event neither starts nor ends with trailing semicolons
                                        trailing_semicolon_count = lookahead.count(";")
                                        if complete_attribute is True:
                                            trailing_semicolon_count = trailing_semicolon_count - 1
                                        regex_string = regex_string + "(?:[^\\s;]+;){" + str(trailing_semicolon_count) + "}"
                            else:
                                # Lookahead contains at leat one gap, hence local
                                # window size handling is necessary
                                if pos >= len(query_string):
                                    # Lookahead reached the end of the query string, hence
                                    # handling of last window size tuple is needed.
                                    lower_bound_partial = lower_bound%self._query_event_dimension
                                    upper_bound_partial = upper_bound%self._query_event_dimension
                                    lower_bound = lower_bound-lower_bound_partial
                                    upper_bound = upper_bound-upper_bound_partial
                                    assert lower_bound%self._query_event_dimension == 0
                                    assert upper_bound%self._query_event_dimension == 0
                                    # Complete the current event if necessary
                                    if upper_bound_partial>0:
                                        regex_string = regex_string + "(?:[^\\s;]+;){" + str(lower_bound_partial) + "," + str(upper_bound_partial)+ "}"
                                    lower_bound = (lower_bound+1 - self._query_event_dimension)//self._query_event_dimension
                                    upper_bound = (upper_bound+1 - self._query_event_dimension)//self._query_event_dimension
                                    if self._query_event_dimension == 1:
                                        lower_bound = lower_bound-self._query_event_dimension
                                        upper_bound = upper_bound-self._query_event_dimension
                                    if lower_bound>-1 and upper_bound>0:
                                        if pos_copy_first_trailing_semicolon == 0 and upper_bound_partial == 0:
                                            regex_string = regex_string + "(((?:[^\\s;]+;){" + str(self._query_event_dimension) + "})\\s){" + str(lower_bound) + "," + str(upper_bound)+ "}"
                                        else:
                                            regex_string = regex_string + "\\s(((?:[^\\s;]+;){" + str(self._query_event_dimension) + "})\\s){" + str(lower_bound) + "," + str(upper_bound)+ "}"
                                        regex_string = regex_string + "(?:[^\\s;]+;){" + str(self._query_event_dimension) + "}"
                                    LOGGER.debug('query_string_to_regex - Current position is a placeholder: Current regex after semicolon handling: %s', regex_string)
                                    LOGGER.debug('query_string_to_regex - Reached the end of query string during handling of trailing semicolons')
                                    break
                                # Lookahead did not reach the end of the query string
                                else:
                                    lower_bound_partial = self._query_event_dimension
                                    upper_bound_partial = self._query_event_dimension
                                    if not pos_copy_first_trailing_semicolon <=0:
                                        lower_bound_partial = self._query_event_dimension - pos_first_semicolon_in_current_event
                                        upper_bound_partial = self._query_event_dimension - pos_first_semicolon_in_current_event
                                    lower_bound = lower_bound-lower_bound_partial
                                    upper_bound = upper_bound-upper_bound_partial
                                    # Complete the current event if necessary
                                    if upper_bound_partial>0:
                                        regex_string = regex_string + "(?:[^\\s;]+;){" + str(lower_bound_partial) + "," + str(upper_bound_partial)+ "}"
                                    # The next event after the gap does not start with
                                    # placeholder (";") or consists only of ";"s
                                    if lower_bound%self._query_event_dimension == 0 and upper_bound%self._query_event_dimension == 0:
                                        lower_bound = lower_bound//self._query_event_dimension
                                        upper_bound = upper_bound//self._query_event_dimension
                                        if pos_copy_first_trailing_semicolon == 0 and upper_bound_partial == 0:
                                            regex_string = regex_string + "(((?:[^\\s;]+;){" + str(self._query_event_dimension) + "})\\s){" + str(lower_bound) + "," + str(upper_bound)+ "}"
                                        else:
                                            regex_string = regex_string + "\\s(((?:[^\\s;]+;){" + str(self._query_event_dimension) + "})\\s){" + str(lower_bound) + "," + str(upper_bound)+ "}"
                                    # The next event after the gap starts with placeholder (";")
                                    else:
                                        lower_bound_partial_rest = lower_bound%self._query_event_dimension
                                        upper_bound_partial_rest = upper_bound%self._query_event_dimension
                                        lower_bound = (lower_bound-lower_bound_partial_rest)//self._query_event_dimension
                                        upper_bound = (upper_bound-upper_bound_partial_rest)//self._query_event_dimension
                                        regex_string = regex_string + "\\s(((?:[^\\s;]+;){" + str(self._query_event_dimension) + "})\\s){" + str(lower_bound) + "," + str(upper_bound)+ "}"
                                        regex_string = regex_string + "(?:[^\\s;]+;){" + str(lower_bound_partial_rest) + "," + str(upper_bound_partial_rest)+ "}"
                    # Lookahead detected no trailing semicolons
                    else:
                        # Query string starts with exactly one ";"
                        if pos==1:
                            lower_bound = list(self._query_windowsize_local[0])[0]
                            upper_bound = list(self._query_windowsize_local[0])[1]
                            if lower_bound>-1 and upper_bound>-1:
                                regex_string = regex_string + "(?:[^\\s;]+;){" + str(lower_bound) + "," + str(upper_bound)+ "}"
                            index_lws = index_lws+1
                        else:
                            regex_string = regex_string + ";"
                            index_lws = index_lws+1
            # Handling of a variable
            elif pos < len(self._query_string) and query_string[pos] == "$":
                variable = query_string[pos+1:]
                variable = variable.split(';')[0]
                LOGGER.debug('query_string_to_regex - Current position %s in %s is variable: %s', str(pos), query_string, variable)
                if variable not in self._query_repeated_variables:
                    LOGGER.debug('query_string_to_regex - Current position is variable: non-repeated')
                    regex_string = regex_string + "(?P<" + str(variable) + ">[^\\s;]+)"
                    LOGGER.debug('query_string_to_regex - Current position is variable: non-repeated: Current regex: %s', regex_string)
                elif var_dict_count[variable] == 0:
                    LOGGER.debug('query_string_to_regex - Current position is variable: repeated, first occurence')
                    regex_string = regex_string + str(var_dict_regex[variable])
                    var_dict_count[variable] = 1
                    LOGGER.debug('query_string_to_regex - Current position is variable: repeated, first occurence: Current regex: %s', regex_string)
                else:
                    LOGGER.debug('query_string_to_regex - Current position is variable: repeated')
                    regex_string = regex_string + str(var_dict_regex_backref[variable])
                    LOGGER.debug('query_string_to_regex - Current position is variable: repeated: Current regex: %s', regex_string)
                pos = pos + len(variable) + 1
            # Handling of a type
            else:
                LOGGER.debug('query_string_to_regex - Current position %s in %s is part of a type', str(pos), query_string)
                if pos < len(self._query_string):
                    regex_string = regex_string + str(self._query_string[pos])
                    pos = pos+1
                LOGGER.debug('query_string_to_regex - Current position is part of a type: Current regex after type handling: %s', regex_string)
        self.set_query_string_regex(regex_string)
        LOGGER.debug('query_string_to_regex - Query string %s has final regex %s', self._query_string, regex_string)
        LOGGER.debug('query_string_to_regex - Finished')

    ##################################################

    def _parent(self):
        """Returns the parent query of a given Multidimquery,
            in other words the Multidimquery from which it was generated
            acording to the constrain-based rule set.
            query (MultidimQuery): an instance of MultidimQuery

        Returns:
            parent (MultidimQuery): an instance of MultidimQuery
        """
        querystring= self._query_string
        query_list = self.get_query_list()
        if not querystring:
            return MultidimQuery()
        #self.set_pos_last_type_and_variable()
        # pos_last_type_and_variable= self._pos_last_type_and_variable
        # pos_last_type= pos_last_type_and_variable[0]
        # pos_first_var= pos_last_type_and_variable[1]
        # pos_last_var = pos_last_type_and_variable[2]


        
        var_int = -1
        pos_first_var = -1
        pos_last_var = -1
        pos_last_type = -1
        for pos, event in enumerate(query_list):
            for dom, letter in enumerate(event.split(';')[:-1]):
                if '$x' in letter:
                    if int(letter[2:]) > var_int:
                        var_int = int(letter[2:])
                        pos_first_var = pos
                    elif int(letter[2:]) == var_int:
                        pos_last_var = pos
                elif letter:
                    pos_last_type = pos
                    

        var = False
        letter = False
        domain_cnt = query_list[0].count(';')
        gen_event = ';' *domain_cnt
        if pos_last_type > pos_first_var:
            last_position = pos_last_type
            letter = True

        elif pos_last_type < pos_first_var:
            last_position = pos_last_var
            var = True
        else:
            last_position = pos_last_type
            current_event = query_list[last_position]
            filled_domains = self.non_empty_domain(current_event)
            last_domain =filled_domains[-1]
            last_type = current_event.split(';')[last_domain]
            if last_type.count("$") > 0:
                last_position = pos_last_var
                var = True
            else:
                letter = True

        current_event = query_list[last_position]
        filled_domains = self.non_empty_domain(current_event)
        if var:
            var_numb = -1
        for domain in filled_domains:
            att = current_event.split(';')[domain]
            if letter and '$' not in att:
                last_domain= domain
            if var and '$' in att and int(att.strip('$x')) > var_numb:
                last_domain = domain
                var_numb = int(att.strip('$x'))
        #last_domain =filled_domains[-1]
        last_type = current_event.split(';')[last_domain]
        if len(filled_domains) >1:
            current_event_split = current_event.split(';')
            current_event_split[last_domain] = ''
            current_event = ';'.join(current_event_split)
            #current_event= ';'.join(current_event.split(';')[:last_domain]) +';' + ';'.join(current_event.split(';')[last_domain+1:]) +';'
        else:
            current_event = gen_event

        if current_event != gen_event:
            parentstring = " ".join(query_list[:last_position]) + " "+ current_event + " " + " ".join(query_list[last_position+1:])
        else:
            parentstring = " ".join(query_list[:last_position]) + " "+ " ".join(query_list[last_position+1:])
        if last_type.count('$') !=0:
            if parentstring.count(last_type) == 1:
                last_position = pos_first_var
                current_event = query_list[last_position]
                current_event= current_event.replace(last_type, '')

                if current_event != gen_event:
                    parentstring = " ".join(parentstring.split()[:last_position]) + " "+ current_event + " " + " ".join(parentstring.split()[last_position+1:])
                else:
                    parentstring = " ".join(parentstring.split()[:last_position]) + " "+ " ".join(parentstring.split()[last_position+1:])

        parent=MultidimQuery()
        parentstring = parentstring.strip()
        if parentstring and parentstring != gen_event:
            parent.set_query_string(parentstring.strip(), recalculate_attributes=False)

        return parent


    def non_empty_domain(self, last_event):
        """Returns the number of the last non empty domain.

        Args:
            querystring (String)

        Returns:
            List of domain-numbers that are not empty.
        """
        #last_event = querystring.split()[-1]
        last_event_split = last_event.split(';')
        non_empty_domains= [idx for idx, i in enumerate(last_event_split) if i]

        return non_empty_domains


    def query_pos_dict(self, event_db:dict, sample:MultidimSample, event_dictionary:dict|None=None, trace_list:list = [0]) -> dict:
        """
            Calculates for each given query the instances that each query occures
            in for each trace.

            Args:
                event_db (dictionary): result from function
                    discovery.create_vertical_sequence_database()

                sample (MultidimSample): an instance of MultidimSample.

                event_dictionary (dict): {querystring: trace_id : [list of trace positions]}. Defaults to: None.

                trace_list (List): List of trace id's which should be calculated. Defaults to: [0].

            Returns:
                event_dictionary: {querystring: trace_id : [list of trace positions]}
        """
        
        querystring = self._query_string
        
        
        if querystring:
            sample_set = sample._sample
            if not event_dictionary:
                event_dictionary = {}
            event_dictionary[querystring]= {}
            dim_querystring = querystring.replace(';', '')
            dim_querystring_split = dim_querystring.split()
            symbol_counts = {symbol: dim_querystring.count(symbol) for symbol in set(dim_querystring_split)}
            domain = len(querystring) - len(querystring.lstrip(';'))
            for trace_idx in trace_list:
                trace_matches = True
                cur_trace = sample_set[trace_idx]
                cur_trace_split = cur_trace.split()
                

                # for pos, event in enumerate(querystring_split):
                for pos, event in enumerate(dim_querystring_split):
                    first_domain = False
                    # for domain, symbol in enumerate(event.split(';')[:-1]):
                    
                        # symbol_count = querystring.count(symbol)
                    symbol = event
                    symbol_count = symbol_counts[symbol]
                    if symbol in event_db[domain]:
                        if trace_idx not in event_db[domain][symbol]:
                            break
                        pos_list=event_db[domain][symbol][trace_idx]

                    else:
                        pos_list= []
                        for letter in event_db[domain].keys():
                            if trace_idx in event_db[domain][letter].keys():
                                if len(event_db[domain][letter][trace_idx]) >= symbol_count:
                                    pos_list.extend(event_db[domain][letter][trace_idx])
                            
                    if trace_idx not in event_dictionary[querystring]:
                        event_dictionary[querystring][trace_idx]= [[position] for position in pos_list]
                        first_domain = True
                    else:
                        instances=[]
                        for instance in event_dictionary[querystring][trace_idx]:
                            last_position= instance[-1]
                            if symbol in event_db[domain]:
                                new_positions = pos_list

                            else:
                                # if " ".join(querystring_split[:pos]).count(symbol) == 0:
                                if " ".join(dim_querystring_split[:pos]).count(symbol) == 0:
                                    new_positions= sorted(pos_list)
                                    
                                else:

                                    var_start= dim_querystring_split.index(symbol)
                                    if var_start < len(instance):
                                        trace_event= cur_trace_split[instance[var_start]]
                                        letter = trace_event.split(';')[domain]
                                        new_positions = event_db[domain][letter][trace_idx]
                                    else:
                                        trace_matches = False
                                        event_dictionary[querystring].pop(trace_idx)
                                        break

                            if not first_domain:
                                for new_pos in new_positions[::-1]:
                                    if new_pos > last_position:
                                        instances.append(instance+ [new_pos])
                                        # assert len({len(inst) for inst in instances}) == 1
                                    else:
                                        break
                            else:
                                if last_position in new_positions:
                                    instances.append(instance)
                                    # assert len({len(inst) for inst in instances}) == 1
                        if not trace_matches:
                            break
                        first_domain = True


                        if instances:
                            event_dictionary[querystring][trace_idx] = instances
        return event_dictionary # type: ignore


    def event_db_positions(self, event_db, query_domains, trace_idx, last_position=-1):
        positions = []

        for pos, domain_events in enumerate(query_domains):
            symbol_count = sum(symbol in event_db[domain] for domain, symbol in enumerate(domain_events))
            
            for domain, symbol in enumerate(domain_events):
                if symbol in event_db[domain]:
                    if trace_idx in event_db[domain][symbol]:
                        positions.extend(event_db[domain][symbol][trace_idx])
                        break
                else:
                    for letter, letter_positions in event_db[domain].items():
                        if trace_idx in letter_positions and len(letter_positions[trace_idx]) >= symbol_count:
                            positions.extend(letter_positions[trace_idx])
        
        if last_position >= 0:
            positions = [pos for pos in positions if pos > last_position]
        
        return sorted(positions)

    ##################################################
    def check_consistency(self):
        """
            Checks whether attributes of the current query are consistent.

            Inconsistent states:
                - query string and query string length are incompatible
                - local window sizes are given
                    - but are incompatible with the query string length
                    - and a tuple has lower bound which is greater than upper
                    bound
                    - and lower and upper bound are not equal modulo the event
                    dimension
                - a global window size is given and the query string is in
                normalform
                - gap constraints are given and
                    - their number is incompatible with the query string length
                    - the query string is in normalform

            Raises:
                InvalidQueryStringLengthError: If the query string length is
                    incompatible with the query string.

                InvalidQueryLocalWindowSizeError: If local_window_size is
                    incompatible with query string length.

                InconsistentQueryError: If the query string is in normalform
                    but a global window size or gap constraints are given.

                InvalidQueryGapConstraintError: If the number of gap
                    constraints is incompatible with the query string length.
        """
        if self._query_string == "" and self._query_string_length != 0:
            raise InvalidQueryStringLengthError('CHECK_CONSISTENCY: Query string length has to be zero.')
        elif self._query_string != "":
            expected_query_string_length = self._query_string.count(" ")+1
            if int(self._query_string_length) != int(expected_query_string_length):
                raise InvalidQueryStringLengthError(f'CHECK_CONSISTENCY: Query string length {self._query_string_length} should be {expected_query_string_length}.')

        if len(self._query_windowsize_local)>0:
            if self._query_string == "" and len(self._query_windowsize_local)>1:
                raise InvalidQueryLocalWindowSizeError('CHECK_CONSISTENCY: Number of local window size tuples has to be 1 since the query string is empty.')
            elif self._query_string != "" and self._query_is_in_normalform is False:
                if len(self._query_windowsize_local)!=self._query_string.count(";")+1:
                    raise InvalidQueryLocalWindowSizeError('CHECK_CONSISTENCY: Length of local window size list incompatible with query string length.')
            for window_size_tuple in self._query_windowsize_local:
                if list(window_size_tuple)[0] > list(window_size_tuple)[1]:
                    raise InvalidQueryLocalWindowSizeError('CHECK_CONSISTENCY: Lower bound for local window size tuple is greater than the upper bound.')
                if (list(window_size_tuple)[0]%self._query_event_dimension) != (list(window_size_tuple)[1]%self._query_event_dimension):
                    raise InvalidQueryLocalWindowSizeError('CHECK_CONSISTENCY: Lower and upper bound should be equal modulo the event dimension.')

        if self._query_windowsize_global > -1:
            if self._query_is_in_normalform is True:
                raise InconsistentQueryError('CHECK_CONSISTENCY: A query with a global window size can not be in normalform.')

        if len(self._query_gap_constraints)>0:
            if len(self._query_gap_constraints)!=self._query_string.count(";")-1:
                raise InvalidQueryGapConstraintError('CHECK_CONSISTENCY: Number of gap constraints is incompatible with the query string length.')
            if self._query_is_in_normalform is True:
                raise InconsistentQueryError('CHECK_CONSISTENCY: A query with gap constraints can not be in normalform.')

    def _syntactically_equal(self, other_query:Self) -> bool:
        """
            Checks whether two queries are syntactically equal by comparing the
            following attributes:
                _query_class
                _query_string_length
                _query_string
                _query_event_dimension
                _query_gap_constraints
                _query_windowsize_global
                _query_windowsize_local
                _query_repeated_variables
                _query_is_in_normalform
                _query_typeset

            Note:
                _query_windowsize_local of both query are treated as equal as
                well, if one has (-1,-1) placeholder in front of and behind the
                local windows.

            Args:
                other: Other query to check with.

            Returns:
                True iff they are syntactically equal (equal in all mentioned
                attributes). False otherwise.
            Raise:
                TypeError: Argument is no instance of 'query'

                InvalidQueryGapConstraintEror: One query has not the right
                    amount of gap constraints.
        """
        if not isinstance(other_query, Query):
            raise TypeError("'syntactically equal' is not defined for " + str(type(other_query)))

        if self._query_class != other_query._query_class:
            return False
        if self._query_string_length != other_query._query_string_length:
            return False
        if self._query_event_dimension != other_query._query_event_dimension:
            return False
        if self._query_typeset != other_query._query_typeset:
            return False
        if self._query_repeated_variables != other_query._query_repeated_variables:
            return False
        if self._query_string != other_query._query_string:
            return False
        if len(self._query_gap_constraints) != len(other_query._query_gap_constraints):
            raise InvalidQueryGapConstraintError('Wrong count of gap constraints for current query string length.')
        for index in range(0,len(self._query_gap_constraints)):
            if sorted(self._query_gap_constraints[index]) != sorted(other_query._query_gap_constraints[index]):
                return False
        if self._query_windowsize_global != other_query._query_windowsize_global:
            return False
        if len(self._query_windowsize_local) == len(other_query._query_windowsize_local):
            if self._query_windowsize_local != other_query._query_windowsize_local:
                return False
        else:
            if len(self._query_windowsize_local) < len(other_query._query_windowsize_local):
                if self._query_windowsize_local != other_query._query_windowsize_local[1:len(other_query._query_windowsize_local)-2]:
                    return False
                if other_query._query_windowsize_local[0] != (-1,-1):
                    return False
                if other_query._query_windowsize_local[len(other_query._query_windowsize_local)-1] != (-1,-1):
                    return False
            else:
                if self._query_windowsize_local[1:len(self._query_windowsize_local)-2] != other_query._query_windowsize_local:
                    return False
                if self._query_windowsize_local[0] != (-1,-1):
                    return False
                if self._query_windowsize_local[len(self._query_windowsize_local)-1] != (-1,-1):
                    return False
        return True

    ##################################################

    def init_most_general_query_for_shinohara(self, query_string_length:int, windowsize_local:list, discovery_algorithm:str, sample:MultidimSample, event_dimension:int=1, gap_constraints:list|None=None, query_class:str="normal") -> Self:
        """
            Initializes the most general query (regarding the query string).

            Espacially intended for building the base query string for
            Shinoharas Algorithm.

            Args:
                query_string_length: Length of the query string as an integer.

                windowsize_local: List of integer tuples which represents the
                    local window sizes.

                query_discovery_algorithm: String which contains the name of
                    the choosen discovery algorithm.

                sample: Sample instance.

                event_dimension: Integer which represents the number of
                    attributes per event. Uses 1 as default.

                gap_constraints: List of sets of strings as described in the
                    class Query.

                query_class: A string which describes the query class as
                    described in the class Query.

            Returns:
                The created query instance.
        """
        if query_string_length < 1:
            raise InvalidQueryStringLengthError(f'Invalid query string length:{query_string_length} is less than 1.')
        if windowsize_local is None:
            raise InvalidQueryLocalWindowSizeError('The given window sizes are None.')
        if event_dimension<1:
            raise InvalidEventDimensionError('The event dimension has to be greater than zero.')

        query_string = ""
        alph_count=1
        for i in range(1,query_string_length+1):
            for _ in range(0,event_dimension):
                query_string = query_string + "$" + str(chr(alph_count+64)) + ";"
                alph_count=alph_count+1
            if i < query_string_length:
                query_string = query_string + " "

        self.set_query_string(query_string)
        assert self._query_string_length == query_string_length
        assert self._query_event_dimension == event_dimension
        self.set_query_windowsize_local(windowsize_local)
        if gap_constraints is not None:
            self.set_query_gap_constraints(gap_constraints)
        self.set_query_discovery_algorithm(discovery_algorithm)
        self.set_query_class(query_class)
        self.set_query_sample(sample)
        return self

    ##################################################

    def set_query_string(self, querystring:str, recalculate_attributes:bool=True, to_regex:bool=True) -> None:
        """
            Sets _query_string to querystring and does some updates.

            Updates _query_string_length and _query_repeated_variables as well
            to avoid an inconsistent query.

            Args:
                querystring: A string which represents the new query string. We
                    assume that querystring fits to the described format of a
                    query string.

                recalculate_attributes: Bool which indicates whether the other
                    query attributes should be updated. Sets them to default
                    values if recalculate_attributes is False.

                to_regex: Bool which indicates whether _query_string_regex
                    should be updated. Used True as default but is set to False
                    during a function call of query_string_to_normalform to
                    save runtime.
        """
        assert isinstance(querystring, str)
        self._query_string = querystring

        if recalculate_attributes is True:
            self.set_query_string_length()
            self.set_query_typeset() #sets query_event_dimension as well
            self.set_query_repeated_variables()
            if self._query_string_regex and to_regex is True:
                self.query_string_to_regex()
        else:
            self._query_string_length = 0
            self._query_event_dimension = 1
            self._query_typeset = set()
            self._query_attribute_typesets = dict()
            self._query_repeated_variables = set()
            self._query_string_regex = ""

    def set_query_string_length(self) -> None:
        """
            Determines and sets the length of the query string.

            Note that the query string length equals the number of events which
            occur within the string.
        """
        Query.set_query_string_length(self)

    def set_query_string_regex(self, regex_string:str) -> None:
        """
            Sets _query_string_regex.

            We assume regex_string to be a correct regex string of the current
            query string.

            Args:
                regex_string: String which represents the regex for the current
                    query string.
        """
        self._query_string_regex = regex_string

    def set_query_event_dimension(self) -> None:
        """
            Determines and set the maximum event dimension.

            Usually we assume that all events have the same dimension.
        """
        query_string_copy = copy(self._query_string)
        if query_string_copy == "":
            return
        dimension = -1
        while len(query_string_copy)>0:
            if query_string_copy[0] == " ":
                query_string_copy = query_string_copy[1:]
            else:
                event = query_string_copy.split(' ')[0]
                event_dimension = event.count(";")
                if event_dimension>dimension:
                    dimension = event_dimension
                query_string_copy = query_string_copy[len(event):]

        self._query_event_dimension = dimension

    def set_query_gap_constraints(self, gap_constraints:list) -> None:
        """
            Sets _query_gap_constraints to gap_constraints.

            The length of the given gap_constraints should equal
            (_query_string_length*_query_event_dimension)-placeholder_count-1.

            Note that we assume all events to have the same dimension.

            Args:
                gap_constraints: List of sets of strings describing which types
                    are not allowed between two consecutive events in the query
                    string.

            Raises:
                InvalidQueryGapConstraintError: Wrong count of gap constraints
                    for current query string length.
        """
        missing_attributes_count = self._get_missing_attributes_count()
        if len(gap_constraints) == (self._query_string_length*self._query_event_dimension)-missing_attributes_count-1:
            self._query_gap_constraints = gap_constraints
        else:
            raise InvalidQueryGapConstraintError('Wrong count of gap constraints for current query string length.')

    def set_query_windowsize_global(self, global_window_size:int) -> None:
        """
            Sets _query_windowsize_global to global_window_size.

            Args:
                global_window_size: Integer which desbribes the global window
                    size of the query, i.e. the range for a match in a trace.

            Raises:
                QueryInvalidGlobalWindowSizeError: If global_window_size is
                    incompatible with query string length.
        """
        Query.set_query_windowsize_global(self, global_window_size=global_window_size)

    def set_query_windowsize_local(self, local_window_size:list, during_nf_transformation:bool=False) -> None:
        """
            Sets _query_windowsize_local to local_window_size.

            The length of the given local_window_size should equal either
            (_query_string_length*_query_event_dimension)-placeholder_count+1
            or
            (_query_string_length*_query_event_dimension)-placeholder_count-1
            depending on whether the placeholder tuples (-1,-1) are already
            part of the given local_window_size.

            Exp.: The query string "a;;  b;c; ;;" is compatible with the local
            window sizes [(-1,-1),(1,3),(0,0),(2,6),(0,0),(-1,-1)].

            Note that we assume all events to have the same dimension.

            Args:
                local_window_size: List of integer tuples which describes lower
                    and upper bounds for the length of each gap between two
                    consecutive events in the query string.

            Raises:
                InvalidQueryLocalWindowSizeError: If local_window_size is
                    incompatible with query string length.
        """
        assert isinstance(local_window_size, list)
        if self._query_string_length == 0 and len(local_window_size) == 1:
            self._query_windowsize_local = local_window_size
            return
        missing_attributes_count = self._get_missing_attributes_count()
        if len(local_window_size) == (self._query_string_length*self._query_event_dimension)-missing_attributes_count+1 or during_nf_transformation is True:
            self._query_windowsize_local = local_window_size
            return
        #Add special window size tuples at the beginning and the end
        elif len(local_window_size) == (self._query_string_length*self._query_event_dimension)-missing_attributes_count-1:
            local_window_size.insert(0, (-1,-1))
            local_window_size.append((-1,-1))
            self._query_windowsize_local = local_window_size
        elif len(local_window_size)==0:
            self._query_windowsize_local = []
        else:
            raise InvalidQueryLocalWindowSizeError('Length of local window size list incompatible with query string length.')

    def set_query_class(self, queryclass:str) -> None:
        """
            Sets _query_class to queryclass if contained in QUERY_CLASS_LIST.

            Args:
                queryclass: String which describes the class of the query and
                    has to be chosen from the global variable QUERY_CLASS_LIST.

            Raises:
                QueryInvalidClassError: queryclass is not contained in
                    QUERY_CLASS_LIST.
        """
        Query.set_query_class(self, queryclass=queryclass)

    def set_query_repeated_variables(self) -> None:
        """
            Determines and sets the set of repeated variables.

            Only variable names are stored, hence entries of the set will not
            start with '$'. Since _query_repeated_variables is a set, the
            variables are not ordered. The set will be empty if the length of
            the query string is 0, i.e. no query string is set.
        """
        self._query_repeated_variables.clear()
        variable_dict = {}

        if len(self._query_string) == 0:
            self._query_repeated_variables = set()

        query_string_copy = copy(self._query_string)
        while len(query_string_copy) > 0:
            if query_string_copy[0] == "$":
                query_string_copy = query_string_copy[1:]
                variable = query_string_copy.split(';')[0]
                if variable in variable_dict:
                    variable_dict[variable] = variable_dict[variable]+1
                else:
                    variable_dict[variable] = 1
                query_string_copy = query_string_copy[len(variable):]
            query_string_copy = query_string_copy[1:]

        for elem in variable_dict:
            if variable_dict[elem] > 1:
                self._query_repeated_variables.add(elem)

    def set_query_is_in_normalform(self, boolean:bool) -> None:
        """
            Sets _query_is_in_normalform to boolean.
        """
        Query.set_query_is_in_normalform(self, boolean=boolean)

    def set_query_sample(self, sample:MultidimSample) -> None:
        """
            Sets _query_sample to Sample instance sample.

            Args:
                sample: Sample instance.
        """
        Query.set_query_sample(self, sample=sample)

    def set_query_sample_support(self, supp:float) -> None:
        """
            Sets _query_sample_support to support.

            Intended to save the calculated support for the current quey during
                match_sample(self, sample, supp).

            Args:
                supp: Float between 0 and 1 which represents the support of the
                    current query regarding the given sample.

            Raises:
                InvalidQuerySupportError: supp is less than 0 or greater than 1.
        """
        Query.set_query_sample_support(self, supp=supp)

    def set_query_typeset(self) -> None:
        """
            Sets _query_typeset and _query_attribute_typesets to set of types
            occuring in the _query_string.
        """
        self._query_typeset = set()
        self.set_query_event_dimension()
        self._query_attribute_typesets = dict()
        query_string_copy = copy(self._query_string)

        for i in range(1,self._query_event_dimension+1):
            self._query_attribute_typesets[i] = set()
        attribute_count = 0

        while len(query_string_copy) > 0:
            if query_string_copy[0] == "$":
                attribute_count = attribute_count + 1
                var_attribute = query_string_copy.split(';')[0]
                query_string_copy = query_string_copy[len(var_attribute):]
            elif query_string_copy[0] == " ":
                attribute_count = 0
                query_string_copy = query_string_copy[1:]
            elif query_string_copy[0] == ";":
                query_string_copy = query_string_copy[1:]
            else:
                attribute_count = attribute_count + 1
                type_attribute = query_string_copy.split(';')[0]
                query_string_copy = query_string_copy[len(type_attribute):]
                self._query_typeset.add(type_attribute)
                if attribute_count > 0 and attribute_count < self._query_event_dimension+1:
                    self._query_attribute_typesets[attribute_count].add(type_attribute)

    def set_query_discovery_algorithm(self, algorithm:str) -> None:
        """
            Sets the discovery algorithm for this query.

            Args:
                algorithm: String which has to be chosen from the global
                    variable DISCOVERY_ALGORITHM_LIST.

            Raises:
                InvalidQueryDiscoveryError: Given algorithm is not specified in
                    DISCOVERY_ALGORITHM_LIST.
        """
        Query.set_query_discovery_algorithm(self, algorithm=algorithm)

    def set_query_matchtest(self, algorithm:str) -> None:
        """
            Sets the matching algorithm for this query.

            Args:
                algorithm: String which has to be chosen from the global
                    variable MATCH_TEST_LIST.

            Raises:
                InvalidQueryMatchtestError: Given algorithm is not specified in
                    MATCH_TEST_LIST.
        """
        Query.set_query_matchtest(self, algorithm=algorithm)

    def set_pos_last_type_and_variable(self) -> None:
        """
            Returns last position of a type and first and last position of the
            last variable.

            Args:
                query: an instance of Query

            Returns: three args
                last type position (int), first position of last variable (int)
                and last position of last variable (int).
        """
        Query.set_pos_last_type_and_variable(self)

    ##################################################

    def _get_missing_attributes_count(self, query_string:str|None=None) -> int:
        """
            Return the number of missing attributes, i.e. placeholder
            semicolons, in the given string.

            Args:
                query_string: String which uses default None to determine the
                    number of placeholder semicolons in self._query_string. May
                    be used to compute the corresponding number in a given
                    prefix instead.

            Returns:
                Integer representing the number of placeholder semicolons.
        """
        missing_attributes_count = 0
        pos = 0
        if query_string is None:
            query_string = self._query_string
        while pos < len(query_string):
            if query_string[pos]==";":
                if pos==0 or query_string[pos-1]==";" or query_string[pos-1]==" ":
                    missing_attributes_count = missing_attributes_count+1
            pos = pos+1
        return missing_attributes_count
    
    def get_query_list(self) -> list:
        """
            Returns a list of the query string.

            Returns:
                List of strings.
        """
        if not self._query_string:
            return []
        if not self._query_list:
            self._query_list = self._query_string.split()
        return self._query_list

    # until here: in Class <MultidimQuery>
###############################################################################
    # outside of Class <MultidimQuery>

def _pos_last_type_and_variable(querystring:str) -> np.ndarray:
    """
        Returns last position of a type and first and last position of the last
        variable.

        Args:
            querystring: an instance of Query

        Returns: three args
            last type position (int), first position of last variable (int) and
            last position of last variable (int)
    """
    if not querystring:
        return np.array([-1,-1,-1])

    variables_set = set()
    query_split = querystring.split()
    for event in query_split:
        for symbol in event.split(';'):
            if symbol.count('$') !=0:
                variables_set.add(symbol[1:])
    variables=sorted(list(variables_set))
    
    querylength = len(query_split)
    if querystring.count(';') == 0:
        if querystring.count('$x') !=0:
            pos_last_var= querylength - 1 - query_split[::-1].index('$'+variables[-1])
            pos_first_var= query_split.index('$x'+str(variables[-1][1]))

            if len(query_split) == querystring.count('$x'):
                pos_last_type=-1
                return np.array([-1, pos_first_var, pos_last_var])
            else:
                position=-1
                while query_split[position].count('$x')!=0:
                    position-=1

                pos_last_type= querylength + position
                return np.array([pos_last_type, pos_first_var, pos_last_var])

        else:
            pos_first_var=-1
            pos_last_var =-1

            return np.array([querylength-1, pos_first_var, pos_last_var])

    if querystring.count('$x') !=0:
        last_var= variables[-1]
        string_pos = querystring.find(last_var)
        pos_first_var = querystring[:string_pos].count(' ')
        string_pos2= querystring.rfind(last_var)
        pos_last_var= querystring[:string_pos2].count(' ')

        if len(query_split) == querystring.count('$x'):
            pos_last_type=-1
            return np.array([-1, pos_first_var, pos_last_var])
        else:
            position=-1
            no_letter = True
            while no_letter and position >= -querylength:
                for event in query_split[position].split(';')[:-1]:
                    if event.count('$') == 0 and event:
                        no_letter = False
                        break
                if no_letter:
                    position-=1

            pos_last_type= querylength + position
            return np.array([pos_last_type, pos_first_var, pos_last_var])

    else:
        pos_first_var=-1
        pos_last_var =-1

        return np.array([querylength-1, pos_first_var, pos_last_var])
