#!/usr/bin/python3
""" Contains the class for handling Queries"""
import re
import logging
from copy import copy
from typing import Pattern, Match
from typing_extensions import Self
import numpy as np
from sample import Sample
from error import EmptySampleError,InconsistentQueryError,QueryRegexError,InvalidQueryStringLengthError,InvalidQuerySupportError,InvalidQueryGapConstraintError,InvalidQueryGlobalWindowSizeError,InvalidQueryLocalWindowSizeError,InvalidQueryClassError,InvalidQueryDiscoveryError,InvalidQueryMatchtestError

DISCOVERY_ALGORITHM_LIST = [
    'shinohara_icdt',
    'bottom_up',
    'top_down'
]

MATCH_TEST_LIST = [
    'regex',
    'smarter'
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

class Query:
    """
        Base class for representing and discovering a Query.

        Queries consist of a query string, global or local window size(s) and
        optional gap constraints.

        The query string is composed of so called types and variabels. Types
        are represented by strings over an alphabet (ascii, excluding
        whitespace and $).

        Gap constraints describe which types are forbidden between to positions
        of the query string.

        A query matches a trace t from a sample iff there exists a mapping from
        the query string to t, i.e. the query string is a subsequence of t, and
        neither the window size(s) nor the gap constraints are violated.

        Given a sample and a support this class offers the functionality to
        discover queries which fullfill the support.

        Attributes:
            _query_string: A query string consists of types and variables. They
                are represented as blocks within _query_string, separated by
                whitespaces. The beginning of a variable is marked by $, every
                unmarked block is a type. Each block consists of only one type
                or variable and models an 1-dimensional event.

            _query_string_length: Integer which represents the length of the
                query string without counting whitespaces and $.

            _query_gap_constraints: List of sets of strings, describing which
                types are not allowed between two consecutive events in the
                query string.

            _query_windowsize_global: Global window size of the query as
                integer, i.e. the range for a match in a trace. Uses -1 as
                default, which means no global window size is set.

            _query_windowsize_local: List of local window sizes. Each entry is
                a tuple of two integers, which describes lower and upper bounds
                for the length of each gap between two consecutive events in
                the query string. A query in normalfrom may have up to two
                additional window size tuples, up to one left of the first and
                up to one right of the last event. We use the tuple (-1,-1) as
                a placeholder for these two special window sizes.

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
                sample, s.t. a successful match test was performed. Note
                that this list may not contain all traces that match the query!

            _query_not_matched_traces: List of indices of all traces in
                _query_sample, i.e. the positions of the traces within the
                sample, s.t. an unsuccessful match test was performed. Note
                that this list may not contain all traces that do not match
                the query!

            _query_sample_support: Float between 0 and 1 which represents the
                support of the query regarding the given sample. Uses -1.0 as
                default if no support or sample is set.

            _query_typeset: Set of types occuring in the _query_string.

            _query_discovery_algorithm: A string which defines the requested
                discovery algorithm. Uses 'shinohara' as default. Has to be
                chosen from the global variable DISCOVERY_ALGORITHM_LIST.

            _query_matchtest: A string which defines the requested match test.
                Uses 'regex' as default and has to be chosen from the global
                variable MATCH_TEST_LIST.

            _pos_last_type_and_variable: A numpy array containing last type
                position first position of last variable, last position of last
                variable. Default value: np.array([-1,-1,-1])
    """
    def __init__(self, given_query_string=None, given_query_gap_constraints=None, given_query_windowsize_global=-1, given_query_windowsize_local=None, is_in_normalform=False) -> None:
        LOGGER.debug('Creating an instance of Query')
        self._query_repeated_variables:set = set()
        """Set of variables occuring more than once in _query_string. Default: set()"""
        self._query_typeset:set = set()
        """Set of types occuring in the _query_string. Default: set()"""

        if given_query_string is not None:
            assert isinstance(given_query_string, str)
            self._query_string:str = given_query_string
            """String consisting of types and variables, separated by whitespaces. Default: ''"""
            self.set_query_string_length()
            self.set_query_repeated_variables()
            self.set_query_typeset()
            if given_query_gap_constraints is not None:
                self._query_gap_constraints:list = given_query_gap_constraints
                """List of sets of string containing forbidden types between events in _query_string. Default: []"""
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
                """List of tuples defining lower and upper bounds for the gap length between events in _query_string. Default: []"""
        else:
            self._query_string = ""
            self._query_string_length:int = 0
            """Number of events within _query_string, i.e. number of types and variables. Default: 0"""
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
        self._query_sample:Sample|None = None
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
        """
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
        return ""

    ##################################################

    def match_sample(self, sample, supp, complete_test=False, dict_iter = None,
                     patternset = None, parent_dict = None, max_query_length= -1):
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
                
                dict_iter= dict_iter (dictionary): nested dictionary for each 
                    query and trace the last matching position is value. 
                    Default None
                
                patternset: set of types occurring twice in at least one trace. 
                    Default None
                
                parent_dict: Dictionary containing parent query to each querystring. 
                    Default None

            Returns:
                True iff the query matches the given sample with given supp.

            Raises:
                EmptySampleError: The given sample is empty.
                InvalidQuerySupportError: Supp is <0 or >1.
                InvalidQueryMatchtestError: The match test is not defined.
        """
        LOGGER.debug('match_sample - Started')
        assert isinstance(sample, Sample)
        if supp != 1 and supp !=0:
            assert isinstance(supp, float)

        if supp < 0 or supp > 1:
            raise InvalidQuerySupportError(f'Support {supp} has to be between 0 and 1.')

        sample.set_sample_size()
        trace_count = sample._sample_size
        match_count = 0
        test_count = 0

        if trace_count == 0:
            raise EmptySampleError('The given sample is empty.')

        
        if self._query_matchtest == 'regex':
            # Preprocessing:
            if self._query_is_in_normalform is False and len(self._query_windowsize_local) > 0 and len(self._query_gap_constraints)==0 and self._query_windowsize_global == -1:
                LOGGER.debug('match_sample - preprocessing - regex - Use function query_string_to_normalform')
                self.query_string_to_normalform()
            if not self._query_string_regex:
                LOGGER.debug('match_sample - preprocessing - regex - Use function query_string_to_regex')
                self.query_string_to_regex()
            LOGGER.debug('match_sample - preprocessing - regex - Compile _query_string_regex')
            regex = re.compile(self._query_string_regex)

            # Matching:
            for trace in sample._sample:
                LOGGER.debug('match_sample - Current trace: %s', trace)
                if self._query_matchtest == 'regex':
                    LOGGER.debug('match_sample - Use function match_trace_regex')
                    witness = self.match_trace_regex(trace, regex)
                    test_count = test_count+1
                if witness is not None:
                    LOGGER.debug('match_sample - Match of trace %s was successfull', trace)
                    match_count = match_count+1
                    #Index of current trace is stored in _query_matched_traces
                    self._query_matched_traces.append(sample._sample.index(trace))
                else:
                    #Index of current trace is stored in _query_not_matched_traces
                    LOGGER.debug('match_sample - Match of trace %s was not successfull', trace)
                    self._query_not_matched_traces.append(sample._sample.index(trace))
                #Check whether we have to continue or not
                if complete_test is False and ((match_count+(trace_count-test_count) / trace_count < supp)
                    or (match_count/trace_count >= supp)):
                    break
            sample_support = match_count/trace_count
            # The query sample support is set if a complete test was executed
            if complete_test is True or trace_count==test_count:
                self.set_query_sample_support(sample_support)
            LOGGER.debug('match_sample - Requested sample support: %s', str(supp))
            LOGGER.debug('match_sample - Sample support of query:  %s', str(sample_support))
            if sample_support >= supp:
                LOGGER.debug('match_sample - Finished with True')
                return True
            LOGGER.debug('match_sample - Finished with False')
            return False
        
        elif self._query_matchtest == 'smarter':
            querystring = self._query_string
            if max_query_length != -1 and self._query_string_length > max_query_length:
                return False
            matching = self.matching_smarter(sample=sample, supp = supp, dict_iter = dict_iter, patternset = patternset, parent_dict = parent_dict)
            if querystring.count('$')!=0:
                matchingcount= len(matching)
                self._query_matched_traces = list(matching.keys())
            else:
                matchingcount=0
                matched_traces = []
                for key,value in matching.items():
                    if value != -1:
                        matchingcount +=1
                        matched_traces.append(key)
                self._query_matched_traces = matched_traces

            matchsupport= matchingcount/sample._sample_size
            if matchsupport >= supp:
                for trace_index, value in matching.items():
                    if querystring not in dict_iter:
                        dict_iter[querystring]= {}
                    dict_iter[querystring][trace_index]= value
                return True
            return False
        else:
            raise InvalidQueryMatchtestError(f'The match test {self._query_matchtest} is not defined.')

    def match_trace_regex(self, trace:str, regex:Pattern) -> Match[str] | None:
        """
            Checks whether the query matches a given trace.

            Args:
                trace: String which represents a trace as described in Sample.

                regex: Regex Object of _query_string_regex

            Returns:
                A MatchObject as witness iff the query matches the given trace
                and None otherwise.
        """
        LOGGER.debug('match_trace_regex - Started')
        assert isinstance(self, Query)
        assert isinstance(trace, str)

        witness = regex.search(trace)
        LOGGER.debug('match_trace_regex - Finished')
        return witness

    ##################################################

    def matching_smarter(self, sample, supp, dict_iter, patternset,parent_dict):
        """Matches a query against all traces in the sample

        Args:
            supp:Float between 0 and 1 which describes the requested support. Default: 1
            dict_iter= dict_iter (dictionary): nested dictionary for each query and trace 
                the last matching position is value. Default None
            patternset: set of types occurring twice in at least one trace. Default None
            parent_dict: Dictionary containing parent query to each querystring. Default None

        Returns:
            Trace dictionary containing trace index as keys and a dictionary of groups with 
            the matched string and span as values.
        """
        querystring = self._query_string
        if querystring in parent_dict:
            parent = parent_dict[querystring]
        else:
            parent= self._parent()
            parent_dict[querystring] = parent
        parentstring = parent._query_string
        trace_matches={}
        sample_size = len(sample._sample)

        if querystring.count('$x') == 0:
            num_trace_match = sample_size
            for trace_idx, trace in enumerate(sample._sample):
                if num_trace_match/sample_size < supp:
                    break
                idx = self._smart_trace_match(querystring, trace, trace_idx, dict_iter)
                if trace_idx not in trace_matches:
                    trace_matches[trace_idx]= {}
                trace_matches[trace_idx]= idx
                if idx == -1:
                    num_trace_match -=1

            return trace_matches

        if parentstring.count('$')==0:
            if not parentstring:
                parent_traces = list(range(sample_size))
            else:
                parent_traces= list(dict_iter[parentstring].keys())
            trace_list= parent_traces + list(range(parent_traces[-1]+1,sample_size))
            num_trace_match = len(trace_list)
            for trace in trace_list:
                if num_trace_match/sample_size < supp:
                    break
                for letter in patternset:
                    letter_querystring = querystring.replace('$x0', letter)
                    if letter_querystring in dict_iter:
                        if trace in dict_iter[letter_querystring]:
                            if dict_iter[letter_querystring][trace] !=-1:
                                if trace not in trace_matches:
                                    trace_matches[trace]= {}
                                trace_matches[trace][(letter,)]= dict_iter[letter_querystring][trace]
                        else:
                            idx = self._smart_trace_match(letter_querystring, sample._sample[trace], trace, dict_iter)
                            if idx != -1:
                                if trace not in trace_matches:
                                    trace_matches[trace]= {}
                                trace_matches[trace][(letter,)]= idx
                    else:
                        idx = self._smart_trace_match(letter_querystring, sample._sample[trace], trace, dict_iter)
                        if idx != -1:
                            if trace not in trace_matches:
                                trace_matches[trace]= {}
                            trace_matches[trace][(letter,)]= idx

                if trace not in trace_matches:
                    num_trace_match-=1
            return trace_matches

        else:
            if not parent._query_repeated_variables:
                parent.set_query_repeated_variables()
            parent_variables=sorted(list(parent._query_repeated_variables))
            if parentstring in dict_iter:
                parent_traces= list(dict_iter[parentstring].keys())
            else:
                parent_match = parent.matching_smarter(sample = sample, dict_iter = dict_iter, patternset = patternset, supp = supp, parent_dict = parent_dict)
                dict_iter[parentstring] = parent_match
                parent_traces= list(parent_match.keys())
            trace_list= parent_traces
            num_trace_match = len(trace_list)
            for trace in trace_list:
                if num_trace_match/sample_size < supp:
                    break
                group_list= list(dict_iter[parentstring][trace].keys())
                for group in group_list:
                    letter_querystring=querystring
                    assert len(group) == len(parent_variables)
                    for val, letter in enumerate(group):
                        letter_querystring = letter_querystring.replace(f'${parent_variables[val]}', letter)
                    if letter_querystring.count('$')>0:
                        for letter in patternset:
                            letter_querystring2 = letter_querystring.replace(f'$x{len(group)}', letter)
                            if letter_querystring2 in dict_iter:
                                if trace in dict_iter[letter_querystring2]:
                                    if dict_iter[letter_querystring2][trace] !=-1:
                                        if trace not in trace_matches:
                                            trace_matches[trace]= {}
                                        trace_matches[trace][group + (letter,)]= dict_iter[letter_querystring2][trace]

                                else:
                                    idx = self._smart_trace_match(letter_querystring2, sample._sample[trace], trace, dict_iter)
                                    if idx != -1:
                                        if trace not in trace_matches:
                                            trace_matches[trace]= {}
                                        trace_matches[trace][group + (letter,)]= idx

                            else:
                                idx = self._smart_trace_match(letter_querystring2, sample._sample[trace], trace, dict_iter)
                                if idx != -1:
                                    if trace not in trace_matches:
                                        trace_matches[trace]= {}
                                    trace_matches[trace][group + (letter,)]= idx

                    else:
                        if letter_querystring in dict_iter:
                            if trace in dict_iter[letter_querystring]:
                                if dict_iter[letter_querystring][trace] !=-1:
                                    if trace not in trace_matches:
                                        trace_matches[trace]= {}
                                    trace_matches[trace][group]= dict_iter[letter_querystring][trace]
                            else:
                                idx = self._smart_trace_match(letter_querystring, sample._sample[trace], trace, dict_iter)
                                if idx != -1:
                                    if trace not in trace_matches:
                                        trace_matches[trace]= {}
                                    trace_matches[trace][group]= idx
                        else:
                            idx= self._smart_trace_match(letter_querystring, sample._sample[trace], trace, dict_iter)
                            if idx != -1:
                                if trace not in trace_matches:
                                    trace_matches[trace]= {}
                                trace_matches[trace][group]= idx

                if trace not in trace_matches:
                    num_trace_match-=1
            return trace_matches

    def _smart_trace_match(self, querystring, trace, trace_idx, dict_iter):
        """Given a trace and a querystring the matching position is calculated and in case of a match
        the dict_iter is updated.

        Args:
            querystring (String): querystring for query
            trace (String): trace from the sample
            trace_idx (int): index of the given trace
            dict_iter (dictionary): nested dictionary for each query and trace the last matching position is value.

        Returns:
            [dictionary]: updated dict_iter
        """
        parentstring = ' '.join(querystring.split()[:-1])
        if querystring not in dict_iter:
            dict_iter[querystring]= {}
        if not querystring:
            dict_iter[querystring][trace_idx]=0
            return 0
        if parentstring not in dict_iter:
            idx= self._smart_trace_match(parentstring, trace, trace_idx, dict_iter)
        if trace_idx in dict_iter[parentstring]:
            parent_end_pos = dict_iter[parentstring][trace_idx]
        else:
            idx= self._smart_trace_match(parentstring, trace, trace_idx, dict_iter)
            parent_end_pos = dict_iter[parentstring][trace_idx]
        if not parentstring:
            try:
                end_pos = trace.split().index(querystring)
            except ValueError:
                end_pos = -1
            dict_iter[querystring][trace_idx]= end_pos
            return end_pos
        if parent_end_pos !=-1:
            trace_list = trace.split()[parent_end_pos+1:]
            try:
                idx = trace_list.index(querystring.split()[-1])
            except ValueError:
                idx = -1
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

    def query_string_to_normalform(self, check_consistency:bool=False) -> None:
        """
            Converts the _query_string into normalform.

            A query string with neither gap constraints nor a global window
            size is in normalform iff it does not contain non-repreated
            variables. This function deletes all non-repeated variables and
            merges the local window sizes.

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

        result = string_to_normalform(self._query_string, self._query_repeated_variables, self._query_windowsize_local)
        query_string_nf, updated_windowsize_local = result

        self.set_query_windowsize_local(updated_windowsize_local, True)
        self.set_query_string(query_string_nf)
        self.set_query_string_length()
        self.set_query_is_in_normalform(True)

        if check_consistency is True:
            self.check_consistency()

        LOGGER.debug('query_string_to_normalform - Finished')

    def query_string_to_regex(self) -> None:
        """
            Sets a regular expression for the current _query_string.

            Encoding
                type: type
                first occurence of a variable with name var: (?P<var>\\S+)
                further occurences of var: (?P=var\\S+)
                whitespace: \\s
                gap constraint gap and local window size (lower, upper):
                (?:[^\\s gap]+\\s){lower,upper}

            Raises:
                InconsistentQueryError: If the number of gap constraints or
                    local window sizes do not match the query string length.

                QueryRegexError: Gap constraint or local window size handling
                    went wrong.
        """
        LOGGER.debug('query_string_to_regex - Started')
        var_dict_regex = {}
        var_dict_regex_backref = {}
        var_dict_count = {}
        for var in self._query_repeated_variables:
            var_dict_regex[var] = "(?P<" + str(var) + ">\\S+)"
            var_dict_regex_backref[var] = "(?P=" + str(var) + ")"
            var_dict_count[var] = 0

        regex_string = ""
        query_string = copy(self._query_string)
        pos = 0

        # Special case: empty query string
        if self._query_string_length == 0:
            LOGGER.debug('query_string_to_regex - Special case: Empty query string')
            if len(self._query_windowsize_local)>0:
                assert len(self._query_windowsize_local)==1
                if list(self._query_windowsize_local[0])[0] > -1 and list(self._query_windowsize_local[0])[1] > -1:
                    LOGGER.debug('query_string_to_regex - Special case: Empty query string: Add local window size tuple to regex')
                    lower_bound = list(self._query_windowsize_local[0])[0]-1
                    upper_bound = list(self._query_windowsize_local[0])[1]-1
                    assert lower_bound<=upper_bound
                    regex_string = regex_string + "(?:[^\\s]+\\s){" + str(lower_bound) + "," + str(upper_bound) + "}" + "[^\\s]+"#"(?:[^\\s]+){0,1}"
                LOGGER.debug('query_string_to_regex - Special case: Empty query string: Final regex: %s', regex_string)
        # Special case: query string contains only one type
        elif self._query_string_length == 1:
            LOGGER.debug('query_string_to_regex - Special case: Query string contains only one type')
            if len(self._query_windowsize_local)>0:
                assert len(self._query_windowsize_local)==2
                if list(self._query_windowsize_local[0])[0] > -1 and list(self._query_windowsize_local[0])[1] > -1:
                    LOGGER.debug('query_string_to_regex - Special case: Query string contains only one type: Add first local window size tuple to regex')
                    lower_bound = list(self._query_windowsize_local[0])[0]
                    upper_bound = list(self._query_windowsize_local[0])[1]
                    regex_string = regex_string + "(?:[^\\s]+\\s){" + str(lower_bound) + "," + str(upper_bound) + "}"
                type_event = query_string[pos:]
                type_event = type_event.split(' ')[0]
                regex_string = regex_string + str(type_event)
                if list(self._query_windowsize_local[1])[0] > -1 and list(self._query_windowsize_local[1])[1] > -1:
                    LOGGER.debug('query_string_to_regex - Special case: Query string contains only one type: Add last local window size tuple to regex')
                    lower_bound = list(self._query_windowsize_local[1])[0]-1
                    upper_bound = list(self._query_windowsize_local[1])[1]-1
                    assert lower_bound<=upper_bound
                    regex_string = regex_string + "\\s(?:[^\\s]+\\s){" + str(lower_bound) + "," + str(upper_bound) + "}" + "[^\\s]+"

                LOGGER.debug('query_string_to_regex - Special case: Query string contains only one type: Final regex: %s', regex_string)
            else:
                regex_string = query_string + "(\\s.)*?"

        else:
            while True:
                if pos > len(query_string)-1:
                    if len(self._query_windowsize_local) > 0:
                        LOGGER.debug('query_string_to_regex - Reached the end of query string')
                        pos = len(query_string)-1
                        prefix = query_string[0:pos]
                        index = prefix.count(" ")+1
                        lower_bound = list(self._query_windowsize_local[index])[0]-1
                        upper_bound = list(self._query_windowsize_local[index])[1]-1
                        if lower_bound > -1 and upper_bound > -1:
                            assert lower_bound<=upper_bound
                            LOGGER.debug('query_string_to_regex - Reached the end of query string: Add last local window size tuple to regex')
                            regex_string = regex_string + "\\s(?:[^\\s]+\\s){" + str(lower_bound) + "," + str(upper_bound) + "}" + "[^\\s]+"#"(?:[^\\s]+){0,1}"
                        LOGGER.debug('query_string_to_regex - Reached the end of query string: Final regex: %s', regex_string)
                    break
                # Start regex_string with the first local window size tuple if local window sizes are given and their value is greater than -1
                if pos == 0 and len(self._query_windowsize_local) > 0:
                    LOGGER.debug('query_string_to_regex - Begin of query string')
                    lower_bound = list(self._query_windowsize_local[0])[0]
                    upper_bound = list(self._query_windowsize_local[0])[1]
                    if lower_bound > -1 and upper_bound > -1:
                        LOGGER.debug('query_string_to_regex - Begin of query string: Start regex with first local window size tuple')
                        regex_string = regex_string + "(?:[^\\s]+\\s){" + str(lower_bound) + "," + str(upper_bound) + "}"
                    LOGGER.debug('query_string_to_regex - Begin of query string: Current regex: %s', regex_string)
                if query_string[pos] == " ":
                    LOGGER.debug('query_string_to_regex - Current position %s in %s is gap', str(pos), query_string)
                    prefix = query_string[0:pos-1]
                    index = prefix.count(" ")
                    gap_constraint_bool = False
                    window_size_bool = False
                    # Include gap constraints and local window sizes in regex_string if they exist
                    gap_constraint_string = ""
                    lower_bound = -1
                    upper_bound = -1
                    if len(self._query_gap_constraints) > 0:
                        LOGGER.debug('query_string_to_regex - Current position is gap: Build gap constraint regex string')
                        if len(self._query_gap_constraints) != self._query_string_length-1:
                            raise InconsistentQueryError('Number of gap constraints do not match the query string length.')
                        gap_constraint = self._query_gap_constraints[index]
                        for gap in gap_constraint:
                            gap_constraint_string = gap_constraint_string + str(gap)
                        gap_constraint_bool = True
                    if len(self._query_windowsize_local) > 0:
                        LOGGER.debug('query_string_to_regex - Current position is gap: Build local window size tuple regex string')
                        if len(self._query_windowsize_local) > self._query_string_length+1:
                            raise InconsistentQueryError('Number of local window sizes do not match the query string length.')
                        index = index+1
                        lower_bound = list(self._query_windowsize_local[index])[0]
                        upper_bound = list(self._query_windowsize_local[index])[1]
                        window_size_bool = True
                    if gap_constraint_bool is True and window_size_bool is True:
                        LOGGER.debug('query_string_to_regex - Current position is gap: Add gap constraint and local window size tuple')
                        regex_string = regex_string + "\\s(?:[^\\s" + gap_constraint_string + "]+\\s){" + str(lower_bound) + "," + str(upper_bound) + "}"
                        pos = pos+1
                    elif gap_constraint_bool is True and window_size_bool is not True:
                        LOGGER.debug('query_string_to_regex - Current position is gap: Add gap constraint')
                        # Since no local window sizes are given we add an arbitrary long gap.
                        # By using the question mark we force the match to be the shortest possible match
                        regex_string = regex_string + "\\s(?:[^\\s" + gap_constraint_string + "]+\\s)*?"
                        pos = pos+1
                    elif gap_constraint_bool is not True and window_size_bool is True:
                        LOGGER.debug('query_string_to_regex - Current position is gap: Add local window size tuple')
                        regex_string = regex_string + "\\s(?:[^\\s]+\\s){" + str(lower_bound) + "," + str(upper_bound) + "}"
                        pos = pos+1
                    elif gap_constraint_bool is not True and window_size_bool is not True:
                        LOGGER.debug('query_string_to_regex - Current position is gap: Add neither gap constraint nor local window size tuple')
                        # Since no local window sizes are given we add an arbitrary long gap.
                        # By using the question mark we force the match to be the shortest possible match
                        regex_string = regex_string + "\\s(?:[^\\s]+\\s)*?"
                        pos = pos+1
                    else:
                        raise QueryRegexError('Gap constraint or local window size handling went wrong.')
                    LOGGER.debug('query_string_to_regex - Current position is gap: Current regex after gap handling: %s', regex_string)
                # Handling of a variable
                elif query_string[pos] == "$":
                    variable = query_string[pos+1:]
                    variable = variable.split(' ')[0]
                    LOGGER.debug('query_string_to_regex - Current position %s in %s is variable: %s', str(pos), query_string, variable)
                    if variable not in self._query_repeated_variables:
                        LOGGER.debug('query_string_to_regex - Current position is variable: non-repeated')
                        regex_string = regex_string + "(?P<" + str(variable) + ">\\S+)"
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
                    regex_string = regex_string + str(self._query_string[pos])
                    pos = pos+1
                    LOGGER.debug('query_string_to_regex - Current position is part of a type: Current regex after type handling: %s', regex_string)
        self.set_query_string_regex(regex_string)
        LOGGER.debug('query_string_to_regex - Query string %s has final regex %s', self._query_string, regex_string)
        LOGGER.debug('query_string_to_regex - Finished')

    def _parent(self):
        """returns the parent query of a given query, in other words the query from which it was generated acording to the constrain-based rule set:
            query: an instance of Query

        Returns:
            parent: an instance of Query
        """
        querystring= self._query_string
        if not querystring:
            return self
        last_position= self._last_inserted_element()[1]
        last_type = querystring.split()[last_position]
        parentstring=" ".join(querystring.split()[:last_position]) + " " + " ".join(querystring.split()[last_position+1:])
        parent=Query()
        
        parentstring=parentstring.strip()
        parent.set_query_string(parentstring, recalculate_attributes=False)
        parent.query_string_to_normalform()
        return parent

    def _last_inserted_element(self):
        """Returns the element that was last inserted according to the rule set and its position and if its type or variable.
        Args:
            query: an instance of Query

        Returns:
            type or variable (string)
            position in querystring (int)
            true if type, false if variable, None for empty string
        """
        querystring= self._query_string

        if not querystring:
            return "", -1, None

        if self._pos_last_type_and_variable.sum() == -3:
            self.set_pos_last_type_and_variable()
        pos_last_type_and_variable= self._pos_last_type_and_variable
        pos_last_type= pos_last_type_and_variable[0]
        pos_first_var= pos_last_type_and_variable[1]
        pos_last_var = pos_last_type_and_variable[2]

        if pos_last_type > pos_first_var:
            return querystring.split()[pos_last_type], pos_last_type, True

        if pos_last_type < pos_first_var:
            return querystring.split()[pos_last_var], pos_last_var, False
    
    def query_pos_dict(self, event_db:dict, sample:Sample, event_dictionary:dict|None=None, trace_list:list = [0]) -> dict:
        """
            Calculates for each given query the instances that each query occures
            in for each trace.

            Args:
                event_db (dictionary): result from function
                    discovery.create_vertical_sequence_database()

                sample (Sample): an instance of Sample

                event_dictionary (dict): {querystring: trace_id : [list of trace positions]}. Defaults to: None.

                trace_list (List): List of trace id's which should be calculated. Defaults to: [0].

            Returns:
                event_dictionary: {querystring: trace_id : [list of trace positions]}
        """
        sample_set = sample._sample
        if not event_dictionary:
            event_dictionary = {}
        querystring = self._query_string
        if querystring:
            event_dictionary[querystring]= {}
            for trace_idx in trace_list:
                cur_trace = sample_set[trace_idx]
                cur_trace_split = cur_trace.split()
                querystring_split = querystring.split()
                for pos, symbol in enumerate(querystring_split):
                    symbol_count = querystring.count(symbol)
                    instances=[]
                    if symbol in event_db:
                        if trace_idx not in event_db[symbol]:
                            break
                        pos_list=event_db[symbol][trace_idx]

                    else:
                        pos_list= []
                        for letter in event_db.keys():
                            if trace_idx in event_db[letter].keys():
                                if len(event_db[letter][trace_idx]) >= symbol_count:
                                    pos_list.extend(event_db[letter][trace_idx])
                    if pos == 0:
                        event_dictionary[querystring][trace_idx]= [[position] for position in pos_list]
                    else:
                        for instance in event_dictionary[querystring][trace_idx]:
                            last_position= instance[-1]
                            if symbol in event_db:
                                new_positions = pos_list

                            else:
                                if " ".join(querystring_split[:pos]).count(symbol) == 0:
                                    new_positions= sorted(pos_list)
                                    
                                else:
                                    var_start= querystring.split().index(symbol)
                                    letter= cur_trace_split[instance[var_start]]
                                    new_positions = event_db[letter][trace_idx]
                            for new_pos in new_positions[::-1]:
                                if new_pos > last_position:
                                    instances.append(instance+ [new_pos])
                                else:
                                    break
                        if instances:
                            event_dictionary[querystring][trace_idx] = instances
                        else:
                            event_dictionary[querystring].pop(trace_idx, None)
                            break


        return event_dictionary
    
    ##################################################

    def check_consistency(self) -> None:
        """
            Checks whether attributes of the current query are consistent.

            Inconsistent states:
                - query string and query string length are incompatible
                - local window sizes are given
                    - but are incompatible with the query string length
                    - and a tuple has lower bound which is greater than zero
                    but the query string is not in normalform
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

                InconsistentQueryError: If the query string is in normalform but
                    a global window size or gap constraints are given.

                InvalidQueryGapConstraintError: If the number of gap constraints is
                    incompatible with the query string length.
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
            elif self._query_string != "" and len(self._query_windowsize_local)!=self._query_string_length+1:
                raise InvalidQueryLocalWindowSizeError('CHECK_CONSISTENCY: Length of local window size list incompatible with query string length.')

            for window_size_tuple in self._query_windowsize_local:
                if list(window_size_tuple)[0] > 0 and self._query_is_in_normalform is False:
                    raise InconsistentQueryError('CHECK_CONSISTENCY: Lower bound for local window size tuple is greater than zero but query string is not in normalform.')

        if self._query_windowsize_global > -1:
            if self._query_is_in_normalform is True:
                raise InconsistentQueryError('CHECK_CONSISTENCY: A query with a global window size can not be in normalform.')

        if len(self._query_gap_constraints)>0:
            if len(self._query_gap_constraints) != self._query_string_length-1:
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

    def init_most_general_query_for_shinohara(self, query_string_length:int, windowsize_local:list, discovery_algorithm:str, sample:Sample, gap_constraints:list|None=None, query_class:str="normal") -> Self:
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

        query_string = "$A"
        for i in range(2,query_string_length+1):
            query_string = query_string + " $" + str(chr(i+64))
        self.set_query_string(query_string)
        self.set_query_string_length()
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
        self._pos_last_type_and_variable = np.array([-1,-1,-1])
        if recalculate_attributes is True:
            self.set_query_string_length()
            self.set_query_typeset()
            self.set_query_repeated_variables()
            if self._query_string_regex and to_regex is True:
                self.query_string_to_regex()
        else:
            self._query_string_length = 0
            self._query_typeset = set()
            self._query_repeated_variables = set()
            self._query_string_regex = ""
        

    def set_query_string_length(self) -> None:
        """
            Determines and sets the length of the query string without counting
            whitespaces and $.
        """
        if self._query_string == "":
            self._query_string_length = 0
        else:
            self._query_string_length = self._query_string.count(" ")+1

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

    def set_query_gap_constraints(self, gap_constraints:list) -> None:
        """
            Sets _query_gap_constraints to gap_constraints.

            Args:
                gap_constraints: List of sets of strings describing which types
                    are not allowed between two consecutive events in the query
                    string.

            Raises:
                InvalidQueryGapConstraintError: Wrong count of gap constraints
                    for current query string length.
        """
        if len(gap_constraints) == self._query_string_length-1:
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
        assert isinstance(global_window_size, int)
        if global_window_size >= self._query_string_length:
            self._query_windowsize_global = global_window_size
        else:
            raise InvalidQueryGlobalWindowSizeError('Global window size incompatible with query string length.')

    def set_query_windowsize_local(self, local_window_size:list, during_nf_transformation:bool=False) -> None:
        """
            Sets _query_windowsize_local to local_window_size.

            Args:
                local_window_size: List of integer tuples which describes lower
                    and upper bounds for the length of each gap between two
                    consecutive events in the query string.

            Raises:
                InvalidQueryLocalWindowSizeError: If local_window_size is
                    incompatible with query string length.
        """
        assert isinstance(local_window_size, list)
        if len(local_window_size) == self._query_string_length+1 or during_nf_transformation is True:
            self._query_windowsize_local = local_window_size
            return
        #Add special window size tuples at the beginning and the end
        elif len(local_window_size) == self._query_string_length-1:
            local_window_size.insert(0, (-1,-1))
            local_window_size.append((-1,-1))
            self._query_windowsize_local = local_window_size
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
        if queryclass in QUERY_CLASS_LIST:
            self._query_class = queryclass
        elif queryclass is None:
            return
        else:
            raise InvalidQueryClassError(f'Invalid query class {queryclass} specified.')

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
                variable = query_string_copy.split(' ')[0]
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
        assert isinstance(boolean, bool)
        self._query_is_in_normalform = boolean

    def set_query_sample(self, sample:Sample) -> None:
        """
            Sets _query_sample to Sample instance sample.

            Args:
                sample: Sample instance.
        """
        assert isinstance(sample, Sample)
        self._query_sample = sample
        self.set_query_typeset()

    def set_query_sample_support(self, supp:float) -> None:
        """
            Sets _query_sample_support to support.

            Intended to save the calculated support for the current quey during
            match_sample(self, sample, supp).

            Args:
                supp: Float between 0 and 1 which represents the support of the
                current query regarding the given sample.

            Raises:
                InvalidQuerySupportError: supp is <0 or >1.
        """
        assert isinstance(supp, float)
        if supp >= 0.0 and supp <= 1.0:
            self._query_sample_support = supp
        else:
            raise InvalidQuerySupportError(f'Support {supp} has to be between 0 and 1.')

    def set_query_typeset(self) -> None:
        """
            Sets _query_typeset to set of types occuring in the _query_string.
        """
        self._query_typeset = set()
        query_string_copy = copy(self._query_string)
        while len(query_string_copy) > 0:
            if query_string_copy[0] == "$":
                variable = query_string_copy.split(' ')[0]
                query_string_copy = query_string_copy[len(variable):]
            elif query_string_copy[0] == " ":
                query_string_copy = query_string_copy[1:]
            else:
                type_event = query_string_copy.split(' ')[0]
                query_string_copy = query_string_copy[len(type_event):]
                self._query_typeset.add(type_event)

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
        if algorithm in DISCOVERY_ALGORITHM_LIST:
            self._query_discovery_algorithm = algorithm
        else:
            raise InvalidQueryDiscoveryError(f'Invalid discovery algorithm {algorithm} specified.')

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
        if algorithm in MATCH_TEST_LIST:
            self._query_matchtest = algorithm
        else:
            raise InvalidQueryMatchtestError(f'Invalid matching algorithm {algorithm} specified.')

    def set_pos_last_type_and_variable(self) -> None:
        """
            Sets last position of a type and first and last position of the
            last variable.

            Args:
                query: an instance of Query
        """
        querystring = self._query_string
        if not querystring:
            self._pos_last_type_and_variable = np.array([-1,-1,-1])
            return

        variables_set = set()
        querystring_split = querystring.split()
        for event in querystring_split:
            for symbol in event.split(';'):
                if symbol.count('$') !=0:
                    variables_set.add(symbol[1:])
        variables=sorted(list(variables_set))

        querylength = len(querystring_split)
        if querystring.count(';') == 0:
            if len(self._query_repeated_variables) !=0:
                pos_last_var= querylength - 1 - querystring_split[::-1].index('$'+variables[-1])
                pos_first_var= querystring_split.index('$x'+str(variables[-1][1]))

                if len(querystring_split) == querystring.count('$x'):
                    pos_last_type=-1
                    self._pos_last_type_and_variable = np.array([-1, pos_first_var, pos_last_var])
                else:
                    position=-1
                    while querystring_split[position].count('$x')!=0:
                        position-=1

                    pos_last_type= querylength + position
                    self._pos_last_type_and_variable = np.array([pos_last_type, pos_first_var, pos_last_var])

            else:
                pos_first_var=-1
                pos_last_var =-1

                self._pos_last_type_and_variable = np.array([querylength-1, pos_first_var, pos_last_var])
        else:
            xcount = querystring.count('$x')
            if xcount !=0:
                last_var= variables[-1]
                string_pos = querystring.find(last_var)
                pos_first_var = querystring[:string_pos].count(' ')
                string_pos2= querystring.rfind(last_var)
                pos_last_var= querystring[:string_pos2].count(' ')

                if len(querystring_split) == xcount:
                    pos_last_type=-1
                    self._pos_last_type_and_variable = np.array([-1, pos_first_var, pos_last_var])
                    return
                else:
                    position=-1
                    no_letter = True
                    while no_letter and position >= -querylength:
                        for event in querystring_split[position].split(';')[:-1]:
                            if event.count('$') == 0 and event:
                                no_letter = False
                                break
                        if no_letter:
                            position-=1

                    pos_last_type= querylength + position
                    self._pos_last_type_and_variable = np.array([pos_last_type, pos_first_var, pos_last_var])
                    return

            else:
                pos_first_var=-1
                pos_last_var =-1

                self._pos_last_type_and_variable = np.array([querylength-1, pos_first_var, pos_last_var])
                return

    # until here: in Class <Query>
###############################################################################
    # outside of Class <Query>

def string_to_normalform(query_string:str, repeated_variables:set|None=None, query_windowsize_local:list|None=None) -> tuple:
    """
        Converts the query_string into normalform.

        A query string with neither gap constraints nor a global window size is
        in normalform iff it does not contain non-repreated variables. This
        function deletes all non-repeated variables and merges the local window
        sizes.

        Args:
            query_string: a query_string.

            repeated_variables [= None]: optional parameter, if the
                repeated_variables are already known

            query_windowsize_local [= None]: optional parameter, if the
                query_string contains local windowsizes. They will be adjusted
                on normalization process.

        Returns:
            tuple of query_string, query_windowsize_local.
                They contain the updated versions.

        Raises:
            None
    """
    LOGGER.debug('string_to_normalform - Started')
    if repeated_variables is None:
        # find repeated variables
        repeated_variable_dict = {}
        repeated_variables = set()

        query_array = query_string.split(' ')
        for item in query_array:
            if len(item) > 0 and item[0] == '$':
                if item in repeated_variable_dict:
                    repeated_variable_dict[item] += 1
                else:
                    repeated_variable_dict[item] = 1

        for item, value in repeated_variable_dict.items():
            if value > 1:
                repeated_variables.add(item[1:])

    query_string_nf = copy(query_string)
    query_string_nf_prefix = ""
    pos = 0
    while True:
        if pos >= len(query_string_nf) -1:
            break
        if query_string_nf[pos] == "$":
            variable = query_string_nf[pos+1:]
            variable = variable.split(' ')[0]
            LOGGER.debug('string_to_normalform - Current variable %s', variable)
            if variable not in repeated_variables:
                LOGGER.debug('string_to_normalform - Current variable %s is not repeated and has to be deleted', variable)
                # Delete the repeated variable in the query string
                if pos == 0:
                    query_string_nf = query_string_nf[len(variable)+2:]
                    query_string_nf_string_length = query_string_nf.count(" ")+1
                else:
                    query_string_nf_prefix = query_string_nf[0:pos-1]
                    query_string_nf_suffix = query_string_nf[pos+len(variable)+1:]
                    query_string_nf = query_string_nf_prefix + query_string_nf_suffix
                    query_string_nf_string_length = query_string_nf.count(" ")+1
                pos = pos-1

                # Update the local window sizes
                if query_windowsize_local:
                    LOGGER.debug('string_to_normalform - Local window size tuples have to be updated')
                    if pos <= 0:
                        tuple_index = -1
                        window_size_tuple_at_first_index = list(query_windowsize_local[0])
                        window_size_tuple_at_second_index = list(query_windowsize_local[1])
                    else:
                        tuple_index = query_string_nf_prefix.count(" ")+1
                        window_size_tuple_at_first_index = list(query_windowsize_local[tuple_index])
                        window_size_tuple_at_second_index = list(query_windowsize_local[tuple_index+1])

                    # Variable at first position in query string is deleted
                    if tuple_index == -1:
                        LOGGER.debug('string_to_normalform - Update the first local window size tuple')
                        # Variable is also the last block within the query string
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
                    # Variable at last position in query string is deleted
                    elif query_string_nf_string_length==tuple_index:
                        LOGGER.debug('string_to_normalform - Update the last local window size tuple')
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

                    LOGGER.debug('string_to_normalform - Build new list of local window size tuples')
                    updated_windowsize_local = []
                    count = 0
                    while len(query_windowsize_local) > 0:
                        if count == 0 and tuple_index == -1:
                            LOGGER.debug('string_to_normalform - Add new first local window size tuple')
                            updated_windowsize_local.append(window_size_tuple)
                            query_windowsize_local.pop(0)
                            query_windowsize_local.pop(0)
                            count = count+2
                        elif count == tuple_index:
                            LOGGER.debug('string_to_normalform - Add new local window size tuple')
                            updated_windowsize_local.append(window_size_tuple)
                            query_windowsize_local.pop(0)
                            query_windowsize_local.pop(0)
                            count = count+2
                        else:
                            LOGGER.debug('string_to_normalform - Add unchanged local window size tuple')
                            updated_windowsize_local.append(query_windowsize_local[0])
                            query_windowsize_local.pop(0)
                            count = count+1
                    query_windowsize_local = updated_windowsize_local

                LOGGER.debug('string_to_normalform - Current variable %s is now deleted', variable)
            else:
                pos = pos+len(variable)
        pos = pos + 1

    # Variables are renamed in order of first appearance to $x0,$x1 and so on
    gen_variables=list(dict.fromkeys([s for s in query_string_nf.split() if "$" in s]))
    vars_nf= ['$x' + str(i) for i in range(len(gen_variables))]
    new_vars = { v:k for k,v in dict(zip( vars_nf, gen_variables)).items()}
    query_string_nf=" ".join([new_vars.get(item,item)  for item in query_string_nf.split()])

    LOGGER.debug('string_to_normalform - Query string in normalform: %s', query_string)
    LOGGER.debug('string_to_normalform - Finished')

    return query_string_nf, query_windowsize_local
