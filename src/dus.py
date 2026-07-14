#!/usr/bin/python3
"""D-U-S: Domain Unified Symbolic discovery algorithm."""
from math import ceil

from discovery_shared import _domain_separated_discovery
from duc import discover_duc
from profiling_helpers import profile_call

import logging

LOG_FORMAT = '| %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)


def _per_domain_duc(domain_sample, supp, max_query_length, domain_patternset, profile: bool = False):
    """Call D-U-C for a single domain (used as per_domain_fn in the outer loop)."""
    return discover_duc(
        sample=domain_sample,
        supp=supp,
        max_query_length=max_query_length,
        find_descriptive_only=False,
        all_patternset=domain_patternset,
        profile=profile,
    )


def discover_dus(sample, supp: float, max_query_length: int = -1, profile: bool = False) -> dict:
    """D-U-S: per-domain D-U-C, then merge across domains.

    Args:
        sample: MultidimSample instance.
        supp: Support threshold in [0, 1].
        max_query_length: Maximum query length (-1 = auto-compute).
        profile: If True, collect profiling data for this run. Two things
            get captured:
            1. Each domain's own D-U-C profiling (via discover_duc's
               `profile` flag), collected as this function's per-domain
               loop runs — one entry per domain in
               profiling['per_domain_profiling'].
            2. A cProfile + wall-clock wrap around the *entire* call into
               `discovery_shared._domain_separated_discovery`, since that
               function's merge-phase internals aren't something this file
               can instrument directly. Its cProfile output therefore
               covers everything: per-domain discovery AND the merge step.
               Note: because that outer cProfile is active for the whole
               call, each domain's own inner cProfile (inside discover_duc)
               will typically fail to start (only one cProfile can be
               active per process at a time) and fall back to timing-only —
               this is expected and handled gracefully; the per-domain wall
               clock numbers are still accurate.

    Returns:
        Result dict with keys: queryset, matching_dict, domain_queries,
        merged queries. When profile=True, also includes 'profiling'.
    """
    domain_cnt = sample._sample[0].split(' ')[0].count(';')
    if domain_cnt == 1:
        return discover_duc(sample=sample, supp=supp, max_query_length=max_query_length, profile=profile)

    if max_query_length == -1:
        threshold = ceil(sample._sample_size * supp)
        trace_length = sorted([len(trace.split()) for trace in sample._sample])
        max_query_length = trace_length[sample._sample_size - threshold]

    per_domain_profiling = []

    def _per_domain_duc_tracked(domain_sample, dom_supp, dom_max_query_length, domain_patternset):
        result = _per_domain_duc(domain_sample, dom_supp, dom_max_query_length, domain_patternset, profile=profile)
        if profile:
            per_domain_profiling.append(result.get('profiling'))
        return result

    def _run():
        return _domain_separated_discovery(
            sample=sample,
            supp=supp,
            matchtest='smarter',
            max_query_length=max_query_length,
            per_domain_fn=_per_domain_duc_tracked,
        )

    result_dict, profiling = profile_call(
        _run, enabled=profile, logger=None, label='D-U-S (discover_dus)',
    )

    if profile:
        profiling['per_domain_profiling'] = per_domain_profiling
        result_dict['profiling'] = profiling
        _log_dus_summary(profiling)

    return result_dict


def _log_dus_summary(profiling: dict) -> None:
    LOGGER.info('===== D-U-S profiling summary =====')
    LOGGER.info('Wall clock total (discovery + merge): %.3fs', profiling['wall_clock_total'])
    LOGGER.info('--- Per-domain D-U-C stats ---')
    for i, dom_profiling in enumerate(profiling.get('per_domain_profiling') or []):
        if dom_profiling is None:
            continue
        LOGGER.info(
            'domain %i: wall_clock=%.3fs querycount=%i',
            i, dom_profiling.get('wall_clock_total', 0.0), dom_profiling.get('querycount', 0),
        )
    LOGGER.info(
        'Full cProfile (covers per-domain discovery + merge phase, top 30 by '
        'cumulative time) written to profiling["cprofile_text"]'
    )
    LOGGER.info('=======================================')
