#!/usr/bin/python3
"""D-U-S: Domain Unified Symbolic discovery algorithm."""
from discovery_shared import _domain_separated_discovery
from duc import discover_duc


def _per_domain_duc(domain_sample, supp, max_query_length, domain_patternset):
    """Call D-U-C for a single domain (used as per_domain_fn in the outer loop)."""
    return discover_duc(
        sample=domain_sample,
        supp=supp,
        max_query_length=max_query_length,
        find_descriptive_only=False,
        all_patternset=domain_patternset,
    )


def discover_dus(sample, supp: float, max_query_length: int = -1) -> dict:
    """D-U-S: per-domain D-U-C, then merge across domains.

    Args:
        sample: MultidimSample instance.
        supp: Support threshold in [0, 1].
        max_query_length: Maximum query length (-1 = auto-compute).

    Returns:
        Result dict with keys: queryset, matching_dict, domain_queries,
        merged queries.
    """
    domain_cnt = sample._sample[0].split(' ')[0].count(';')
    if domain_cnt == 1:
        return discover_duc(sample=sample, supp=supp, max_query_length=max_query_length)

    if max_query_length == -1:
        from math import ceil
        threshold = ceil(sample._sample_size * supp)
        trace_length = sorted([len(trace.split()) for trace in sample._sample])
        max_query_length = trace_length[sample._sample_size - threshold]

    return _domain_separated_discovery(
        sample=sample,
        supp=supp,
        matchtest='smarter',
        max_query_length=max_query_length,
        per_domain_fn=_per_domain_duc,
    )
