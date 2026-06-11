#!/usr/bin/python3
"""B-S-S: Bottom-up Separated Symbolic discovery algorithm."""
from math import ceil

from discovery_shared import _domain_separated_discovery
from bsc import discovery_bu_pts_multidim


def _per_domain_bsc(domain_sample, supp, max_query_length, domain_patternset):
    """Call B-S-C for a single domain (used as per_domain_fn in the outer loop)."""
    _, _, result_dict = discovery_bu_pts_multidim(
        sample=domain_sample,
        supp=supp,
        use_smart_matching=True,
        discovery_order='type_first',
        use_tree_structure=True,
        max_query_length=max_query_length,
        find_descriptive_only=False,
        all_patternset=domain_patternset,
    )
    return result_dict


def discover_bss(sample, supp: float, max_query_length: int = -1) -> dict:
    """B-S-S: per-domain B-S-C, then merge across domains.

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
        from bsc import discover_bsc
        return discover_bsc(sample=sample, supp=supp, max_query_length=max_query_length)

    if max_query_length == -1:
        threshold = ceil(sample._sample_size * supp)
        trace_length = sorted([len(trace.split()) for trace in sample._sample])
        max_query_length = trace_length[sample._sample_size - threshold]

    return _domain_separated_discovery(
        sample=sample,
        supp=supp,
        matchtest='pattern-split-sep',
        max_query_length=max_query_length,
        per_domain_fn=_per_domain_bsc,
    )
