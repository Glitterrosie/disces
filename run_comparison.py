#!/usr/bin/python3
"""Compare all four DISCES discovery algorithms by runtime.

Usage:
    python run_comparison.py [--sample-size N] [--min-trace-length N] [--max-trace-length N]
                             [--dimensions N] [--support FLOAT]
                             [--output PATH]
"""
import argparse
import csv
import functools
import os
import sys
import time
import ray
import numpy as np

os.environ.setdefault('RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO', '0')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generator_multidim import MultidimSampleGenerator
from duc import discover_duc
from duct import discover_duc_tree
from ducm import discover_ducm, partition_traces_naive, partition_traces_by_length


from dus import discover_dus
from dusd import discover_dus_dimension
from dusm import discover_dusm

from bsc import discover_bsc
from bss import discover_bss


ALGORITHMS = [
    ('D-U-C', discover_duc),
    ('D-U-C-T', discover_duc_tree),
    ('D-U-C-M-naive', functools.partial(discover_ducm, partition_fn=partition_traces_naive)),
    ('D-U-C-M-balanced', functools.partial(discover_ducm, partition_fn=partition_traces_by_length)),

    ('D-U-S', discover_dus),
    ('D-U-S-D', discover_dus_dimension),
    ('D-U-S-M', discover_dusm),
]

ray.init(runtime_env={"env_vars": {"PYTHONPATH": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')}}, ignore_reinit_error=True)


def run_comparison(sample, supp: float, max_query_length: int = -1):
    results = []
    for name, fn in ALGORITHMS:
        t0 = time.perf_counter()
        result = fn(sample=sample, supp=supp, max_query_length=max_query_length)
        elapsed = time.perf_counter() - t0
        queryset = result.get('queryset', set())
        results.append({
            'algorithm': name,
            'time_s': round(elapsed, 4),
            'queries_found': len(queryset),
            'queryset': queryset,
        })
    return results


def print_table(results, sample_size, trace_length_min, trace_length_max, dimensions, supp):
    name_width = max(10, max(len(r['algorithm']) for r in results))
    header = f"{'Algorithm':<{name_width}}  {'Time (s)':>10}  {'Queries':>8}  Found"
    sep = '-' * len(header)
    if trace_length_min == trace_length_max:
        trace_length_desc = str(trace_length_min)
    else:
        trace_length_desc = f"{trace_length_min}-{trace_length_max}"
    print(f"\nSample: {sample_size} traces × {trace_length_desc} events × {dimensions}D  |  supp={supp}")
    print(sep)
    print(header)
    print(sep)
    indent = ' ' * (name_width + 24)
    for r in results:
        queries = sorted(r['queryset'])
        first = queries[0] if queries else ''
        print(f"{r['algorithm']:<{name_width}}  {r['time_s']:>10.4f}  {r['queries_found']:>8}  {first}")
        for q in queries[1:]:
            print(f"{indent}{q}")
    print(sep)


def save_csv(results, output_path, sample_size, trace_length_min, trace_length_max, dimensions, supp):
    file_exists = os.path.isfile(output_path)
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'algorithm', 'sample_size', 'trace_length_min', 'trace_length_max', 'dimensions',
            'support', 'time_s', 'queries_found',
        ])
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow({
                'algorithm': r['algorithm'],
                'sample_size': sample_size,
                'trace_length_min': trace_length_min,
                'trace_length_max': trace_length_max,
                'dimensions': dimensions,
                'support': supp,
                'time_s': r['time_s'],
                'queries_found': r['queries_found'],
            })
    print(f"\nResults appended to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare DISCES discovery algorithms by runtime.')
    parser.add_argument('--sample-size', type=int, default=20, help='Number of traces (default: 20)')
    parser.add_argument('--min-trace-length', type=int, default=10, help='Minimum events per trace (default: 10)')
    parser.add_argument('--max-trace-length', type=int, default=10, help='Maximum events per trace (default: 10)')
    parser.add_argument('--dimensions', type=int, default=2, help='Event dimensions (default: 2)')
    parser.add_argument('--support', type=float, default=1.0, help='Support threshold 0-1 (default: 1.0)')
    parser.add_argument('--max-query-length', type=int, default=-1, help='Max query length (-1 = auto)')
    parser.add_argument('--type-count', type=int, default=5, help='Number of event types (default: 5)')
    parser.add_argument('--output', type=str, default='results.csv', help='CSV output path (default: results.csv)')
    args = parser.parse_args()

    if args.min_trace_length > args.max_trace_length:
        parser.error('--min-trace-length must not exceed --max-trace-length')

    gen = MultidimSampleGenerator()
    sample = gen.generate_random_sample(
        sample_size=args.sample_size,
        min_trace_length=args.min_trace_length,
        max_trace_length=args.max_trace_length,
        event_dimension=args.dimensions,
        type_count=args.type_count,
    )

    results = run_comparison(
        sample=sample,
        supp=args.support,
        max_query_length=args.max_query_length,
    )

    print_table(results, args.sample_size, args.min_trace_length, args.max_trace_length, args.dimensions, args.support)
    save_csv(results, args.output, args.sample_size, args.min_trace_length, args.max_trace_length, args.dimensions, args.support)


if __name__ == '__main__':
    main()
