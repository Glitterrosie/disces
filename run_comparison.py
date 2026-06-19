#!/usr/bin/python3
"""Compare all four DISCES discovery algorithms by runtime.

Usage:
    python run_comparison.py [--sample-size N] [--trace-length N]
                             [--dimensions N] [--support FLOAT]
                             [--output PATH]
"""
import argparse
import csv
import os
import sys
import time
import ray

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generator_multidim import MultidimSampleGenerator
from duc import discover_duc
from duc_smartest import discover_duc_smartest

from dus import discover_dus
from dus_smartest import discover_dus_smartest

from bsc import discover_bsc
from bss import discover_bss


ALGORITHMS = [
    ('D-U-C', discover_duc),
    ('D-U-C-S', discover_duc_smartest),

    ('D-U-S', discover_dus),
    ('D-U-S-S', discover_dus_smartest),

    ('B-S-C', discover_bsc),
    ('B-S-S', discover_bss),
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


def print_table(results, sample_size, trace_length, dimensions, supp):
    header = f"{'Algorithm':<10}  {'Time (s)':>10}  {'Queries':>8}  Found"
    sep = '-' * len(header)
    print(f"\nSample: {sample_size} traces × {trace_length} events × {dimensions}D  |  supp={supp}")
    print(sep)
    print(header)
    print(sep)
    indent = ' ' * 34
    for r in results:
        queries = sorted(r['queryset'])
        first = queries[0] if queries else ''
        print(f"{r['algorithm']:<10}  {r['time_s']:>10.4f}  {r['queries_found']:>8}  {first}")
        for q in queries[1:]:
            print(f"{indent}{q}")
    print(sep)


def save_csv(results, output_path, sample_size, trace_length, dimensions, supp):
    file_exists = os.path.isfile(output_path)
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'algorithm', 'sample_size', 'trace_length', 'dimensions',
            'support', 'time_s', 'queries_found',
        ])
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow({
                'algorithm': r['algorithm'],
                'sample_size': sample_size,
                'trace_length': trace_length,
                'dimensions': dimensions,
                'support': supp,
                'time_s': r['time_s'],
                'queries_found': r['queries_found'],
            })
    print(f"\nResults appended to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare DISCES discovery algorithms by runtime.')
    parser.add_argument('--sample-size', type=int, default=20, help='Number of traces (default: 20)')
    parser.add_argument('--trace-length', type=int, default=10, help='Events per trace (default: 10)')
    parser.add_argument('--dimensions', type=int, default=2, help='Event dimensions (default: 2)')
    parser.add_argument('--support', type=float, default=1.0, help='Support threshold 0-1 (default: 1.0)')
    parser.add_argument('--max-query-length', type=int, default=-1, help='Max query length (-1 = auto)')
    parser.add_argument('--type-count', type=int, default=5, help='Number of event types (default: 5)')
    parser.add_argument('--output', type=str, default='results.csv', help='CSV output path (default: results.csv)')
    args = parser.parse_args()

    gen = MultidimSampleGenerator()
    sample = gen.generate_random_sample(
        sample_size=args.sample_size,
        min_trace_length=args.trace_length,
        max_trace_length=args.trace_length,
        event_dimension=args.dimensions,
        type_count=args.type_count,
    )

    results = run_comparison(
        sample=sample,
        supp=args.support,
        max_query_length=args.max_query_length,
    )

    # print sample traces for reference
    for trace in sample._sample:
        print(trace)
    print_table(results, args.sample_size, args.trace_length, args.dimensions, args.support)
    save_csv(results, args.output, args.sample_size, args.trace_length, args.dimensions, args.support)


if __name__ == '__main__':
    main()
