#!/usr/bin/python3
"""
Naive online wrapper for D-U-C-M.

Design
------
No incremental computation is performed. Every time new
sequences are added to the database, we:

  1. append the new trace strings to the (in-memory / on-disk) sequence
     store, using the same serialization as the offline implementation
     (whitespace-separated events, ';'-separated domain values within an
     event, e.g. "a1;b1 a2;b2 ..."),
  2. rebuild a MultidimSample from the entire updated store
  3. call `discover_ducm` again on the whole sample, from scratch,
  4. replace the previous result set with the new one.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import ray

from generator_multidim import MultidimSampleGenerator
from sample_multidim import MultidimSample

# Adjust this import to wherever discover_ducm / partition_traces_by_length
# actually live in your project (the module shown to me doesn't give its
# own filename).
from ducm import discover_ducm, partition_traces_by_length

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sequence database (append-only)
# ---------------------------------------------------------------------------

class SequenceStore:

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._traces: List[str] = []
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            self._traces = [
                line for line in self.path.read_text().splitlines() if line
            ]

    def append(self, new_traces: List[str]) -> None:
        with self._lock:
            self._traces.extend(new_traces)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    for trace in new_traces:
                        f.write(trace + "\n")

    def snapshot(self) -> List[str]:
        """A consistent copy of the current full sequence database."""
        with self._lock:
            return list(self._traces)

    def __len__(self) -> int:
        with self._lock:
            return len(self._traces)


def build_sample(traces: List[str]) -> MultidimSample:
    sample = MultidimSample()
    sample.set_sample(traces)
    sample.calc_sample_typeset(calculate_all=True)
    return sample


class OnlineDucmWrapper:
    """
    Wraps discover_ducm to support adding complete sequences at runtime.
    Every add_sequences() call triggers a full re-run of discover_ducm
    over the whole updated database; there is no incremental state kept
    across calls other than the raw sequence store itself.
    """

    def __init__(
        self,
        supp: float,
        store: Optional[SequenceStore] = None,
        max_query_length: int = -1,
        only_types: bool = False,
        find_descriptive_only: bool = True,
        partition_fn: Optional[Callable] = None,
        profile: bool = False,
        ray_init_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.supp = supp
        self.max_query_length = max_query_length
        self.only_types = only_types
        self.find_descriptive_only = find_descriptive_only
        self.partition_fn = partition_fn or partition_traces_by_length
        self.profile = profile

        self.store = store or SequenceStore()
        self._result_lock = threading.Lock()
        self._latest_result: Optional[dict] = None
        self._latest_runtime: Optional[float] = None
        self._latest_db_size: int = 0
        self._run_count: int = 0

        if not ray.is_initialized():
            ray.init(**(ray_init_kwargs or {}))

    # -- core operation --------------------------------------------------

    def add_sequences(self, new_traces: List[str]) -> dict:
        """
        Append `new_traces` (already-complete sequences, in DISCES trace
        format) to the database and rerun discover_ducm on the entire
        updated database. Blocks until the rerun has finished; returns
        the fresh result dict (same keys discover_ducm returns).
        """
        if not new_traces:
            # Nothing changed -> nothing to recompute; return cached result.
            latest = self.get_latest_result()
            if latest is None:
                raise ValueError("add_sequences called with no traces and no prior result exists")
            return latest

        self.store.append(new_traces)
        traces = self.store.snapshot()
        sample = build_sample(traces)

        actors_before = self._live_actor_count()
        t0 = time.perf_counter()

        result = discover_ducm(
            sample,
            supp=self.supp,
            max_query_length=self.max_query_length,
            only_types=self.only_types,
            find_descriptive_only=self.find_descriptive_only,
            all_patternset=None,  # must be recomputed: patternsets are whole-sample dependent
            partition_fn=self.partition_fn,
            profile=self.profile,
        )

        elapsed = time.perf_counter() - t0
        self._check_no_actor_leak(actors_before)

        with self._result_lock:
            self._latest_result = result
            self._latest_runtime = elapsed
            self._latest_db_size = len(traces)
            self._run_count += 1

        throughput = len(traces) / elapsed if elapsed > 0 else float("inf")
        LOGGER.info(
            "run #%d: reprocessed %d sequences (+%d new) in %.3fs "
            "-> %d descriptive queries (throughput: %.2f seq/s)",
            self._run_count, len(traces), len(new_traces), elapsed,
            len(result["queryset"]), throughput,
        )

        return result

    # -- accessors ---------------------------------------------------------

    def get_latest_result(self) -> Optional[dict]:
        with self._result_lock:
            return self._latest_result

    def get_latest_stats(self) -> dict:
        with self._result_lock:
            return {
                "run_count": self._run_count,
                "db_size": self._latest_db_size,
                "runtime_seconds": self._latest_runtime,
                "throughput_seq_per_sec": (
                    self._latest_db_size / self._latest_runtime
                    if self._latest_runtime else None
                ),
            }

    # -- internal safety net -----------------------------------------------

    @staticmethod
    def _live_actor_count() -> int:
        try:
            return len(ray.util.list_named_actors(all_namespaces=True))
        except Exception:
            # Not all Ray versions/backends support this the same way;
            # fail open rather than block on a diagnostics helper.
            return -1

    def _check_no_actor_leak(self, actors_before: int) -> None:
        if actors_before < 0:
            return
        actors_after = self._live_actor_count()
        if actors_after > actors_before:
            LOGGER.warning(
                "Ray actor count grew from %d to %d after a discover_ducm run -- "
                "ChunkWorker actors may not be getting garbage collected "
                "(check they are still non-detached).",
                actors_before, actors_after,
            )


# ---------------------------------------------------------------------------
# Benchmark harness for the online-setting throughput measurements.
# ---------------------------------------------------------------------------

def benchmark_online(
    wrapper: OnlineDucmWrapper, batches: List[List[str]]
) -> List[Dict[str, Any]]:
    """
    Feed `batches` of new sequences to `wrapper` one batch at a time and
    record per-batch timing/throughput. A batch of size 1 measures
    per-sequence-arrival latency; larger batches measure the coarser but
    cheaper "ingest in bulk" mode.
    """
    stats = []
    for i, batch in enumerate(batches):
        t0 = time.perf_counter()
        wrapper.add_sequences(batch)
        elapsed = time.perf_counter() - t0
        db_size = len(wrapper.store)
        stats.append({
            "batch_index": i,
            "batch_size": len(batch),
            "db_size_after": db_size,
            "elapsed_seconds": elapsed,
            "throughput_new_seq_per_sec": len(batch) / elapsed if elapsed > 0 else float("inf"),
            "throughput_total_seq_per_sec": db_size / elapsed if elapsed > 0 else float("inf"),
        })
    return stats


# ---------------------------------------------------------------------------
# Example usage / sanity check against a static full run.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="| %(message)s")

    gen = MultidimSampleGenerator()
    sample = gen.generate_random_sample(
        sample_size=50,
        min_trace_length=7,
        max_trace_length=7,
        event_dimension=2,
        type_count=4,
    )
    all_traces = sample._sample

    batch_size = 5
    batches = [all_traces[i:i + batch_size] for i in range(0, len(all_traces), batch_size)]

    wrapper = OnlineDucmWrapper(supp=1)
    perf_stats = benchmark_online(wrapper, batches)
    for row in perf_stats:
        print(row)

    # Sanity check: running discover_ducm once, directly, on the full
    # static database must give the same descriptive queryset as ending
    # up at that same database through the online wrapper.
    static_sample = build_sample(all_traces)
    static_result = discover_ducm(static_sample, supp=1)
    online_result = wrapper.get_latest_result()
    assert static_result["queryset"] == online_result["queryset"], (
        "Online wrapper's final result diverges from a direct static run -- "
        "investigate before trusting throughput numbers."
    )
    print("Sanity check passed: online == static on the same final database.")
