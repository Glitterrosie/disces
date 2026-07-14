#!/usr/bin/python3
"""Shared profiling utilities for the discovery algorithms (D-U-C, D-U-C-tree,
D-U-S, D-U-S-dimension, D-U-C-M, D-U-S-M — duc.py, duct.py, dus.py, dusd.py,
ducm.py, dusm.py).

These algorithms share a handful of shapes even though their distribution
strategy differs (no Ray at all, persistent chunk-worker actors, one-shot
remote functions, or an opaque external call we can't instrument directly).
This module factors out the profiling machinery for all of them:

- cprofile_enable / cprofile_disable / cprofile_text: cProfile helpers that
  tolerate an already-active profiler (e.g. a PyCharm scientific-mode
  notebook or a Jupyter debugger installs its own tracing hook, and
  cProfile raises ValueError if you try to stack a second one on top).
  Profiling degrades gracefully to timing-only instead of crashing the run.
- WorkerProfiler: wraps a single unit of work (a persistent actor's match
  call, or a one-shot remote function's whole task body) with optional
  cProfile + wall-clock timing. No-ops entirely when disabled. Despite the
  name, nothing about it is Ray-specific — it works for any callable.
- DriverProfiler: tracks wall clock plus dispatch/collection timing for one
  "send work, wait for it" phase, whether that's many small rounds (D-U-C-M's
  BFS loop) or a single round (D-U-C-tree's one-shot future dispatch, or a
  plain sequential match_sample loop with no dispatch phase at all).
- profile_call: wraps one opaque function call in cProfile + wall-clock
  timing. Useful when the call goes into code you don't control (e.g.
  dus.py's call into discovery_shared._domain_separated_discovery) and
  can't be instrumented with WorkerProfiler/DriverProfiler internally —
  cProfile still sees everything that runs underneath it.
- log_profiling_summary: prints a compact, human-readable report built from
  a DriverProfiler summary + a list of WorkerProfiler stats dicts.
"""
import cProfile
import io
import pstats
import time

__all__ = [
    'cprofile_enable',
    'cprofile_disable',
    'cprofile_text',
    'CallTimer',
    'WorkerProfiler',
    'DriverProfiler',
    'profile_call',
    'log_profiling_summary',
    'log_sequential_profiling_summary',
]


# ---------------------------------------------------------------------------
# Low-level cProfile helpers
# ---------------------------------------------------------------------------

def cprofile_enable(profiler: cProfile.Profile) -> bool:
    """Enable a cProfile.Profile, tolerating an already-active profiler.

    Returns True if profiling actually started, False if it couldn't
    (another profiler/tracer was already active in this process).
    """
    try:
        profiler.enable()
        return True
    except ValueError:
        return False


def cprofile_disable(profiler: cProfile.Profile, was_enabled: bool) -> None:
    """Disable a cProfile.Profile that was started with cprofile_enable."""
    if was_enabled:
        try:
            profiler.disable()
        except ValueError:
            pass


def cprofile_text(profiler: cProfile.Profile, top_n: int = 20) -> str:
    """Return a pstats text dump (sorted by cumulative time) for `profiler`."""
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(top_n)
    return stream.getvalue()


def profile_call(fn, *args, enabled: bool = True, top_n: int = 30, logger=None,
                  label: str = 'call', **kwargs):
    """Run fn(*args, **kwargs) wrapped in cProfile + wall-clock timing.

    Useful for opaque calls you can't instrument internally (e.g. a call
    into external/shared code). cProfile still captures everything that
    executes underneath the call, function-level, regardless of whether you
    have that code in front of you.

    Returns (result, profiling_dict_or_None). When enabled=False, just calls
    fn directly and returns (result, None) with no overhead.

    profiling_dict has keys: wall_clock_total, cprofile_text.
    """
    if not enabled:
        return fn(*args, **kwargs), None

    profiler = cProfile.Profile()
    cprofile_ok = cprofile_enable(profiler)
    if not cprofile_ok and logger is not None:
        logger.info(
            'cProfile already active in this process (e.g. PyCharm/Jupyter '
            'debugger) — skipping cProfile for %s, wall-clock timing still recorded.',
            label,
        )

    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        cprofile_disable(profiler, cprofile_ok)
        elapsed = time.perf_counter() - t0

    if cprofile_ok:
        text = cprofile_text(profiler, top_n)
    else:
        text = (
            '(cProfile unavailable — another profiler was already active in '
            'this process, e.g. a PyCharm/Jupyter debugger. Wall-clock timing '
            'above is still accurate.)'
        )

    profiling = {'wall_clock_total': elapsed, 'cprofile_text': text}

    if logger is not None:
        logger.info('===== %s summary =====', label)
        logger.info('Wall clock total: %.3fs', elapsed)
        logger.info(
            'cProfile (top %d by cumulative time) written to profiling["cprofile_text"]',
            top_n,
        )
        logger.info('=======================================')

    return result, profiling


# ---------------------------------------------------------------------------
# Lightweight repeated-call timing (no cProfile) for single-process hot loops
# ---------------------------------------------------------------------------

class CallTimer:
    """Lightweight per-call timing (no cProfile) for repeated calls made
    inside a block that's already wrapped by an enclosing DriverProfiler's
    cProfile — nesting a second cProfile.Profile in the same process/thread
    just gets skipped anyway (see cprofile_enable), so there's no point
    trying. Use this for single-process, non-Ray hot loops, e.g. duc.py's
    per-query match_sample calls, or a sequential merge phase.

    Usage:

        call_timer = CallTimer(enabled=profile)
        ...
        matching = call_timer.record(query.match_sample, sample=sample, ...)
        ...
        stats = call_timer.stats()  # None if not enabled
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.call_count = 0
        self.total_time = 0.0
        self.max_time = 0.0
        self.min_time = float('inf')

    def record(self, fn, *args, **kwargs):
        if not self.enabled:
            return fn(*args, **kwargs)
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            self.call_count += 1
            self.total_time += elapsed
            self.max_time = max(self.max_time, elapsed)
            self.min_time = min(self.min_time, elapsed)

    def stats(self) -> dict | None:
        if not self.enabled:
            return None
        return {
            'call_count': self.call_count,
            'total_time': self.total_time,
            'avg_time': self.total_time / self.call_count if self.call_count else 0.0,
            'max_time': self.max_time,
            'min_time': self.min_time if self.call_count else 0.0,
        }


# ---------------------------------------------------------------------------
# Worker-side profiling (owned by a ChunkWorker actor)
# ---------------------------------------------------------------------------

class WorkerProfiler:
    """Per-actor profiling state. Owned by a ChunkWorker instance.

    Usage inside a ChunkWorker actor:

        def __init__(self, ..., profile=False):
            self._profiler = WorkerProfiler(profile)

        def match(self, query):
            def _do_match():
                ...actual matching logic...
                return matched, matched_traces
            return self._profiler.record(_do_match)

        def get_profile_stats(self):
            return self._profiler.stats(self.chunk_id, self.chunk._sample_size)

    When `enabled=False`, `record` just calls the function directly with no
    timing/profiling overhead.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.call_count = 0
        self.total_time = 0.0
        self.max_time = 0.0
        self.min_time = float('inf')
        self._cprofile = cProfile.Profile() if enabled else None
        self._cprofile_available = True  # flips False if enable() ever fails

    def record(self, fn, *args, **kwargs):
        """Call fn(*args, **kwargs), optionally profiled, and record timing.

        Returns fn's return value unchanged.
        """
        if not self.enabled:
            return fn(*args, **kwargs)

        t0 = time.perf_counter()
        cprofile_ok = self._cprofile_available and cprofile_enable(self._cprofile)
        if not cprofile_ok:
            self._cprofile_available = False
        try:
            result = fn(*args, **kwargs)
        finally:
            cprofile_disable(self._cprofile, cprofile_ok)
            elapsed = time.perf_counter() - t0
            self.call_count += 1
            self.total_time += elapsed
            self.max_time = max(self.max_time, elapsed)
            self.min_time = min(self.min_time, elapsed)
        return result

    def stats(self, chunk_id, chunk_size) -> dict | None:
        """Return a summary dict, or None if profiling wasn't enabled."""
        if not self.enabled:
            return None

        if self._cprofile_available and self.call_count > 0:
            text = cprofile_text(self._cprofile)
        else:
            text = (
                '(cProfile unavailable in this actor process — another '
                'profiler was already active; timing stats below are still valid)'
            )

        return {
            'chunk_id': chunk_id,
            'chunk_size': chunk_size,
            'call_count': self.call_count,
            'total_match_time': self.total_time,
            'avg_match_time': self.total_time / self.call_count if self.call_count else 0.0,
            'max_match_time': self.max_time,
            'min_match_time': self.min_time if self.call_count else 0.0,
            'cprofile_text': text,
        }


# ---------------------------------------------------------------------------
# Driver-side profiling (owned by the top-level discover_* function, or any
# distributed-matching phase such as the D-U-S-M merge phase)
# ---------------------------------------------------------------------------

class _Timer:
    """Context manager that accumulates elapsed time into a DriverProfiler
    attribute. No-ops entirely when the owning profiler is disabled.
    """

    def __init__(self, profiler: 'DriverProfiler', attr_name: str, also_round: bool = False):
        self.profiler = profiler
        self.attr_name = attr_name
        self.also_round = also_round
        self._t0 = None

    def __enter__(self):
        if self.profiler.enabled:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.profiler.enabled:
            elapsed = time.perf_counter() - self._t0
            setattr(self.profiler, self.attr_name, getattr(self.profiler, self.attr_name) + elapsed)
            if self.also_round:
                self.profiler.round_times.append(elapsed)
        return False


class DriverProfiler:
    """Driver-side profiling state for one distributed-matching phase.

    Usage:

        dp = DriverProfiler(enabled=profile)
        dp.start()
        ...
        for query in queries:
            with dp.time_dispatch():
                futures = [w.match.remote(query) for w in workers]
            with dp.time_collection():
                for f in futures:
                    matched = ray.get(f)
                    if not matched:
                        dp.record_early_stop()
                        break
        ...
        worker_stats = ray.get([w.get_profile_stats.remote() for w in workers])
        dp.stop()
        timeline_path = dp.write_ray_timeline('phase_timeline.json')
        summary = dp.build_summary(querycount, worker_stats, timeline_path)
        log_profiling_summary(LOGGER, summary, label='My phase')

    When `enabled=False`, every method is a cheap no-op, so call sites don't
    need to branch on whether profiling is on.

    Pass `use_cprofile=False` when this DriverProfiler wraps a phase that
    itself calls into code which does its own cProfile-based profiling
    (e.g. dus.py's outer wrapper around per-domain discover_duc calls that
    each run their own cProfile) — only one cProfile.Profile can be active
    in a process/thread at a time, so trying to hold one here would just
    make the inner ones silently fall back to timing-only. With
    use_cprofile=False, wall-clock and dispatch/collection timing still work
    normally; driver_cprofile_text() just returns ''.
    """

    def __init__(self, enabled: bool, use_cprofile: bool = True):
        self.enabled = enabled
        self.use_cprofile = enabled and use_cprofile
        self._cprofile = cProfile.Profile() if self.use_cprofile else None
        self._cprofile_ok = False
        self.dispatch_time_total = 0.0
        self.collection_time_total = 0.0
        self.round_times = []
        self.non_match_early_stop_count = 0
        self._wall_start = None
        self.wall_clock_total = 0.0

    def start(self, logger=None):
        if not self.enabled:
            return
        if self.use_cprofile:
            self._cprofile_ok = cprofile_enable(self._cprofile)
            if not self._cprofile_ok and logger is not None:
                logger.info(
                    'cProfile already active in this process (e.g. PyCharm/Jupyter '
                    'debugger) — skipping driver cProfile, keeping timing-based stats.'
                )
        self._wall_start = time.perf_counter()

    def stop(self):
        if not self.enabled:
            return
        if self.use_cprofile:
            cprofile_disable(self._cprofile, self._cprofile_ok)
        self.wall_clock_total = time.perf_counter() - self._wall_start

    def time_dispatch(self) -> _Timer:
        return _Timer(self, 'dispatch_time_total')

    def time_collection(self) -> _Timer:
        return _Timer(self, 'collection_time_total', also_round=True)

    def record_early_stop(self) -> None:
        if self.enabled:
            self.non_match_early_stop_count += 1

    def driver_cprofile_text(self) -> str:
        if not self.enabled or not self.use_cprofile:
            return ''
        if self._cprofile_ok:
            return cprofile_text(self._cprofile)
        return (
            '(cProfile unavailable — another profiler was already active in '
            'this process, e.g. a PyCharm/Jupyter debugger. Timing-based '
            'stats above are still accurate.)'
        )

    def write_ray_timeline(self, path: str, logger=None):
        """Best-effort dump of a chrome://tracing-compatible timeline."""
        if not self.enabled:
            return None
        try:
            import ray
            ray.timeline(filename=path)
            return path
        except Exception as exc:  # pragma: no cover - defensive, timeline is best-effort
            if logger is not None:
                logger.info('Could not write ray timeline: %s', exc)
            return None

    def build_summary(self, querycount: int, worker_stats_list, timeline_path=None) -> dict:
        return {
            'wall_clock_total': self.wall_clock_total,
            'dispatch_time_total': self.dispatch_time_total,
            'collection_time_total': self.collection_time_total,
            'avg_round_time': (sum(self.round_times) / len(self.round_times)
                                if self.round_times else 0.0),
            'max_round_time': max(self.round_times) if self.round_times else 0.0,
            'non_match_early_stop_count': self.non_match_early_stop_count,
            'querycount': querycount,
            'worker_stats': worker_stats_list,
            'driver_cprofile_text': self.driver_cprofile_text(),
            'ray_timeline_path': timeline_path,
        }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def log_profiling_summary(logger, profiling: dict, label: str = 'profiling') -> None:
    """Print a compact, human-readable profiling report built by
    DriverProfiler.build_summary(), with worker stats from WorkerProfiler.stats().
    """
    logger.info('===== %s summary =====', label)
    logger.info('Wall clock total:      %.3fs', profiling['wall_clock_total'])
    logger.info('Dispatch time total:   %.3fs', profiling['dispatch_time_total'])
    logger.info('Collection time total: %.3fs', profiling['collection_time_total'])
    logger.info('Avg round time:        %.5fs', profiling['avg_round_time'])
    logger.info('Max round time:        %.5fs', profiling['max_round_time'])
    logger.info('Query count:           %i', profiling['querycount'])
    logger.info(
        'Early non-match stops: %i (%.1f%% of rounds)',
        profiling['non_match_early_stop_count'],
        100.0 * profiling['non_match_early_stop_count'] / max(profiling['querycount'], 1),
    )
    logger.info('--- Per-worker stats ---')
    for ws in profiling['worker_stats'] or []:
        if ws is None:
            continue
        logger.info(
            'chunk %i (size=%i): calls=%i total=%.3fs avg=%.5fs max=%.5fs min=%.5fs',
            ws['chunk_id'], ws['chunk_size'], ws['call_count'],
            ws['total_match_time'], ws['avg_match_time'],
            ws['max_match_time'], ws['min_match_time'],
        )
    if profiling.get('ray_timeline_path'):
        logger.info(
            'Ray timeline written to %s (load in chrome://tracing)',
            profiling['ray_timeline_path'],
        )
    logger.info('Driver cProfile (top 20 by cumulative time) written to profiling["driver_cprofile_text"]')
    logger.info('Per-worker cProfile dumps available in profiling["worker_stats"][i]["cprofile_text"]')
    logger.info('=======================================')


def log_sequential_profiling_summary(logger, profiling: dict, label: str = 'profiling') -> None:
    """Print a compact report for a single-process (non-distributed) run,
    built from a DriverProfiler wall-clock + a CallTimer's call_stats.

    Expects profiling to have 'wall_clock_total' and, optionally,
    'call_stats' (from CallTimer.stats()) and 'cprofile_text'.
    """
    logger.info('===== %s summary =====', label)
    logger.info('Wall clock total: %.3fs', profiling.get('wall_clock_total', 0.0))
    call_stats = profiling.get('call_stats')
    if call_stats:
        logger.info(
            'Match calls: count=%i total=%.3fs avg=%.5fs max=%.5fs min=%.5fs',
            call_stats['call_count'], call_stats['total_time'],
            call_stats['avg_time'], call_stats['max_time'], call_stats['min_time'],
        )
    if profiling.get('cprofile_text'):
        logger.info('cProfile (top 20 by cumulative time) written to profiling["cprofile_text"]')
    logger.info('=======================================')
