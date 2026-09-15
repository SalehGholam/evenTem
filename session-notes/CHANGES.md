# Changes made to EvenTem this session

This documents every change made to the actual EvenTem C++/Python source in
`EvenTem/src/` and `EvenTem/eventem.py`, on top of the unmodified project. It covers
two pieces of work: (1) a new feature, charge-weighted declustered ROI extraction,
and (2) two real crash bugs found while testing it, plus parallelizing the
declustering algorithm.

Everything here is scoped to the **CHEETAH (.tpx3) detector path** and the `Roi`
processing mode, per the focus of this session. `Ricom`, the GPRI/LibTorch GPU path,
and the Merlin/Advapix/Numpy detector backends were not touched.

No behavior changes for existing users: everything below is either a new, opt-in
field (default value preserves the old behavior exactly) or a bug fix in code that
was either dead (never exercised with `n_threads > 1`) or leaking memory without
otherwise affecting output. `Electron`'s existing `.electron` conversion output is
unchanged (regression-tested: identical to pre-change output on real data).

---

## 1. New feature: `Roi.Decluster` / `Roi.TotPerElectron`

**What it does:** today, ROI extraction from a `.tpx3` file counts every raw Timepix3
pixel activation as one electron. In reality, one physical electron typically lights
up several adjacent pixels (charge sharing), and occasionally two electrons land close
together in space and time (pile-up). This feature groups raw activations into
clusters (the same dspace/dtime/cluster_range windowed grouping the existing
`Electron` class already uses for `.electron` conversion), resolves each cluster's
ToT-weighted centroid position, and credits that position with
`round(cluster_total_ToT / TotPerElectron)` electrons instead of a flat `+1` per raw
hit -- so ordinary charge-sharing collapses to 1 electron, and genuine multi-electron
pile-up correctly resolves to more than 1.

**New Python-facing `Roi` fields** (`EvenTem/src/pybind.cpp`, `EvenTem/eventem.py`):

| snake_case (raw `eventem` module) | CamelCase (`EvenTem.Roi` wrapper) | Meaning |
|---|---|---|
| `decluster` (bool, default `False`) | `Decluster` | Turn the feature on. Off by default -- unchanged behavior unless set. |
| `dtime` (default 100) | `Dtime` | Max ToA-tick gap between two hits in the same cluster. |
| `dspace` (default 6) | `Dspace` | Max pixel distance between two hits in the same cluster. |
| `cluster_range` (default 256) | `ClusterRange` | How many subsequent hits are checked as merge candidates. |
| `tot_per_electron` (no default -- required) | `TotPerElectron` | Calibration constant: the typical summed ToT of a single electron's cluster at your beam energy/threshold. Must be read off your own data (e.g. the new histogram below); `decluster=True` with this unset or `<=0` raises `ValueError`/`std::invalid_argument` immediately. |
| `clustersize_histogram` (readonly) | *(same, unwrapped)* | Histogram of resolved cluster sizes. |
| `energy_histogram` (readonly) | *(same, unwrapped)* | Histogram of resolved clusters' total summed ToT. **Caveat:** bin 0 conflates genuinely zero-ToT clusters with raw hits merged into another cluster -- use `clustersize_tot_histogram` instead for anything involving zero/low ToT. |
| `clustersize_tot_histogram` (readonly) | `ClustersizeTotHistogram` | **New**: joint 2D histogram, shape `(50, 4096)` = `[cluster size][total ToT]`. One entry per real, resolved cluster -- doesn't have the bin-0 problem `energy_histogram` has. This is the "hits vs. summed ToT" calibration plot. |

Only supported for `.tpx3` (CHEETAH) files -- `decluster=True` on any other camera
type raises `std::runtime_error` immediately, before any processing starts.

**Files touched:**
- `EvenTem/src/utils/ClusterResolver.hpp` (**new file**) -- `ClusterInfo` struct
  (resolved cluster's centroid + total ToT) and `resolve_electron_count()`, the
  `round(tot_sum / tot_per_electron)` formula, floored at 0 (not 1 -- a cluster far
  below the calibration constant is treated as noise, not a real electron).
- `EvenTem/src/utils/Declusterer.hpp` -- added ToT-weighted centroid tracking
  (`wx_buffer`/`wy_buffer`) alongside the existing per-cluster ToT summation; added an
  optional `p_cluster_callback` hook invoked once per kept cluster with its seed hit
  and resolved `ClusterInfo` (used by `Roi`; `nullptr` by default, so `Electron`'s
  behavior is untouched); added `disable_file_write` (skips the `.electron` write path
  entirely when driven by a callback-only consumer like `Roi`, which has no output
  file); added the joint histogram accumulation described above.
- `EvenTem/src/utils/Roi4D.hpp` -- added `increment_by(x, y, kx, ky, n)` alongside the
  existing `increment()`, so a single resolved cluster can add `n > 1` to one 4D-cube
  voxel (needed for pile-up). Existing `increment()` (still used by the raw,
  non-declustered path) is unchanged.
- `EvenTem/src/detectors/Timepix.hpp` -- added `enable_roi_declustered<T>(...)`, which
  wires a `Roi` object's declustered extraction through the same `Declusterer` +
  `CHEETAH` machinery `Electron` already uses, via the new callback hook. Also added a
  `roi_declustered_dummy_file` member (an unopened `std::ofstream`, needed only to
  satisfy `Declusterer::init()`'s file-reference parameter when file writing is
  disabled) and `#include <fstream>`.
- `EvenTem/src/core/Roi.h` / `Roi.cpp` -- added the new fields listed above; `run()`
  now throws upfront if `decluster=True` is combined with a non-CHEETAH camera or an
  unset `tot_per_electron`, before touching the file.
- `EvenTem/src/pybind.cpp` -- bindings for the new `Roi` fields.
- `EvenTem/eventem.py` -- `Roi.Decluster` / `.Dtime` / `.Dspace` / `.ClusterRange` /
  `.TotPerElectron` / `.ClustersizeTotHistogram` CamelCase properties, matching the
  existing style already used for `Electron`'s equivalent fields.

**Known limitation carried over from the raw path, not new:** the `_stack` variants
(`Roi_scan_image_stack`, `Roi_diffraction_pattern_stack`) aren't populated in
declustered mode, same as the existing `extract_4D=True` path.

---

## 2. Two crash bugs found, both fixed

Both were specific to the `decluster=True` path added above -- `Electron`'s existing,
untouched `decluster=True` behavior for `.electron` conversion was never affected by
either bug (verified: its output is byte-for-byte identical before/after this
session's changes).

### 2.1 `Declusterer` leaked native memory on every `.run()` call

**File:** `EvenTem/src/utils/Declusterer.hpp`

> This one turned out to be the root cause of two separate symptoms that initially
> looked unrelated: the "second `decluster=True` call in a process segfaults" crash,
> *and* the "calling `.run()` inside a Jupyter cell kills the kernel" crash. See the
> measurement at the end of this section.

Two members were raw `new`'d pointers with no corresponding `delete` anywhere in the
class:
```cpp
BoundedThreadPool *pool = new BoundedThreadPool;              // leaked
std::vector<cluster_event> *buffer[n_buffer];                 // each buffer[i] leaked
```
Since a fresh `Declusterer` is constructed on every `Roi`/`Electron` `.run()` call
(it's owned by the `CHEETAH`/`TIMEPIX` object, itself a local variable inside
`Roi::run()`), calling `decluster=True` repeatedly in the same process accumulated
one leaked thread pool and 128 leaked buffer vectors per call, with no bound on how
large those buffers' retained capacity could grow. Confirmed with a real repro: two
sequential `decluster=True` calls on a real ~1.3 GB file, each processing tens of
millions of events, crashed with `STATUS_ACCESS_VIOLATION` (Windows access
violation, exit code 139/3221225477) on the second call, even outside any Python/
Jupyter involvement (a plain `python script.py`).

**Fix:** both are now owned via `std::unique_ptr` (`std::unique_ptr<BoundedThreadPool>
pool`, `std::unique_ptr<std::vector<cluster_event>> buffer[n_buffer]`) -- RAII cleans
them up automatically when the owning `Declusterer` is destroyed at the end of each
`.run()` call. No custom destructor needed.

**Verified fixed:** 4 sequential `decluster=True` calls (varying `n_threads` and
`tot_per_electron`) in a single process on the real file, no crash.

**Measured, before vs. after** (process RSS after each sequential `decluster=True`
run, small 8x8 ROI so only ~3.5M events are decoded per run -- a real ROI leaks far
more per run):

| | baseline | run 1 | run 2 | run 3 | run 4 |
|---|---|---|---|---|---|
| before (leaking) | 32 MB | 101 MB | 170 MB | 239 MB | *(crashes eventually)* |
| after (RAII fix) | 32 MB | 33 MB | 33 MB | 33 MB | 33 MB |

Exactly +69 MB per run, perfectly linear, never released -- versus completely flat
after the fix.

**This also explains the Jupyter kernel crash.** Calling `Roi.run()` with
`decluster=True` from inside a Jupyter/`ipykernel` cell used to kill the kernel
outright (`DeadKernelError`) on the *first* call, while the identical code in a plain
`python script.py` survived the first call and only died on the second. That
difference was never ipykernel-specific behavior -- it's just that a Jupyter kernel
process starts with a much larger baseline footprint (IPython + numpy + matplotlib
already loaded) than a lean script process, so the same per-run leak crossed the
limit one run sooner there. With the leak fixed, calling `.run()` directly inside a
notebook cell works normally. Verified in a real `ipykernel` (via `nbclient`), with
`faulthandler` armed to a file so any fatal signal would leave evidence, across four
cases -- all pass, no fatal signal:

| case | result |
|---|---|
| `decluster=True`, small ROI, `n_threads=1` (the exact case that used to kill the kernel) | passes |
| `decluster=False`, small ROI (control) | passes |
| `decluster=True`, small ROI, `n_threads=8` (parallel path) | passes |
| `decluster=True`, ROI (207,231,30,30), `n_threads=8` (the case that killed the real notebook) | passes |

### 2.2 A genuine data race in the shared `BoundedThreadPool`

**File:** `EvenTem/src/utils/BoundedThreadPool.hpp`

This class (a generic bounded task queue + worker pool, used elsewhere for `Ricom`/
`FourD`'s per-line convolution offload) used **two different mutexes** to guard the
same `std::queue<std::function<void()>> tasks`: `mtx_queue_full` for `push_task()`,
`mtx_queue_empty` for the worker's pop/front-access. A push and a pop could run
concurrently on the same underlying queue -- a textbook data race. It went unnoticed
because nothing in this codebase had ever actually run the pool with more than one
worker thread before (`n_threads` was accepted by `Declusterer::init()` but the line
that would have started the pool was commented out).

Enabling real multi-threaded declustering (see below) surfaced it immediately: a
worker thread would occasionally dequeue a corrupted/empty `std::function`, throwing
`std::bad_function_call` (visible as a stray `bad function call` line in stdout,
caught and logged by the pool's own exception handler), and in the worst case a task
could be silently lost entirely -- which is what caused the "every cluster resolves
to 0 electrons" scenario (extreme `tot_per_electron`, `Roi`'s callback always taking
its early-return path) to hang or crash with a larger ROI: a lost task meant its
buffer's "declustered" counter never incremented, so the code waiting for "all
submitted buffers are done" spun forever, or corrupted state elsewhere led to a
second `STATUS_ACCESS_VIOLATION`.

**Fix:** a single mutex now guards the whole queue (plus a `busy_workers` counter, so
`wait_for_completion()` also stops returning prematurely while a dequeued task is
still executing -- a second, smaller pre-existing correctness gap in the same class,
fixed as part of the same change since it's the same lock).

**Verified fixed:** the extreme-`tot_per_electron` case (previously hung/crashed)
now completes normally with the large ROI, at `n_threads` up to 8, with correct
(all-zero) output.

**Note for anyone using `BoundedThreadPool` elsewhere** (`Ricom`/`FourD`): this fix
only changes internal synchronization, not the public API (`push_task`,
`wait_for_completion`, `init`, constructors all have identical signatures/semantics,
just now actually correct under real concurrency) -- no call-site changes needed
there, but code that assumed the old racy behavior was "probably fine because nothing
uses >1 thread" no longer needs that assumption to hold.

---

## 3. Declustering is now genuinely parallel

**Files:** `EvenTem/src/utils/Declusterer.hpp`, `EvenTem/src/detectors/Timepix.hpp`
(`enable_roi_declustered`), `EvenTem/src/core/Roi.cpp`.

Declustering was previously always single-threaded: `Roi.n_threads` /
`Electron.n_threads` were accepted parameters but never actually used to run more
than one worker (`Declusterer::init()`'s pool-start line was commented out). Ring
buffer slots are fully independent of each other for clustering purposes (each
slot's algorithm only ever looks within its own buffer), so they're safe to
decluster concurrently. Now:
- `Declusterer::init()` actually constructs the thread pool with the requested
  `n_threads` (`std::thread::hardware_concurrency()` if `< 1`).
- `schedule_declustering()` submits each buffer as a task to the pool instead of
  processing it synchronously inline; a new `n_buffer_submitted` counter (separate
  from `n_buffer_declustered`, which now means "truly finished", incremented at the
  real end of `decluster()`) tracks how many buffers have been handed off.
- Since buffers can now finish out of order (buffer 5 might complete before buffer 3
  if they land on different threads with different workloads), a new
  `std::atomic<bool> buffer_ready[n_buffer]` flag per ring slot lets
  `schedule_writing()` still emit buffers in their original order.
- The three histograms (`clustersize_histogram`, `energy_histogram`,
  `clustersize_tot_histogram`) are accumulated into thread-local buffers during each
  buffer's clustering (no locking needed there, since that's the expensive part) and
  merged into the shared histograms under a mutex in one short critical section per
  buffer -- not held during the actual O(cluster_range) windowed grouping loop.
- `Roi::run()` (CHEETAH branch) now passes its own `n_threads` (a `Roi`/
  `LiveProcessor` field that already existed and defaults to 1) through to
  `enable_roi_declustered`, instead of a hardcoded `1`.

**Default behavior is unchanged**: `n_threads` defaults to 1, which still runs
through the same (now-fixed) thread-pool machinery with a single worker -- functionally
equivalent to the old serial behavior, just with a small amount of queue/lock
overhead. Set `Roi.n_threads > 1` (or `0` for "use all available cores") to get real
parallelism.

**Measured speedup** (real ~1.3 GB file, ROI `(207,231,30,30)`, ~99.5M raw hits in
scope): 121.8s at `n_threads=1` (matches pre-change timing) -> 35.2s at `n_threads=4`
(3.46x) -> 21.6s at `n_threads=8` (5.63x). All three produce byte-for-byte identical
`Roi_scan_image` / `Roi_diffraction_pattern` / `clustersize_tot_histogram` output --
parallelism doesn't change results, only wall time.

A full-scan run (the entire file, all 512 scan lines -- see the note in section 5
about `finish_line`) completed in **39 seconds at `n_threads=8`**, processing all
169,536,281 raw events into 76,357,116 resolved clusters.

---

## 4. Correctness verification performed this session

There's no automated test suite in this project. Verification was done empirically
against the real example file (`raw_0643_-40.00_000000.tpx3`, ~1.3 GB, 200 keV,
512x512 scan):
- **Regression**: `decluster=False` (default) output is byte-for-byte identical to a
  baseline captured before any of these changes, on both a small and a large ROI.
- **Cross-check**: `Electron`'s existing `.electron`-conversion `decluster=True` path
  (completely unmodified code) was run on the same file before and after these
  changes; its `energy_histogram` output is identical, confirming the shared
  `Declusterer` changes didn't alter its behavior.
- **Parallel correctness**: `n_threads` in {1, 4, 8} on the same ROI/file produce
  identical `Roi_scan_image`, `Roi_diffraction_pattern`, and
  `clustersize_tot_histogram` -- multithreading changes speed, not results.
- **Both crash repros** (double `decluster=True` in one process; extreme
  `tot_per_electron` with a large ROI) now complete without error.
- **Joint histogram internal consistency**: `clustersize_tot_histogram.sum()` equals
  `clustersize_histogram.sum()` (both count real clusters); the well-known
  `energy_histogram[0]` discrepancy is fully explained (see the field table above).

---

## 5. Things intentionally left alone / not fixed

- **`energy_histogram`'s bin-0 semantics** (conflates merged-away hits with real
  zero-ToT clusters) is pre-existing behavior, not touched -- `clustersize_tot_histogram`
  is the fix/workaround, added alongside rather than changing existing behavior.
- **`Roi`'s `finish_line` early-termination** (a partial ROI stops decoding once the
  scan has passed the ROI's vertical extent) is pre-existing, unrelated to this
  session's changes -- but it means a partial ROI's declustering histograms only
  reflect the file up to that point, not the true end. Only a full-width/full-height
  ROI (`set_roi(0, 0, nx, ny)`) decodes the entire file. Not a bug, just worth knowing
  when interpreting histogram statistics.
- **The Jupyter `DeadKernelError` is resolved** -- it was the same memory leak as
  section 2.1, not a separate ipykernel problem (see the measurement there). Calling
  `.run()` directly inside a notebook cell now works. `test/run_decluster.py` still
  exists and is still useful for long full-file runs from a terminal, but it's no
  longer a required workaround.
- One genuinely separate Jupyter issue *was* found and fixed along the way, unrelated
  to any of the above: a `%matplotlib qt` cell kills the kernel when there's no GUI
  display session available (headless/remote/sandboxed). Use `%matplotlib inline`.
- Declustering still only parallelizes *across* ring-buffer slots, not *within* one
  slot's O(cluster_range) windowed grouping loop -- doing the latter would require a
  more invasive algorithmic change (splitting one buffer's sequential-dependency
  algorithm across threads with boundary handling) that wasn't attempted.
- Scope was kept to `Roi` + CHEETAH; `Ricom`'s and the GPU/LibTorch path's own
  behavior were not modified (though the `BoundedThreadPool` race fix in section 2.2
  applies to `Ricom`/`FourD` too, since they use the same shared class).

---

## 6. Where to look / how to reproduce

- `EvenTem/src/utils/ClusterResolver.hpp`, `Declusterer.hpp`, `BoundedThreadPool.hpp`,
  `Roi4D.hpp`; `EvenTem/src/detectors/Timepix.hpp`; `EvenTem/src/core/Roi.h`/`.cpp`;
  `EvenTem/src/pybind.cpp`; `EvenTem/eventem.py` -- all the actual changes.
- `test/decluster.ipynb` -- walkthrough notebook (loads pre-computed results; see its
  own notes for why it doesn't call `.run()` directly).
- `test/run_decluster.py` -- standalone script that does the real processing for a
  configurable ROI/`tot_per_electron`, run from a terminal.
- `test/run_full_file.py` -- same, but for the entire scan (all 512 lines), used to
  produce the whole-file calibration histogram.
- `test/results/*.npy` -- saved outputs from both scripts above.
