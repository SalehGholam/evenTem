# EvenTem work session: overview

A narrative summary of everything covered in this engagement -- what was built, what was
investigated but not changed, what broke and got fixed, and what's still open. For the
precise API/file-level diff, see `CHANGES.md`; this document is about the
*process* -- what turned out to be useful, what didn't, and why.

Scope throughout: the CHEETAH (`.tpx3`) detector path, `Roi`/`vSTEM`/`Var`/`FourD`/`Electron`
reconstruction modes. `Ricom`, the GPU/LibTorch path, and the Merlin/Advapix/Numpy backends
were not touched.

All verification happened in live Jupyter notebooks (`test/*.ipynb`) that call `.run()`
directly against real detector data -- never standalone scripts, per a standing rule for
this project. Every C++ change went through the same loop: edit -> rebuild -> assemble a
new `eventem_py312_decluster_build_vN` -> re-run the relevant notebook(s) -> read the
actual numbers back, not just "it compiled."

---

## 1. Charge-weighted declustering (`Roi`, then `vSTEM`/`Var`)

**What it does.** A single physical electron hitting the Timepix3 sensor usually lights up
several adjacent pixels (charge sharing); without correction every one of those pixels
counts as a separate electron. Declustering groups nearby hits into clusters, resolves each
cluster's ToT-weighted centroid, and credits `round(cluster_ToT_sum / TotPerElectron)`
electrons at that position instead of `+1` per raw hit.

**Useful, shipped, verified.** Built for `Roi` first (`Decluster`/`Dtime`/`Dspace`/
`ClusterRange`/`TotPerElectron` + two calibration histograms), then extended identically to
`vSTEM` and `Var` by reusing the exact same `Declusterer` machinery -- no new parallelism
work needed, since thread-pool declustering was already built once and verified
deterministic (`n_threads=1` vs `8` byte-for-byte identical) for `Roi`, and inherited for
free by the other two.

**Real bugs found and fixed along the way** (all in `Declusterer.hpp`, all pre-existing, not
introduced by this session):
- A raw thread pool and 128 raw buffer vectors were leaked on *every* `.run()` call
  (measured +69 MB/run) -- this is what was killing Jupyter kernels after a few runs and
  segfaulting a second `decluster=True` call in one process. Fixed with RAII.
- A genuine data race in the shared thread pool (two different mutexes guarding one task
  queue) that could corrupt or silently drop a task -- unnoticed until multi-threaded
  declustering was exercised for the first time. Fixed with a single mutex.
- `Roi4D`'s storage was a 4-level nested `vector<vector<vector<vector<T>>>>` -- millions of
  tiny heap allocations for a full-frame ROI, plus a double-copy on every `get_4D()` call.
  Replaced with one flat contiguous buffer; verified byte-for-byte identical output.

**Not pursued (explicit scope decision, not a bug):** `Roi.roi_mask` and `vSTEM`'s
multi-annulus/mask variants were not extended to support declustering, matching how the
base feature itself only covers the primary ROI/vSTEM path.

---

## 2. Interactive plotting backend

Small, quick ask: inline matplotlib plots in `decluster.ipynb` couldn't be zoomed or
hovered for values. Installed `ipympl`, switched to `%matplotlib widget`. Useful, no
complications.

---

## 3. Architecture/performance discussion (no code changes)

An exploratory conversation about whether the ROI-extraction pipeline was "optimized" --
trigger search strategy, parallelization opportunities, GPU/Rust rewrite feasibility. This
stayed a discussion, not a task: no code was changed as a result. Two things worth
recording:
- I initially claimed the ToA-overflow-watch thread looked redundant. The user correctly
  recalled that the full ToA range resets every ~20-30 seconds during real acquisitions,
  which is exactly what that thread exists to catch -- verified the arithmetic
  (2^34 ticks x 1.5625 ns ~= 26.8 s) and retracted the claim. **Lesson:** don't second-guess
  hardware-timing-driven design without checking the actual numbers first.
- GPU/Rust rewrite was discussed as a large, separate undertaking and intentionally not
  pursued in this session.

---

## 4. `FourD` (event -> 4D array) conversion overhaul

The largest single piece of work. Three stated problems: slow reads, wrong chunk shape for
the real access pattern (small ROI across a few lines, full diffraction pattern per point),
and a request to add declustering there too.

### 4a. Chunk shape fix -- useful, shipped
Added `ChunkSizeX` (independent scan-x chunk width; previously implicitly the *entire* scan
row, meaning any small-ROI read had to touch full-width chunks regardless of how narrow the
actual query was). Along the way, found and fixed two real, independent, pre-existing bugs
while restructuring the exact code that needed touching anyway:
- The `"shape"` auxiliary dataset and the real `"4D"` dataset disagreed on axis order
  (`{nx,ny,...}` vs `{ny,nx,...}`) -- silent for every square scan tested so far, would
  mislabel a non-square one. Only found by deliberately testing with `nx != ny`.
- The `"shape"` dataset's HDF5 write used an 8-byte buffer type (`hsize_t[4]`) declared as a
  2-byte predtype (`NATIVE_UINT16`) -- corrupted 3 of the 4 shape values on read-back. Fixed
  for the 32-bit and 16-bit writers at the time; **the 8-bit writer had the identical bug and
  was missed** -- caught and fixed later, only because a different task (`SaveMetadata`)
  happened to touch that exact function again.

### 4b. Zarr as an additive output format -- useful, shipped
`FourD.Format = "hdf5"` (default) or `"zarr"`. Hand-rolled a minimal, spec-compliant Zarr v2
writer using zlib rather than pulling in a full C++ Zarr library or Blosc, specifically to
avoid a second heavy dependency chain on top of an already fragile HDF5 build. Required
installing zlib via vcpkg (there was none on the machine -- confirmed by an actual failed
link error, not assumed). Verified: a plain `zarr.open()` (no custom reader) and
`dask.array.from_zarr` both work, and HDF5 vs Zarr outputs of the same run are byte-for-byte
identical.

### 4c. Declustered `FourD` -- useful, shipped, but found a real crash first
The first attempt crashed the Python kernel partway through a full-scan, 8-thread run. Root
cause: the new declustered chunk-flush callback indexed the output buffer using a cluster's
row with no bounds check -- a cluster landing in an already-flushed-and-recycled chunk group
(rare non-monotonic ordering right at a group boundary) produced a negative offset that
wrapped to a huge unsigned index. Fixed by dropping such clusters (same treatment as an
already-unresolved cluster) instead of writing through them. Re-verified clean: cross-check
against `Roi`'s independently-verified declustered output (exact match), determinism at
`n_threads=1` vs `8` (byte-for-byte identical), and a per-chunk-group scan for
dropped/duplicated regions (none found). A small, expected residual remained
(`Dose_image` slightly higher than the on-disk array, ~0.006%) -- this is the fix working as
designed, not a new bug.

### 4d. Dwell-time unit bug -- real, shared, high-value fix
Found via the user's own domain knowledge, not by me: *"the first trigger is read at the
end of the first line; with respect to the dwell time and number of pixels in that line, the
events are then counted."* This pointed straight at a real bug: `CHEETAH`'s constructor
treated its fallback `dt` parameter as nanoseconds, while every Python-facing caller across
the whole codebase (`Roi.dt = 100` etc.) passes microseconds -- a 1000x error, but *only* in
the fallback value used for scan line 1, before the real hardware-calibrated `dt` is derived
from that line's own TDC pulse. Since this fallback lives in shared code (`Cheetah.hpp`), the
bug affected `Roi`/`vSTEM`/`Var`/`FourD`/`Electron` identically. Fixed at the source; recovered
41,347 previously-lost events per run in testing.

### 4e. `Roi.get_4D()` shape investigation -- no bug, twice
Two rounds of "why is `get_4D()` returning a 30x30 frame instead of the full detector?" and
"when did I set a binning?!" Both resolved as *not bugs*: `det_bin=32` was a constant I had
set myself in an earlier demo notebook, not a `Roi` default (the real default is
`det_bin=1`), and the "30x30" was the ROI's scan extent, not a detector crop -- a single
scan point's diffraction pattern is always the full binned detector frame. Time well spent
confirming this precisely (with a dedicated exact-match test) rather than leaving it
ambiguous, since it recurred twice.

---

## 5. Smart-scan / pixel-triggered custom pattern support (`set_pattern_file`)

**Starting point, useful to know:** this wasn't built from scratch. `CHEETAH_pixeltrig` (a
class that resolves scan position directly from a linearized pattern file instead of
ToA/dwell-time) already existed in the codebase and was already wired up for `Roi`/`vSTEM`/
`tcBF`/`Pacbed`. The actual work was auditing what existed, extending it to `Var`/`FourD`,
and fixing what was broken.

**Verified working, useful:**
- `Roi`, `vSTEM` -- already wired, confirmed correct against the real smart-scan file.
- `Var` -- newly wired up (was entirely missing), verified.
- `FourD` with `decluster=True` -- newly wired up, verified via a rigorous per-position
  completeness check (0.1% residual, matching the same small accepted class as 4c above).
- Declustering + a pattern file was silently broken everywhere (ToT always read as `0`,
  because the pixel-trigger event parser had no ToT-aware counterpart to the raster path's)
  -- every cluster would have resolved to 0 electrons. Fixed by mirroring the raster path's
  ToT-aware dispatch.

**Found, NOT fixed, and blocked rather than shipped broken:** `FourD` with
`decluster=False` (the raw path) combined with a pattern file undercounts by ~21% of scan
positions (~7% of total events), with two whole chunk-groups coming back completely empty in
one test. Two different flush-timing fixes were tried (tracking the current event's own row,
then the minimum row across all 4 detector-chip quadrants) with **no meaningful
improvement** -- the mismatch count was identical before and after the second fix. Best
current understanding: raw `.tpx3` files store the 4 chip quadrants in chip-blocked sections
rather than time-interleaved event-by-event, so a purely-sequential-in-file-order flush
tracker never sees the smooth per-row progression it assumes, regardless of how it accounts
for cross-chip timing. Rather than ship this silently wrong, added a runtime guard: `FourD`
with `decluster=False` and a pattern file now raises immediately, pointing at the two
alternatives that *are* verified correct (`decluster=True`, or `Roi` with `extract_4D=True`).
**This is the one substantial piece of this session left genuinely unresolved** -- a real
architectural fix would need to understand the actual chip-interleaving structure of raw
`.tpx3` files, which wasn't reverse-engineered here.

---

## 6. End-of-scan wraparound bug -- real, fixed, closed an old open question

While chasing the pixel-trigger bug above, went back to re-examine a small residual
mismatch (132-3262 events, deferred as low-priority back in section 4d) using fresh,
targeted debug instrumentation rather than more guessing. Traced it precisely: a faster
detector chip's own line count could race past the intended scan length
(`ny * repetitions`) *before* the aggregate stop condition (which waits for the *slowest*
chip) caught up -- during that window, the faster chip's trailing/phantom line wrapped via
modulo back onto "scan row 0," contaminating it with events that belonged to a line beyond
the scan's real end. Fixed by checking each chip's own bound individually, not just the
aggregate one. Confirmed with a fresh full-scale (512x512) test: **zero mismatches across
all 262,144 positions**, where the same test previously showed up to 3,262 wrong events at
position (0,0). This fully resolves what had been an open, deferred item since section 4d.

**Methodology note, useful in hindsight:** the original small-scale reproduction of this bug
used an artificially shrunk scan (64x32) for speed, on a raw file whose real geometry is
512x512. That mismatch between test config and real hardware geometry produced its own,
*different* spurious artifact (modulo-wraparound from re-using row indices every 32 real
lines) that looked similar but had nothing to do with the real bug. Debugging this properly
required re-running at the *real* scan size, not the fast shrunk one -- a reminder that
speed-motivated test shortcuts can manufacture their own bugs.

---

## 7. `Roi.tot_mode` -- new feature

Sums each raw hit's own ToT into the scan image / diffraction pattern / 4D cube, instead of
counting hits or resolving clusters -- "total deposited charge per probe," no clustering
involved, deliberately independent of and mutually exclusive with `decluster`. Mechanically
this filled in existing-but-dead scaffolding: a ToT-aware dispatch path (`roi_ToT()`) already
existed in the codebase but was unreachable under any shipped feature (nothing had ever set
`b_tot=true` with the plain `roi` functionType). Fixed an inconsistency in that dead code
while enabling it (`roi_ToT()` was still doing `+1` on the scan image while summing ToT into
the diffraction pattern) and added the missing 4D-cube counterpart (`roi_4D_ToT()`, didn't
exist at all before). Verified: internally consistent (scan image / diffraction pattern / 4D
cube agree exactly), and correctly raises when combined with `decluster`.

---

## 8. `FourD.SaveMetadata` + making the output loadable by HyperSpy

**The ask:** write only the `"4D"` array (no `"dose_image"`/`"shape"` siblings), so the
output can be loaded via `hyperspy.api.load(fn, lazy=True)`.

**`SaveMetadata=False` itself:** straightforward, shipped, verified for both HDF5 and Zarr.

**The `hs.load()` part -- got this wrong once, then found the real answer.** First attempt:
concluded "no amount of dataset-pruning makes `hs.load()` work" after `hs.load()` raised an
ambiguous-multiple-readers error and even explicitly naming the `HSPY` reader rejected the
file outright ("not a valid HyperSpy hdf5 file"). That conclusion was **premature** -- when
pushed on it, actually inspecting HyperSpy's own writer output (round-tripping a real signal
through `s.save()` and reading the resulting file byte-for-byte with `h5py`, rather than
guessing from error messages) showed the real, minimal schema is small: two root attributes,
one `Experiments/<name>` group, one `axis-i` group per array dimension, and three small
placeholder groups. Separately, the earlier "ambiguous reader" error turned out to be about
the `.hdf5` *file extension* being claimed by multiple unrelated plugins, not about file
validity at all -- `.hspy` is HyperSpy's own unambiguous extension. Added
`eventem.make_hyperspy_compatible(src_path, dst_path)` to `eventem.py`, using
`h5py.Group.move()` so the array is relocated in place with no data copy. Verified:
`hs.load(fn, lazy=True)` now works with zero extra keyword arguments, correct shape/dtype,
lazy dask array, and matching totals.

**Lesson:** "I tried the obvious thing and it failed" is not the same as "this is
impossible" -- worth distinguishing before reporting a limitation as final, especially when
the failure mode (an error message) hadn't actually been read carefully enough to see it was
about file extension dispatch, not content.

---

## 9. Navigation image

A quick visualization request at the end: plotted `dose_image` (the per-scan-position
electron count, i.e. the "navigation image" in HyperSpy terms) from the smart-scan
declustered `FourD` run. Confirmed visually what the numbers already said -- only 15.6% of
scan positions were ever triggered, concentrated in a scatter of irregular blob-shaped
regions rather than a uniform raster, consistent with an adaptive/feature-driven scan.

---

## What was useful (recap)

- Declustering for `Roi`/`vSTEM`/`Var`/`FourD`, with free parallelism from one shared,
  verified-deterministic thread pool.
- The `FourD` rewrite: independent scan-x chunking, Zarr output, and declustered
  conversion -- all verified against real data, not just unit-shaped test cases.
- The dwell-time and end-of-scan wraparound fixes -- both root-caused precisely (one from
  user domain knowledge, one from targeted debug instrumentation) rather than patched
  around, and both shared/affect every reconstruction mode identically.
- `Roi.tot_mode` and `FourD.SaveMetadata` -- both small, clean, additive features.
- `make_hyperspy_compatible` -- a real, working answer to "can this just load in HyperSpy,"
  found by inspecting HyperSpy's own output instead of stopping at the first error.
- Non-square and full-scale test cases specifically, repeatedly -- most of the real bugs in
  this session were invisible at `nx==ny` or at shrunk test scan sizes, and only appeared
  once tested at the real, asymmetric, full-size configuration.

## What wasn't useful, and why

- **The chip-desync "minimum across 4 chips" fix for raw pixel-triggered `FourD`.** Sound
  in principle (mirrors how raster mode already handles the same class of chip-timing
  issue), but measured to have essentially zero effect on the actual bug. Kept in the code
  (not strictly worse than the simpler version it replaced) but the underlying problem is
  still open -- a case where a plausible-sounding fix didn't survive contact with real data,
  and the honest move was to say so and block the feature rather than claim it was fixed.
- **Two abstract hypotheses that were flatly wrong when tested:** a predicted "double-flush"
  bug in `FourD`'s original chunk-flush logic (the last chunk group was actually the
  *largest*, not zero -- the hypothesis had the mechanism backwards), and my own initial claim
  that dropping metadata datasets alone would make `hs.load()` work. Both were corrected only
  because they were actually tested/pushed on rather than left as accepted conclusions.
- **Artificially shrunk test scans for speed** (`nx=ny=64` on a 512x512 file) -- reasonable
  for quick smoke tests of logic, but actively misleading for anything involving chip
  synchronization or trigger-count wraparound, since shrinking the scan changes *when* those
  effects trigger, not just how long the test takes. Full-scale re-tests were what actually
  found and confirmed the real bugs in sections 4d, 5, and 6.
- **A `dt=0`/unset fallback path was found to crash the kernel** (not just lose the first
  line as its own printed warning claims) during testing of two unrelated features. This is
  a real robustness gap, but it only manifests as a user-error scenario (forgetting to set
  dwell time), and was deliberately left uninvestigated to stay focused -- noted here so it
  isn't lost.

## Explicitly out of scope / deferred by choice

- `Electron.h`'s pre-existing `decluster=true` default is inconsistent with this project's
  "off by default" convention everywhere else -- flagged, never changed (no explicit
  decision was ever made on it).
- Snake/zigzag scan pattern support -- confirmed not currently supported; no request was
  made to build it, so nothing changed.
- `roi_mask` / `enable_multi_vSTEM` / `enable_mask_vSTEM` declustering -- consistently left
  out of scope everywhere declustering was extended, matching the base feature's own
  boundary.
- The raw pixel-triggered `FourD` completeness bug (section 5) -- **fixed in a later session**
  (see "Multi-process vSTEM file-splitting" and "Raw pixel-trigger `FourD`, actually fixed"
  below). The runtime guard has been removed.

## Build reference

Each C++ change went through a full rebuild + reassembled binary
(`test/eventem_py312_decluster_build_vN/`) before being trusted. Roughly:

| Build | What it added |
|---|---|
| v5-v6 | Declustering for `Roi`, then `vSTEM`/`Var` |
| v7 | `FourD` chunk-shape fix, dwell-time fix, `"shape"` dataset fix |
| v8 | Zarr output format |
| v9 | Declustered `FourD` conversion (crashed on first test) |
| v10 | Fixed the declustered-`FourD` crash (bounds check on cluster row) |
| v11 | Smart-scan pixel-trigger support (`Var`/`FourD` wiring, ToT-extraction fix) + `FourD.SaveMetadata` |
| v12 | End-of-scan wraparound fix, `Roi.tot_mode` |
| v13 | First (insufficient) fix attempt for the raw pixel-trigger `FourD` bug |
| v14 | Runtime guard blocking the still-broken raw pixel-trigger `FourD` combination |
| v15-v22 | Multi-process `vSTEM` file-splitting: seek/checkpoint mechanism, `seed_dt`/`seed_rise_t`/`seed_rise_fall` (fixed the ~90% boundary-row undercount at N=4), `seed_line_count` (a second, smaller-magnitude fix attempt), authoritative post-`terminate()` re-copy (fixed the standard incremental-copy step discarding a worker's own boundary-row data) |
| v26-v27 | Cleaned-up multi-process build (diagnostics removed); confirmed byte-for-byte exact at `n_processes` in `{1, 2, 4}` for `decluster=False`; `n_processes=3` and `>=8` still show a small residual |
| v28-v33 | Diagnostic builds isolating the raw pixel-trigger `FourD` bug |
| v34 | **Raw pixel-trigger `FourD` fixed**: removed the runtime guard. Root cause was a second, redundant flush trigger (the regular raster-progress one in `FourD.cpp`, guarded only by `!decluster`) firing at the same time as the dedicated pixel-trigger flush mechanism, both driving the same shared `write_and_clean()` counter. Residual after the fix: 0.12% (same accepted-residual class as declustered `FourD`) |
| v35-v38 | Declustered `FourD` chunk-buffer fix (see "Declustered `FourD`'s chunk buffer, partially fixed" below): raster-mode residual eliminated entirely (0 mismatches, was already small but nonzero); pixel-trigger residual reduced ~46% (261 -> 141 mismatched positions) |
| v39 | Fixed `Roi.set_roi_mask()` for smart-scan data: `CHEETAH_pixeltrig`'s per-event dispatch switch was missing a `case` for `roi_mask` entirely (present for `roi`/`roi_4D`), so a mask on pixel-trigger data silently produced all zeros -- no error, no crash. Added the missing case; verified exact against an independent Python ground truth. |
| v40-v43 | Smart-scan performance: confirmed via controlled A/B benchmark that pixel-trigger mode really was ~2.4x slower per event than raster mode on the same operation (despite processing FEWER real events -- not explained by data volume). Fixed 3 real inefficiencies in `CHEETAH_pixeltrig`: `which_type()` checked header/TDC before event (the overwhelming majority case) instead of event first, unlike the raster path; `n_events_processed` was incremented twice per event (cosmetic -- wrong progress-bar rate, not a real slowdown, but a real bug); the same trigger-count modulo was computed twice per event. These closed the gap from ~2.4x to ~2.0x. Tested and ruled out two more hypotheses (pattern array memory footprint, the pattern lookup itself) via direct experiment -- the remaining ~1.7-2x gap is not yet root-caused. |

A full source/build backup was taken before the smart-scan work started:
`backups/pre_smart_scan_pattern_20260911_170939/`.
A full source/build backup was taken before the multiprocessing work started:
`backups/pre_multiprocessing_20260911_213003/`.

## Multi-process `vSTEM` file-splitting (later session)

**What it does.** Splits a `.tpx3` file into N byte ranges (via a fast, TDC-only
pre-pass, `vSTEM.find_checkpoints(n)`), runs each range in its own OS process, and
sums the partial `vSTEM_image` arrays. Targets the one part of the pipeline with zero
built-in parallelism: raw decode is single-threaded regardless of any `n_threads`
setting (only declustering has its own thread pool).

**Three real, distinct bugs found and fixed** (each confirmed by direct instrumentation
after two theory-driven fixes in a row changed nothing):
1. A worker resuming mid-file force-resets all 4 Timepix chips' `rise_fall` state to
   "not yet risen" -- correct for a true start, wrong for 3 of the 4 chips that are, by
   construction, already mid-line at an interior checkpoint. Fixed with `seed_rise_fall`.
2. The standard incremental output-copy step only commits a row once decode has
   advanced *past* it -- but a worker's decode deliberately stops *at* its own boundary
   row, so that row's real data (already written into the staging buffer) never gets
   copied out. Fixed by extending the existing decluster-mode authoritative re-copy
   (zero the output, re-sum the staging buffer once decode is fully done) to also cover
   split workers.
3. (Partial) some chips can be 2+ lines ahead of the slowest one at a checkpoint, not
   just 1 -- `seed_line_count` seeds each chip's own real line count individually
   instead of a uniform value.

**Useful, verified**: exact match at `n_processes` in `{1, 2, 4}`; tunable process count
(`mp_vstem_runner.run_vstem_parallel(..., n_processes=N)`); real speedup on a cold file
cache (~15x measured on this session's test file), none once the OS has it cached
(single-threaded decode alone already hits ~50M events/s).

**Not chased further, by explicit user direction** (the test machine has 8 physical
cores -- pushing split counts where the hardware can't benefit anyway isn't worth more
debugging time): `n_processes=3` and `n_processes>=8` each show a small, boundary-
localized residual (a few hundred pixels out of 262144) not yet root-caused.
**Decluster + multi-process splitting is actively worse, not just unverified**: declustering
already has its own internal thread pool that scales well in a single process; splitting
into multiple processes divides that same thread budget across workers, so total
declustering parallelism goes *down* as `n_processes` goes up. Measured strictly slower
at every `n_processes > 1`, and not byte-for-byte exact (a cluster straddling a worker's
byte-range boundary can be resolved incompletely by two independent `Declusterer`
instances -- a different, unresolved correctness gap from the raw case). Recommendation:
`n_processes=1` with `n_threads_per_worker` set as high as the core count allows.

## Multi-process `vSTEM` file-splitting, pixel-trigger (smart-scan) mode -- fully working (later session)

**Motivation.** Same raw-decode-has-zero-parallelism argument as the raster case above,
plus a genuine architectural advantage for smart-scan: pixel-trigger's position lookup
(`pattern[probe_count_chip[chip_id]]`) is a direct per-chip trigger-count index with no
dwell-time (`dt`/`rise_t`) calibration warm-up at all, unlike raster -- so the whole
`seed_dt`/`seed_rise_t` warm-up bug class raster needed doesn't apply here.

**What was built**: `CHEETAH_pixeltrig::find_trigger_checkpoints(n_splits)` (mirrors
`Cheetah.hpp::find_line_checkpoints`, but splits evenly by raw trigger count against
`pattern.size()*repetitions` -- `current_line` (`=probe_count/nx`) is *not* a usable
progress proxy here, confirmed by direct measurement to stay far below `ny` for a sparse
pattern); a real stop-at-boundary mechanism for pixel-trigger (`process_buffer()`/
`process_tdc()` previously had *no* early-stop path at all -- decode always ran to true
EOF); `seed_probe_count_chip`/`seed_rise_fall` to resume a worker's 4 chips correctly at
an interior checkpoint; `vSTEM::find_checkpoints_pixeltrig()` and pybind exposure.

**A subtle trap found and fixed along the way**: the first version of the new stop check
used `pattern.size()*repetitions` as a fallback trigger-count total *even when no split
was requested* (`stop_at_line<0`), reasoning it should be harmless since that's
approximately the file's real total anyway. It was not harmless -- it reintroduced a full
hang for the plain, non-split case. Root cause: `read_file()`'s loop has no real
end-of-file detection; the *existing*, pre-this-session pixel-trigger code already
depended on reading past the file's true end (into whatever `read_data()` returns at
EOF) to drive `current_line` far enough for `vSTEM`'s incremental output-copy sweep to
reach its `fr_total` and terminate -- confirmed directly (`reached_pp_id` showed the
sweep legitimately reaching within one row of the full `nx*ny` address space on an
unmodified whole-file run, which is only possible via that past-EOF behavior, since
real trigger count is far too small to explain it otherwise). Stopping even slightly
early via the new fallback broke that. Fixed by only ever setting `repetitions_reached`
via this path when `stop_at_line>=0` (a real, deliberate split boundary) -- the
non-split path keeps relying on the old (pre-existing, unrelated to this work)
past-EOF quirk, unchanged.

A second attempted fix, made when real stopping first worked, turned out to be the actual
root cause of the (much larger, ~1-6.6%) data loss measured next: `vSTEM::reset()`'s
`fr_total` formula (`nx*stop_at_line`) is raster-only -- for pixel-trigger, `stop_at_line`
is a trigger count, not a row, and `current_line` (`=probe_count/nx`, integer division)
only advances in whole nx-sized buckets, so it cannot represent a single exact trigger.
Flooring it (`nx*(stop_at_line/nx)`, chosen to guarantee the sweep could still terminate)
let `vSTEM::process_data()`'s own incremental-copy sweep -- running on a *different*
thread than decode -- reach "done" and force-quit both decode threads
(`*p_processor_line=-1`) *before* `CHEETAH_pixeltrig::process_tdc()`'s own exact
`probe_count>=stop_at_line` check ever got to fire. Confirmed directly with a
systematic sweep (`investigate_pixeltrig_stop_loss.ipynb`): for *every* stop point tested
(5000 through the full 40856), a contiguous range of 135-450 triggers immediately
preceding that exact point went missing -- present even for `stop_at_line=40856` (a
"stop" at what should be the natural end), and the exact width varied slightly run to
run, confirming two threads racing to be the one that stops decode first, with the
coarser (nx-bucket) one usually winning.

**Real fix**: stop relying on that sweep for split-worker termination at all.
`CHEETAH_pixeltrig::process_tdc()` now writes `*p_processor_line=-1` itself, directly,
the instant `probe_count` exactly reaches `stop_at_line` -- the same pointer both decode
threads and `vSTEM::process_data()`'s own while-loop already key off, just signaled
precisely instead of via a lossy nx-wide proxy. `vSTEM::reset()` now leaves `fr_total` at
the ordinary whole-file value for `CHEETAH_PIXELTRIG` unconditionally, which guarantees
its sweep can never reach "done" on its own for a sparse pattern (confirmed: `current_line`
stays far below `ny`) -- so termination comes entirely from the camera's direct signal,
never from a race.

One follow-on trap: this closed the loss down to a tiny (0.002%-0.08%) *excess* at the
exact boundary trigger (a few of a faster chip's hits for the shared boundary dwell
slipping through right as the stop signal lands). The obvious-looking fix -- add a guard
dropping any credit with `pattern_idx >= stop_at_line` -- was **tried and reverted**: it's
based on a misreading of the data. That "excess," measured with a single isolated worker
and no real neighbor, is not a double-count risk -- it's a *legitimate* partial
contribution to a pixel that the boundary happens to land in the middle of. The next
worker (starting at exactly the same byte position) picks up the rest of that same
pixel's hits from other chips still mid-dwell there. Since the two workers' byte ranges
are strictly complementary (no gap, no overlap), summing their contributions reconstructs
the true per-pixel total exactly -- adding the guard only made one side of that split
throw its legitimate share away, turning the small excess into an equally small loss at
the same spot. Confirmed by reverting it: the "excess" and the "loss" were numerically
identical (1425 events) for the one boundary tested.

A second bug was found and fixed the same way, since 4-way and 8-way splits still weren't
exact after the fix above (2-way was, by luck -- see below): `CHEETAH_pixeltrig::chip_id`
had no default member initializer (`int chip_id;`), unlike `Cheetah.hpp`'s raster
counterpart (`int chip_id = 0;`). `chip_id` is set only by header packets in the raw byte
stream, and a header can be *arbitrarily far* from any given checkpoint -- confirmed
directly by manually decoding the raw bytes at a real checkpoint's byte offset: 40+
consecutive TDC/event packets with zero header packets in between. A worker resuming
there has no way to know which of the 4 chips its first packets belong to until its own
first header eventually arrives, and an uninitialized `chip_id` used to index
`rise_fall[chip_id]`/`probe_count_chip[chip_id]` (both size 4) in the meantime is
undefined behavior -- usually just misattributes a handful of early hits to the wrong
chip (explaining the ~10,000-position, mostly-canceling error spread across nearly an
entire worker's range that an N=4 split showed), but if the garbage value happens to be
>=4, it's an out-of-bounds write (reproduced directly: adding an unrelated diagnostic
member field nearby was enough to turn the miscredit into a crash, by shifting what
memory the overrun landed on).

**Why 2-way "worked" despite this bug being present all along**: pure luck. With only one
boundary to get right, its checkpoint happened to need chip_id=0 -- the same value the
uninitialized field happened to already hold. Every additional split boundary is another
independent roll of that dice, which is exactly why 4-way and 8-way kept failing while
2-way kept passing.

**Real fix**: `find_trigger_checkpoints()` now also captures `chip_id` at each checkpoint
(5th tuple element) and returns it; a new `seed_chip_id` field (Timepix.hpp, -1 = unseeded
= old default of 0) seeds it back into a resuming worker's `reset()`. `vSTEM.h`/`vSTEM.cpp`/
`pybind.cpp` expose the same field on `find_checkpoints_pixeltrig()`/`vSTEM.seed_chip_id`.

**Status: fully resolved and verified.** Byte-for-byte exact against a single-process
baseline at n_processes in {1, 2, 4, 8, 16, 32} on this session's
40856-trigger/262144-position test file -- both in the same-process sequential-worker
test harness and in the real `mp_vstem_runner.run_vstem_parallel_pixeltrig()` path using
actual separate OS processes (`ProcessPoolExecutor`). `vstem_mp_worker.run_slice_pixeltrig()`
and `mp_vstem_runner.run_vstem_parallel_pixeltrig()` both thread `seed_chip_id` through
correctly. As with the raster case earlier in this document, no real wall-clock speedup
was observed on this specific (small, OS-cache-warm) test file -- CPU-bound single-threaded
decode doesn't benefit from splitting the way disk I/O does, and the checkpoint pre-pass's
own cost (0.7-1.1s here) can outweigh the savings entirely on a small/cached file. The
benefit, if any, would need to be measured on a large, not-recently-touched file.

**Follow-up: measured cache-cold and disk-type effects directly (later session).**
No admin rights/Sysinternals tools available, so the OS file cache was force-evicted by
temporarily allocating and touching ~22GB of RAM (standard non-privileged technique --
Windows reclaims standby/cached pages under real memory pressure), re-run before each
individual configuration to keep every measurement genuinely cold. Tested on both a real
spinning HDD (Seagate Barracuda) and a real NVMe SSD (Samsung 990 Pro), same exact file
on both. Splitting **never won, on any of the 4 combinations tested (HDD/SSD x warm/cold)
x n_processes in {1,2,4,8,16}** -- total time climbed monotonically with n_processes in
every case. Two different reasons: on the HDD, cold reads are genuinely disk-bound (~2.4x
slower than warm) but concurrent processes fight over one physical read head (seek
contention on a single spinning disk gets *worse*, not better, with more readers). On the
SSD, cold vs warm barely differs at all (6.2s vs 5.8s at n=1) -- a modern NVMe SSD delivers
data far faster than one CPU thread can decode it, so decode is CPU-bound regardless of
cache state there, and splitting only adds pure overhead (pre-pass read-ahead, process
startup, merging N result arrays) with no disk-wait time left to overlap it against.
Conclusion: multi-process splitting is not a useful lever for this workload on any tested
storage; the real lever is per-event decode speed (see below).

**Follow-up: staged per-event profiling of the pixel-trigger hot path (later session).**
Confirmed the ~2x pixel-trigger-vs-raster throughput gap (found earlier in this document)
is still present and re-measured cleanly: 25.1M events/s (pixel-trigger) vs 52.5M events/s
(raster), same machine, both files warm-cached, same operation (plain single-annulus
vSTEM). Read/write-wait counters (`reading waited`/`processing waited`, printed by
`terminate()`) already show decode -- not disk I/O -- is the bottleneck under normal
conditions (the read thread waits far more often than the process thread, meaning decode
can't keep up with supply). Isolated exactly which parts of `CHEETAH_pixeltrig::parse_event()`
cost what, via temporary surgical bypasses (each measured, then reverted):
- Bypassing the `pattern[]` memory lookup alone: no measurable change (23.9M vs 25.1M,
  within noise) -- re-confirms this session's earlier finding, the lookup was never the
  cost despite being the most "obvious" suspect (extra memory indirection vs raster's
  pure arithmetic).
- Bypassing the `toa` timestamp computation alone (dead work for plain, non-decluster
  vstem crediting -- computed unconditionally anyway): no measurable change (26.6M vs
  25.1M, within noise).
- Bypassing the entire credit switch-statement + `vstem()` call (keep everything else):
  31.0M events/s, a real ~24% gain -- crediting itself is a genuine, non-trivial cost,
  but explains less than a quarter of the total gap.
- Bypassing *everything* in `parse_event()` (lookup + address calc + credit + toa, only
  the `probe_count_chip[chip_id]%nxy` modulo and bounds check survive): 36.1M events/s,
  the largest single jump (+44% over baseline) -- but still 1.45x short of raster's
  52.5M, and at this point the wait-counter balance flips (process thread starts waiting
  for data more than the read thread waits for buffer space), meaning further parse_event()
  optimization is running into the read/buffer pipeline's own ceiling, not more available
  CPU headroom inside parse_event() itself.
- One genuine, permanent fix landed from this investigation: `_is_fourD_chunked` was 3
  enum comparisons + 2 boolean ORs recomputed on *every single event* regardless of
  functionType (raster's `parse_event()` has no equivalent check at all, since it has no
  FourD-chunk-drop path) -- replaced with `is_fourD_chunked_type`, a bool set once (in
  `Timepix.hpp`'s 3 `enable_FourD` overloads) instead of recomputed every event. Correct,
  harmless, verified not to regress the 8-way split's byte-for-byte match -- but its
  effect size turned out to be too small to separate from run-to-run measurement noise
  (~±10-15% between otherwise-identical runs) in isolation.

**Net conclusion, not yet fully resolved**: roughly a quarter of the 2x gap is
attributable to real, identified, "just how the credit path works" cost; a further chunk
closed by the maximum-bypass test but not attributable to any single line (likely
buffer/dispatch-loop-level overhead, not isolated further); and the investigation started
running into the read/buffer pipeline's own ceiling before fully explaining the remaining
~1.45x. Whoever continues this should start from the read/buffer pipeline itself (why
does the process thread start waiting on the read thread once parse_event() gets fast
enough?) rather than further parse_event()-internal micro-optimization, which is likely
close to exhausted as a lever on its own.

## Raw pixel-trigger `FourD`, actually fixed (later session)

The confirmed ~7%-event / ~21%-position loss from earlier in this document (blocked
behind a runtime guard rather than shipped broken) turned out to have nothing to do
with the two things suspected at the time (chip-to-chip trigger-count drift, or the
row-order-monotonicity assumption in the eviction-timing formula) -- both were measured
directly and ruled out (max real drift was 141 triggers, far smaller than a chunk's
~1200-trigger span; a from-scratch, pattern-order-agnostic rewrite of the eviction
formula changed nothing).

The real bug: `FourD.cpp`'s regular, raster-progress-driven chunk flush trigger was
guarded only by `!decluster`, not also excluding `CHEETAH_PIXELTRIG` -- so for raw
(non-decluster) pixel-trigger data, it fired *at the same time* as
`advance_fourD_pixeltrig_chunk`'s own dedicated flush mechanism in
`Cheetah_pixeltrig.hpp`. Both call `write_and_clean()`, which advances a single shared
sequential counter (`next_chunk_id`) -- so the two independently-timed triggers raced
and corrupted each other's chunk bookkeeping. Fixed by extending the exclusion (both the
regular per-chunksize trigger and the end-of-scan final-flush trigger) to also skip
`CHEETAH_PIXELTRIG`, mirroring the exclusion the decluster path already had.

Found by isolating variables one at a time: forcing a single giant "chunk" (no eviction
possible at all) dropped the mismatch from 55,735 positions to 60, proving the eviction
*mechanism* -- not chip desync, not anything in the per-event decode -- was the dominant
cause; from there, checking what else touches the shared flush/eviction state surfaced
the redundant trigger directly. Verified on the real 40856-trigger/262144-position
smart-scan file: 0.12% residual after the fix, the same small "drop the already-flushed
chunk's 4D voxel, keep Dose_image" tradeoff already accepted for declustered `FourD`.
The runtime guard has been removed; `FourD` with a pattern file and `decluster=False`
now works.

## Declustered `FourD`'s chunk buffer, partially fixed

`enable_FourD_declustered`'s cluster-callback flush (shared by both raster and
pixel-trigger declustered `FourD`) evicts chunk-groups from a 2-slot round-robin buffer.
Direct measurement (a new `diag_declustered_dropped` counter, since removed) found two
real, distinct bugs, both fixed:

1. **The buffer only ever used 1 of its 2 slots as real headroom.** The flush trigger
   evicted the oldest pending chunk-group as soon as ANY later cluster resolved (`> `,
   not `>=`), instead of waiting until slot pressure actually forced it. Fixed to delay
   eviction until a 3rd distinct chunk-group needs a slot (`cluster_chunk_id -
   fourD_declustered_chunk_id >= 2`), using memory that was already allocated.
2. **A latent indexing bug**, exposed by fix #1: the write itself computed its target
   slot/row from `fourD_declustered_chunk_id` (the oldest pending chunk) rather than the
   cluster's own absolute chunk id -- harmless only because the old eager-eviction bug
   kept the two values always equal. Once #1 allowed them to differ by 1, this would
   have written past the end of a chunk's own row range. Fixed to index by the cluster's
   own chunk id, mirroring what the raw path's `count_chunked_32` already did correctly.

**Result**: raster-mode declustered `FourD`, which already had a small residual, is now
**exact** (0/262144 mismatches, confirmed with `n_threads=1` vs `8` determinism and an
exact cross-check against `Roi`'s declustered output). Pixel-trigger declustered `FourD`
improved from 261 to 141 mismatched positions (~46% reduction) -- but not eliminated.

**Not fully fixed, and not straightforward to**: measuring the actual lag (a new
`diag_declustered_max_lag` counter, since removed) found dropped clusters up to **30
chunk-groups** late on the real smart-scan file (out of ~32 groups total) -- the
Declusterer's own `cluster_range` look-ahead, combined with pixel-trigger's chip-blocked
raw file layout, delays cluster resolution far more than raster mode's naturally
near-monotonic order ever does. Closing this fully would mean buffering on the order of
the ENTIRE scan's worth of chunk-groups simultaneously -- fine for this test's tiny
per-chunk size (det_bin=32, ~0.26MB/chunk) but scaling to hundreds of GB for a
full-resolution detector. Not attempted; flagged as a real, understood, but
memory-impractical-to-fully-close limitation specific to pixel-trigger declustered
`FourD`.
