# Known limitations & future improvements

Snapshot as of this session. Everything below is either a known, accepted
limitation (with the reason it's accepted) or a real lead worth picking up later.
See `SESSION_OVERVIEW.md` for the full narrative/history behind each item —
this file is the short, action-oriented version.

## Smart-scan (pixel-trigger) per-event decode speed -- the real lever

Pixel-trigger mode decodes at roughly **half the throughput of raster mode**
(~25M events/s vs ~52M events/s, same machine, same operation, both files
warm-cached). This is the most promising thing to chase next, because -- unlike
multiprocessing (below) -- a fix here would speed up *every* smart-scan run
unconditionally, on any hardware.

What's already known, from direct isolation experiments (each measured by
temporarily bypassing part of `CHEETAH_pixeltrig::parse_event()`, then
reverting):

| bypassed | throughput | change |
|---|---|---|
| nothing (baseline) | 25.1M events/s | — |
| the `pattern[]` lookup | 23.9M | none (ruled out, twice, across two sessions) |
| the `toa` timestamp calc | 26.6M | none (dead work for plain vSTEM crediting, but cheap enough not to matter) |
| the whole credit switch + `vstem()` call | 31.0M | **+24%**, real |
| all of the above combined | 36.1M | **+44%**, but still 1.45x short of raster |

One real, permanent fix landed from this: `_is_fourD_chunked` (3 enum
comparisons + 2 boolean ORs, recomputed on *every* event regardless of
`functionType`) is now a single bool (`is_fourD_chunked_type`, set once in
`Timepix.hpp`'s 3 `enable_FourD` overloads) -- correct, harmless, verified not
to regress the FourD pixel-trigger correctness check, but too small an effect
to separate from run-to-run noise on its own.

**Next step, concretely**: at the "everything bypassed" data point, the
read/process wait-counters flip -- the process thread starts waiting on the
read thread more than the other way around. That means the remaining ~1.45x
gap is **not** inside `parse_event()` anymore; it's in the read/buffer
pipeline (`read_file()`/`schedule_buffer()`'s buffer handshake, or something
about how buffers are sized/filled). Start there, not with more
`parse_event()` micro-optimization -- that lever is close to exhausted.

## Multi-process file-splitting for smart-scan -- correct, but disabled by default

Fully implemented and byte-for-byte correctness-verified at every split count
tested (2, 4, 8, 16, 32), including through the real `ProcessPoolExecutor`
path, not just an in-process test harness. Two real, non-obvious bugs were
found and fixed along the way (both documented in detail in
`SESSION_OVERVIEW.md`):

1. The output-copy sweep that decides "is decode done yet" raced against the
   camera's own exact stop signal, using a coarse proxy that could win the
   race and truncate decode early -- fixed by having the camera signal
   termination directly and exactly.
2. `chip_id` (which of the 4 detector chips a packet belongs to) had no
   default value and wasn't seeded for a resumed worker -- undefined
   behavior, ranging from silently wrong data to an outright crash. Fixed by
   capturing and seeding it at each split checkpoint, same as the other
   per-chip state.

**Despite being correct, it's commented out by default** (`mp_vstem_runner.
run_vstem_parallel_pixeltrig` / `vstem_mp_worker.run_slice_pixeltrig` --
search for "COMMENTED OUT FOR NOW" in both files) because it showed **no
real speedup in any of 4 tested hardware/cache combinations**: a real
spinning HDD and a real NVMe SSD, each both warm and cold. Two different
reasons:
- On the HDD, cold reads are genuinely disk-bound, but concurrent processes
  fight over the one physical read head -- seek contention gets *worse*, not
  better, with more readers on a single spinning disk.
- On the SSD, cold vs warm barely differs at all -- a modern NVMe SSD
  delivers data far faster than one CPU thread can decode it, so decode is
  CPU-bound regardless of cache state, and splitting only adds pure overhead
  (pre-pass read-ahead, process startup, merging N result arrays) with no
  disk-wait time left to overlap against.

**When to reconsider**: if a genuinely much larger file is involved, or
storage where concurrent reads actually scale (RAID, a fast network
filesystem, multiple independent disks) -- not a single conventional drive
of either kind. The commented-out code and the underlying C++ engine
(`vSTEM.find_checkpoints_pixeltrig`, `seed_chip_id`, etc.) are both left in
place and correctness-verified, so re-enabling is just uncommenting the
Python convenience functions.

The equivalent raster-mode feature (`mp_vstem_runner.run_vstem_parallel`,
NOT commented out) is different: it showed a real ~15x win on a cold file
cache in earlier testing, so it stays enabled. It does have its own smaller
open issues -- see the raster section below.

## Pixel-trigger `FourD`: now exact, both raw and declustered

Raster-mode `FourD` (raw and declustered) was already byte-for-byte exact.
Pixel-trigger (smart-scan) `FourD` was not, on EITHER path -- both the raw
(non-declustered) path and the declustered path independently showed
141/262144 mismatched positions (a 0.054%-scale gap, down from an earlier
261/262144 for the declustered path after other fixes) between `Dose_image`
and the on-disk `4D` array.

**Root cause (same shape on both paths):** the on-disk chunk buffer only had 2
physical round-robin slots, evicted (flushed to disk and recycled) as soon as
progress advanced 2 chunk-groups past the oldest pending one --
`fourD_declustered_chunk_id` vs. each resolved cluster's own chunk id on the
declustered path, or the aggregate minimum per-chip trigger progress vs. each
event's own chunk id (`advance_fourD_pixeltrig_chunk`, `Cheetah_pixeltrig.hpp`)
on the raw path. A cluster/event whose own resolution or chip lagged that
2-chunk-group tolerance (pixel-trigger's chip-blocked raw file layout can cause
this on its own; the declusterer's own `cluster_range` lookahead adds a second,
separate source of lag for the declustered path specifically) targeted an
already-recycled slot and had its 4D voxel credit silently dropped
(`Dose_image` stayed correct either way, since that write happens directly
with no intermediate buffer -- only the on-disk 4D array under-counted).

**Fixed**, both paths, the same way: every chunk-group now gets its own
permanently-allocated slot for the whole run (`FourD::declustered_chunk_data`
for the declustered path, `FourD::full_chunk_data` -- bitdepth-generic,
8/16/32 -- for the raw path) instead of a small fixed-size round-robin window
-- no eviction, no ordering assumption between clusters/events, and therefore
no "already flushed" case at all. Flushed to disk in one pass
(`flush_declustered_chunks()`/`flush_full_chunk_buffer()`) only after
`cam.terminate()` guarantees decode/`decluster_thread` has fully finished, i.e.
every credit in the file has already landed. This trades peak memory (the
whole 4D cube stays resident for whichever path is running) for unconditional
exactness -- confirmed 0/262144 mismatches on the same real 40856-trigger
smart-scan file for BOTH paths, and re-confirmed no regression on raster mode
(still exact for bitdepth 16/32; bitdepth 8's large, pre-existing mismatch
count on this same file is `uint8` accumulator overflow on a high-dose scan --
confirmed byte-for-byte identical, mismatch and all, against the pre-fix build,
so unrelated to this change -- `n_threads=1` vs `n_threads=8` still
byte-for-byte identical for declustered raster on
`raw_0643_-40.00_000000.tpx3`).

## Raster-mode multi-process splitting: now exact at every split count

The `chip_id`-seeding lesson from the pixel-trigger fix above *did* turn out
to apply here too, despite the earlier note above suspecting it might not:
raster's `Cheetah.hpp` defaults `chip_id = 0` on a fresh object, which is
harmless for a normal whole-file run (the very first packet is always a
header, correcting it before it's ever used) -- but a resumed multi-process
worker starts reading mid-file, where there's no header at its own
`byte_offset` at all (headers only appear at the START of a chip's block, and
`byte_offset` -- "right after the triggering chip's own TDC-fall" -- almost
always lands MID-block). Until that worker's own first real header happens to
appear, every event it decodes gets misattributed to chip_id 0's
`rise_t`/`dt`/`address_bias`/`address_multiplier`, computing wrong
positions/timing for all of them -- not dropping them, silently miscomputing
them. Found by tracing a ~19%-of-one-row `vSTEM` undercount at exactly the
split boundary row on a real file (`Cycled BSCFO/raw_0140...`) down to this,
after ruling out the dwell-time-units and TDC-fall-guard changes made
earlier the same session as the cause (confirmed via a stale, pre-those-
changes build showing the identical residual).

Fixed the same way `CHEETAH_pixeltrig` already fixed the identical hazard for
itself (`seed_chip_id`): `find_line_checkpoints()` already tracks `chip_id` as
a side effect of scanning every packet (`which_type()`) during its pre-pass,
so its value at the exact checkpoint moment is already correct for whatever
block `byte_offset` resumes into -- it just wasn't being captured or threaded
through to the resuming worker. Added to the checkpoint tuple, seeded in
`CHEETAH::reset()`, and threaded through `vSTEM.h`/`vSTEM.cpp` and the Python
glue (`mp_vstem_runner.py`/`vstem_mp_worker.py`).

Result: `n_processes` in `{1, 2, 3, 4, 8, 16}` all confirmed byte-for-byte
exact on the same real file that previously showed a residual at every one of
those except `{1, 2, 4}` -- superseding the "confirmed exact at {1,2,4} only"
note below and in `mp_vstem_runner.py`'s own docstring.

## Declustering + multi-process splitting together: not recommended (raster)

Measured strictly slower at every `n_processes > 1` compared to
`n_processes=1` with a larger thread count, and not byte-for-byte exact
(a cluster straddling a worker's byte-range boundary can be resolved
incompletely by two independent `Declusterer` instances). This is a
structural mismatch, not a bug to fix: declustering already has its own
internal thread pool that scales well in a single process; splitting into
multiple processes necessarily divides that same thread budget across
workers, so total declustering parallelism goes *down* as `n_processes` goes
up. Recommendation stands as-is: `n_processes=1` with as many declustering
threads as the core count allows.

## Smaller, longstanding items (not touched this session)

- **Roi's small residual at scan position (0,0)** specifically (a <0.03%
  event-count gap, separate from and much smaller than any of the issues
  above) -- flagged in earlier work, deliberately deferred as narrow and
  low-priority.
- **`Electron.h`'s `decluster=true` default** -- inconsistent with the
  project's general "off by default" convention for this feature elsewhere.
  Flagged, never resolved either way.
- **ROI-via-pre-pass for raster mode** (the idea that motivated some of this
  session's benchmarking): reusing the same fast TDC-only pre-pass that
  finds split checkpoints to instead jump straight to a requested scan-line
  range, skipping full decode of everything before it. Not yet built. Should
  be a clean single-process win for raster (a scan-line range is always one
  contiguous byte range there); the equivalent idea for smart-scan was tested
  and hit an unexplained ~10-25x per-event slowdown when resuming mid-file
  that was never root-caused (separate from, and not explained by, either
  correctness fix in this document) -- worth understanding before building a
  smart-scan version of this feature.
