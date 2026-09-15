# How `eventem` computes `Roi` and `vSTEM` results

This explains the actual computation pipeline behind `Roi` and `vSTEM` for `.tpx3`
(Timepix3/CHEETAH) data -- what each raw detector event goes through on its way to becoming
a scan image, a diffraction pattern, or a virtual-STEM image. It's about the *algorithm*, not
the API -- see `EvenTem/eventem.py`'s docstrings for how to call these from Python.

## 1. The big picture

Every raw hit read from the `.tpx3` file goes through the same three-stage pipeline,
regardless of whether it ends up feeding `Roi`, `vSTEM`, `Var`, or `FourD`:

```
raw packet  -->  (A) which scan position did this happen at?
                 (B) which detector pixel did this land on?
            -->  (C) accumulate it into whichever output(s) this mode produces
```

Stage (C) is the only part that differs between `Roi` and `vSTEM` (and `Var`, not covered
here). Stages (A) and (B) are shared, identical code for every mode.

## 2. Stage A: which scan position?

There are two different ways this gets resolved, depending on how the acquisition was
configured.

### 2a. Normal raster scan (the default)

The detector's TDC (time-to-digital converter) input receives one electrical pulse per scan
line, synchronized with the microscope's own scan generator: a rise when the beam starts a
new line, a fall when it ends. From two consecutive line pulses, the dwell time per pixel is
derived directly from real hardware timing, not from what the user thinks it should be:

```
line_interval = fall_time - rise_time         (this line's total duration)
dt            = line_interval / nx            (dwell time per pixel, this line)
```

Each hit's own time-of-arrival (`toa`) is then converted into a column index within the
current line:

```
column = (toa - rise_time) / dt
```

If `column` comes out `>= nx`, the hit arrived after the line was supposed to end (e.g.
during flyback) and is discarded. Otherwise the flat scan index is:

```
probe_position = column + (line_number % ny) * nx
```

**The very first line is a special case.** `dt` above is calibrated from the *previous*
line's own rise/fall -- but there is no "line -1," so line 0 uses a fallback `dt` computed
from the dwell time the user passed in Python (`Roi.dt = 100`, in microseconds). This is a
reasonable approximation as long as the real hardware dwell time matches what was configured
-- it does not need to be recalibrated per line, just seeded once.

**Multiple detector chips.** A quad Timepix3 sensor has 4 independent chip quadrants, each
with its own line counter. The "current line" used everywhere above is the *minimum* across
all 4 chips, not any single chip's own count -- so a faster chip doesn't get to prematurely
signal "this line/scan is done" while a slower chip is still catching up. (A related but
distinct bug -- a faster chip's line counter being allowed to run *past* the intended scan
length before the aggregate stop condition caught up -- was found and fixed this session; see
`SESSION_OVERVIEW.md` section 6.)

### 2b. Smart-scan / pixel-triggered acquisitions (`set_pattern_file`)

Some acquisitions don't raster uniformly -- instead, a pre-computed pattern selects a sparse
subset of scan positions to actually visit (e.g. only regions containing a feature of
interest). For these, the detector's trigger line fires once *per selected pixel*, not once
per line, and there is no dwell-time calculation at all: the `N`-th trigger's scan position
is read directly out of the pattern file, `pattern[N]`, where the file itself is just a list
of `row * nx + col` indices, one per line, in visiting order.

## 3. Stage B: which detector pixel?

Independent of how the scan position was resolved, every hit packet also encodes which
physical pixel on the detector chip it landed on. This gets decoded into a
`(kx, ky)` pixel coordinate in the full, unbinned detector frame (0..511 for a 512x512
Timepix3), using a per-chip address decode (each of the 4 quadrants reports pixels in its own
local coordinate system, with a fixed rotation/flip and offset applied to place it correctly
within the full sensor).

## 4. Stage C, for `Roi`: windowed accumulation + optional 4D cube

### What it computes

For every hit whose scan position falls inside the requested ROI rectangle, `Roi` increments
three (or four) running totals at once, from the exact same event:

- **`Roi_scan_image`** -- one running count per scan position inside the ROI (an image of
  "how many hits landed while the beam was at this position").
- **`Roi_diffraction_pattern`** -- one running count per detector pixel, summed over *every*
  scan position in the ROI (the ROI's averaged/summed diffraction pattern).
- **`Roi_4D`** *(if `extract_4D=True`)* -- the same per-(scan position, detector pixel) count
  as a full 4D cube, not summed over either axis. `Roi_scan_image`/`Roi_diffraction_pattern`
  are literally that cube summed over the detector axes / scan axes respectively -- confirmed
  identical, position by position, during this session's testing.

### The coordinate convention (and a caveat)

`set_roi(x, y, width, height)` takes `(x, y)` in ordinary image coordinates -- origin at the
top-left, `y` increasing downward, matching how you'd describe a crop of a normal image.
Internally, scan row 0 corresponds to the *bottom* of the physical scan, so `set_roi()`
converts the user's top-left-origin rectangle into an internal bottom-left-origin one:

```
internal_bottom = ny - (y + height)
internal_top    = ny - y
```

Each event's row is then flipped back through the same convention before being placed into
`Roi_scan_image`, so that from the outside, `Roi_scan_image` reads top-to-bottom just like the
`(x, y)` you asked for -- the two flips are meant to cancel out.

**Caveat, not verified this session:** the per-event flip inside the hot path
(`roi()`/`roi_4D()`) uses the scan's `nx` (width) for this correction, while `set_roi()`
itself (and everywhere else this flip appears) uses `ny` (height). For a square scan
(`nx == ny`, true of every dataset tested this session) these are identical and nothing looks
wrong. For a genuinely non-square scan this asymmetry has not been tested and may not produce
the ROI you'd expect -- worth a dedicated non-square `Roi` test before relying on it.

## 5. Stage C, for `vSTEM`: annular detector integration

`vSTEM` ignores scan windowing entirely (every scan position counts) and instead filters by
*detector* position: does this hit fall inside a user-defined annulus on the detector?

```
d_squared = (kx - offset_x)^2 + (ky - offset_y)^2
if inner_radius^2 < d_squared <= outer_radius^2:
    vSTEM_image[scan_position] += 1
```

`InnerRadia`/`OuterRadia` are plain pixel-distance radii in the full, unbinned detector frame
(never affected by any detector binning setting) -- squared once internally for a cheap
integer/float comparison, not something you need to square yourself. `Offsets` defaults to
the detector's geometric center (`n_cam/2, n_cam/2`) unless set explicitly. This is exactly a
virtual annular dark-field (or bright-field, with `inner_radius=0`) detector: every hit inside
the ring contributes one count to that scan position's brightness.

`vSTEM_stack` holds one such image per repetition (for time series / multi-frame
acquisitions); `vSTEM_image` is the same data flattened to the single-repetition case.

## 6. The declustered variant (shared by `Roi` and `vSTEM`)

Everything above treats each raw pixel activation as one electron. In reality, one physical
electron typically lights up several *adjacent* pixels at close-together times (the charge
cloud spreads across the sensor) -- so naively, one electron gets over-counted as several.
`decluster=True` fixes this with a completely separate pass, run in place of stages B/C above
for that event stream:

1. **Group nearby hits into clusters.** Hits are examined roughly in arrival order; a hit
   joins an existing open cluster if it's within `Dspace` pixels (independently in x and y) and
   `Dtime` ToA ticks of that cluster's seed hit, searched over the next `ClusterRange`
   subsequent hits. This is a fast, single-pass, greedy grouping -- not a full nearest-neighbor
   search.
2. **Resolve one position per cluster.** All member hits' `(kx, ky)` are averaged, weighted by
   each hit's own ToT (time-over-threshold, roughly proportional to deposited charge) -- so the
   resolved position is the cluster's ToT-weighted centroid, not just the seed pixel or an
   unweighted average.
3. **Resolve how many electrons.** The cluster's *total* summed ToT (across all member hits)
   is divided by `TotPerElectron` -- a calibration constant you read off your own data (the
   typical summed ToT of one real single-electron cluster, at your beam energy and detector
   threshold) -- and rounded to the nearest whole number. A cluster far below
   `TotPerElectron` (noise, X-rays, cosmic rays) resolves to 0 electrons and is dropped
   entirely; a cluster near 2x, 3x, ... `TotPerElectron` resolves to genuine electron pile-up.
4. **Credit `N`, not `+1`.** Whatever `Roi`/`vSTEM` would have incremented by `+1` for a raw
   hit, the resolved cluster instead increments by its resolved electron count `N`, at its
   one resolved centroid position -- for `Roi` this still respects the ROI window (using the
   same coordinate convention as section 4); for `vSTEM` this still requires the centroid to
   fall inside the configured annulus.

This clustering step is what `n_threads` parallelizes: each buffered chunk of raw hits (a
"ring-buffer slot") is independent of the others and can be declustered on its own thread,
while a single dedicated thread delivers the resolved clusters back in strict original order
-- verified to give byte-for-byte identical output regardless of thread count.

## 7. The ToT-sum variant (`Roi.tot_mode`, no clustering)

A third, simpler alternative to counting hits or declustering: `tot_mode=True` makes every
raw hit contribute its own ToT value (not `+1`) directly to `Roi_scan_image` /
`Roi_diffraction_pattern` / `Roi_4D`, individually, with no clustering step at all. The result
is "total deposited charge per probe" rather than "electron count per probe" -- useful as a
cheap, threshold-free intensity proxy, but it still counts each pixel of a multi-pixel charge
cloud separately (unlike declustering, which collapses a whole cluster to one position first).
Deliberately mutually exclusive with `decluster` -- pick one or the other.

## 8. Quick reference

| | `Roi` | `vSTEM` |
|---|---|---|
| Filters by | scan position (ROI rectangle) | detector position (annulus) |
| Produces | scan image, diffraction pattern, optional 4D cube | one image (dose per scan position) |
| Default counting | +1 per raw hit | +1 per raw hit |
| `decluster=True` | credits N electrons at cluster centroid, if centroid is inside the ROI | credits N electrons at cluster centroid, if centroid is inside the annulus |
| `tot_mode=True` | sums each raw hit's own ToT instead of counting it | not available |
| Position source | raster (TDC + dwell time) or `set_pattern_file` (smart-scan) -- same for both |
