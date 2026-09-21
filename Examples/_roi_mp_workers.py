"""Module-level worker for the ROI.ipynb multi-process (sidecar-checkpointed) example.

Needs to live in its own file, not a notebook cell: Windows' multiprocessing "spawn"
start method pickles the worker function by import path, so it must be importable
at module scope in a real .py file (a notebook cell's function isn't importable this
way, so ProcessPoolExecutor.map would fail to pickle it if defined inline).
"""
import sys
import numpy as np

EVENTEM_ROOT = r"C:\My Files\OneDrive - Universiteit Antwerpen\GitHub\evenTem"
EVENTEM_BUILD = r"C:\My Files\OneDrive - Universiteit Antwerpen\GitHub\evenTem\BuildTools\build_MSVC_fresh\Release"


def decode_one_slice(args):
    """Decode this worker's own byte-range slice of the file and return its
    partial scan image / diffraction pattern. Summing every worker's result
    reconstructs the same answer a single-process run would give, because the
    checkpoints split the file into disjoint, non-overlapping line ranges."""
    filename, nx, ny, dwell_time_ns, roi_xywh, file_byte_offset, line_number_offset, stop_at_line, seed = args
    for p in (EVENTEM_ROOT, EVENTEM_BUILD):
        if p not in sys.path:
            sys.path.insert(0, p)
    from EvenTem import Roi

    r = Roi(nx=nx, ny=ny, repetitions=1, filename=filename, extract_4D=False)
    r.DwellTime = dwell_time_ns
    x, y, width, height = roi_xywh
    r.SetROI(x_origin=x, y_origin=y, width=width, height=height)
    r.n_threads = 1
    r.file_byte_offset = file_byte_offset
    r.line_number_offset = line_number_offset
    r.stop_at_line = stop_at_line
    if seed is not None:
        r.seed_dt = seed["dt"]
        r.seed_rise_t = seed["rise_t"]
        r.seed_rise_fall = seed["rise_fall"]
        r.seed_line_count = seed["line_count"]
        r.seed_chip_id = seed["chip_id"]
    r.Run()
    # ScanImage/DiffractionPattern are already reshaped 2D numpy arrays (unlike
    # the raw Roi_scan_image/Roi_diffraction_pattern flat Python lists the
    # underlying C++ binding exposes) -- return them as plain arrays so the
    # caller can just sum the per-worker results directly.
    return np.asarray(r.ScanImage), np.asarray(r.DiffractionPattern)


def build_worker_plan(checkpoints, n_workers):
    """(file_byte_offset, line_number_offset, stop_at_line, seed_dict|None) per worker.

    Worker 0 always starts at the true file start (offset 0, line 0, no seed --
    every chip's very first packet is a real header, so chip attribution is
    unambiguous). Every later worker resumes mid-file at its own checkpoint and
    needs the seed dict to know each chip's in-flight rise/fall state, since it
    has no header of its own to re-derive that from. Only the last worker reads
    all the way to the true end of file (stop_at_line=-1); every other worker
    stops exactly at the next worker's start line.
    """
    plan = []
    for i in range(n_workers):
        if i == 0:
            file_byte_offset, line_number_offset, seed = 0, 0, None
        else:
            cp = checkpoints[i - 1]
            file_byte_offset, line_number_offset = cp[0], cp[1]
            seed = {"dt": cp[2], "rise_t": cp[3], "rise_fall": cp[4], "line_count": cp[5], "chip_id": cp[6]}
        stop_at_line = checkpoints[i][1] if i < n_workers - 1 else -1
        plan.append((file_byte_offset, line_number_offset, stop_at_line, seed))
    return plan
