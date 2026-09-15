/* Copyright (C) 2025 Thomas Friedrich, Chu-Ping Yu, Arno Annys
 * University of Antwerp - All Rights Reserved. 
 * You may use, distribute and modify
 * this code under the terms of the GPL3 license.
 * You should have received a copy of the GPL3 license with
 * this file. If not, please visit: 
 * https://www.gnu.org/licenses/gpl-3.0.en.html
 * 
 * Authors: 
 *   Thomas Friedrich <>
 *   Chu-Ping Yu <>
 *   Arno Annys <arno.annys@uantwerpen.be>
 */

#ifndef FOURD_H
#define FOURD_H


#include <numeric>
#include <type_traits>
#include <cstdint>
#include <string>

#include "LiveProcessor.h"


#include "H5Cpp.h"

template<int BitDepth>
class FourD : public LiveProcessor
{
private: 
   
public: 

    int bitdepth;
    int deflate_factor;

    std::vector<uint64_t> Dose_image;

    // Bare element type matching chunk_data's own vector<T> below (uint8_t/16/32
    // per BitDepth) -- named here so the full-scan buffers further down (which are
    // vector<vector<T>>, not vector<T>[2]) don't have to repeat this same
    // conditional chain.
    using ChunkElemT = typename std::conditional<BitDepth == 8, uint8_t,
                typename std::conditional<BitDepth == 16, uint16_t,
                    typename std::conditional<BitDepth == 32, uint32_t,
                        void // Handle unsupported bitdepths
                    >::type
                >::type
            >::type;

    typename std::conditional<BitDepth == 8, std::vector<uint8_t>,
                typename std::conditional<BitDepth == 16, std::vector<uint16_t>,
                    typename std::conditional<BitDepth == 32, std::vector<uint32_t>,
                        void // Handle unsupported bitdepths
                    >::type
                >::type
            >::type chunk_data[2];

    // Declustered 4D conversion: every chunk-group gets its own permanently-
    // allocated slot for the whole run, instead of chunk_data's 2-slot round-robin
    // buffer above -- a late-resolving cluster can never target an already-
    // flushed-and-recycled slot, so this is unconditionally exact regardless of
    // how far cluster resolution lags decode (previously bounded to a small fixed
    // tolerance -- 1, then 2, chunk-groups -- with anything later silently
    // dropped; this was the source of the declustered-FourD pixel-trigger residual
    // documented in IMPROVEMENTS.md). Trades memory (the whole 4D cube, only for
    // the declustered path) for exactness -- sized once decluster=True's
    // ny/chunksize/diff_pattern_size are known, in run() below. Flushed to disk in
    // one pass, in order, in flush_declustered_chunks() once decluster_thread has
    // fully joined (only then is it certain no more clusters can ever resolve).
    std::vector<std::vector<uint32_t>> declustered_chunk_data;
    std::vector<std::mutex> declustered_chunk_mtx;
    void allocate_declustered_chunks();
    void flush_declustered_chunks();

    // Same fix, generalized across bitdepths, for RAW (non-declustered)
    // pixel-triggered FourD: advance_fourD_pixeltrig_chunk (Cheetah_pixeltrig.hpp)
    // had the identical 2-slot-eviction hazard -- an event whose own chip lagged
    // the aggregate min-progress-driven flush trigger could target an
    // already-flushed, recycled chunk-group and have its 4D voxel write silently
    // dropped (Dose_image still credited). Every chunk-group gets its own
    // permanent slot here too, for the same reason and with the same tradeoff
    // (memory for exactness) -- sized once ny/chunksize/diff_pattern_size are
    // known, in run() below. Flushed once cam.terminate() guarantees decode has
    // fully finished.
    std::vector<std::vector<ChunkElemT>> full_chunk_data;
    std::vector<std::mutex> full_chunk_mtx;
    void allocate_full_chunk_buffer();
    void flush_full_chunk_buffer();

    size_t det_bin = 1;
    size_t scan_bin = 1;
    size_t diff_pattern_size;
    size_t diff_pattern_length;
    size_t chunksize = 16;
    // Scan-x chunk extent -- previously implicit "full row width" (nx), which made
    // every chunk span the ENTIRE scan row regardless of chunksize, so any ROI read
    // spanning only a few lines and a narrow x-range still had to touch/decompress
    // full-width chunks. Small ROI + several lines is the confirmed normal read
    // pattern, and downstream reads always want the FULL diffraction pattern per
    // scan point, so only the scan dimensions need small chunks -- detector
    // dimensions intentionally stay full-extent per chunk (unchanged below).
    size_t chunksize_x = 16;

    // Output format: "hdf5" (default -- unchanged from before this existed) or
    // "zarr". Caller decides; both stay in the API. Validated in init_4D_file().
    // Zarr output is a hand-rolled, spec-compliant Zarr v2 store (directory of
    // chunk files + .zarray/.zgroup JSON metadata) using zlib as the codec --
    // chosen over Blosc/a full C++ Zarr library specifically to avoid a heavy new
    // dependency chain (installed via vcpkg: zlib:x64-windows-static-md, a static
    // lib with no extra DLL to bundle, matching the dynamic CRT pybind11 needs).
    // Zarr's one-file-per-chunk layout is what actually fixes the slow/wrong-shaped
    // read problem -- it's directly, independently readable by any standard
    // zarr-python/dask.array.from_zarr client, no custom reader needed.
    std::string format = "hdf5";
    // Base output path (without extension) and the Zarr store's root directory,
    // captured as real string members (not the raw-pointer-into-a-temporary
    // approach H5FILE_NAME below uses, which is a pre-existing latent dangling-
    // pointer bug -- harmless today only because H5FILE_NAME happens to never be
    // read again after the constructor -- not replicated here).
    std::string base_path;
    std::string zarr_root;

    // When true (default), the output file/store also gets the "dose_image" and
    // "shape" auxiliary datasets alongside "4D" -- unchanged prior behavior. When
    // false, only "4D" is written (init_4D_file_*/init_4D_file_zarr skip creating
    // "shape", save_dose_image/save_dose_image_zarr skip creating "dose_image"
    // entirely) -- Dose_image itself is still computed and available in Python via
    // the .Dose_image property either way, just not persisted to disk. The point:
    // a single-array HDF5/Zarr container is what generic readers expect --
    // hyperspy's hs.load(fn, lazy=True) in particular assumes exactly one
    // (dataset-shaped, chunked) array per file/store and gets confused by extra
    // siblings at the same level.
    bool save_metadata = true;

    // Charge-weighted, declustered 4D conversion (CHEETAH/.tpx3 only, 32-bit only
    // -- see FourD::run() -- and scan_bin=1, see Timepix.hpp's
    // enable_FourD_declustered). Off by default: when false, behavior is
    // byte-for-byte identical to before this feature existed. Mirrors
    // Roi/vSTEM/Var's own decluster fields exactly (same defaults, same histogram
    // shapes) so a single ToT-sum calibration can be reused across all of them.
    bool decluster = false;
    uint64_t dtime = 100;
    uint16_t dspace = 6;
    int cluster_range = 256;
    double tot_per_electron = 0.0;
    // Alternative to tot_per_electron: resolve each cluster's electron count from a
    // 2D (cluster size, ToT) map instead of a single ToT/TotPerElectron ratio -- see
    // ClusterResolver.hpp's resolve_electron_count_lut. Both stay available side by
    // side; electron_count_lut_file (if non-empty) is loaded into electron_count_lut
    // at the start of run(), and a non-empty electron_count_lut is what actually
    // switches a run over to the map-based resolver -- tot_per_electron is otherwise
    // unaffected and remains the default when neither is set.
    std::vector<std::vector<int>> electron_count_lut;
    std::string electron_count_lut_file = "";
    std::vector<int> clustersize_histogram = std::vector<int>(50, 0);
    std::vector<int> energy_histogram = std::vector<int>(4096, 0);
    std::vector<std::vector<int>> clustersize_tot_histogram = std::vector<std::vector<int>>(50, std::vector<int>(4096, 0));

    bool b_save_4D = false;
    bool b_first_run = true;
    bool b_accumulate = false;

    H5::DataSet dataset;
    H5::DataSet dataset4D;

    std::mutex mtx[2];

    // Monotonically incrementing chunk index, one per real flush -- replaces
    // deriving the chunk id from a "line number" argument (the old scheme's
    // end-of-image flush passed the wrong variable, `nx` instead of the actual
    // completed-line count, and could redundantly recompute the same chunk id
    // already just flushed by the regular per-chunksize trigger). Reset to 0 in
    // reset().
    int next_chunk_id = 0;

    void run();
    void reset();

    // n_valid_rows: how many of the buffer's chunksize rows actually hold real data
    // -- always chunksize/scan_bin for a regular full flush, but less for the final
    // flush of a scan whose height isn't an exact multiple of chunksize.
    void write_and_clean(int n_valid_rows);

    void allocate_chunk();

    void init_4D_file_8();
    void init_4D_file_16();
    void init_4D_file_32();
    void init_4D_file();


    void save_dose_image();
    void write_chunk(const std::vector<uint8_t>& chunk , hsize_t chunk_id, hsize_t n_valid_rows);
    void write_chunk(const std::vector<uint16_t>& chunk , hsize_t chunk_id, hsize_t n_valid_rows);
    void write_chunk(const std::vector<uint32_t>& chunk , hsize_t chunk_id, hsize_t n_valid_rows);

    // Zarr-format counterparts. Not templated on BitDepth (unlike the methods
    // above, which are members of this template class) -- these are free
    // functions in FourD.cpp, shared identically across all three instantiations
    // since chunk writing at the byte level doesn't depend on BitDepth beyond the
    // element size already passed in explicitly.
    void init_4D_file_zarr(const std::string& dtype_str);
    void save_dose_image_zarr();


    void line_processor(
        size_t &img_num,
        size_t &first_frame,
        size_t &end_frame,
        ProgressMonitor *p_prog_mon,
        size_t &fr_total_u,
        BoundedThreadPool *pool
    );

    // H5
    const char* H5FILE_NAME;
    H5::H5File h5file;


    // Constructor
    FourD(std::string f,int repetitions, int bitdepth, int deflate_factor) : LiveProcessor(repetitions),
    bitdepth(bitdepth),
    deflate_factor(deflate_factor)
    {
        base_path = f;
        zarr_root = f + ".zarr";
        f = f + ".hdf5";
        H5FILE_NAME = f.c_str();
        // Always created eagerly here, same as before this session -- even if
        // `format` later gets set to "zarr" (format is a post-construction
        // property, like chunksize, so it isn't known yet at this point). This
        // means an unused, empty .hdf5 file is left alongside a Zarr output; a
        // real fix would need restructuring when file creation happens relative to
        // when the caller sets properties, which risks the existing dose-image-
        // only workflow (run() with no init_4D_file() call) that currently relies
        // on this eager creation. Left as a known, minor, documented quirk.
        h5file = H5::H5File(H5FILE_NAME, H5F_ACC_TRUNC);

        if (!((bitdepth ==8) || (bitdepth == 16) || (bitdepth == 32)))
        {
            throw std::invalid_argument("bitdepth must be 8, 16 or 32");
        }
        if (deflate_factor < 1 || deflate_factor > 9)
        {
            throw std::invalid_argument("deflate factor must be between 1 and 9");
        }
    };

    // Destructor
    ~FourD()
    { 
    };
};
#endif // !FOURD_H
