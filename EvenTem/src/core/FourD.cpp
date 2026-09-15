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

#include "FourD.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <zlib.h>

namespace {
    // Writes a Zarr v2 array's ".zarray" metadata file, and creates the array's
    // directory. dtype follows numpy's typestr convention (e.g. "<u4" for
    // little-endian uint32). deflate_factor 0 means no compression
    // (compressor: null); 1-9 maps directly to zlib's compression level. Bin
    // dimension_separator explicitly to "." (the classic Zarr v2 default) rather
    // than relying on a reading client's own default, since that default has
    // changed between zarr-python versions.
    void write_zarr_array_meta(const std::string& array_dir,
        const std::vector<long long>& shape, const std::vector<long long>& chunks,
        const std::string& dtype, int deflate_factor)
    {
        std::filesystem::create_directories(array_dir);
        std::ostringstream ss;
        ss << "{\"zarr_format\":2,\"shape\":[";
        for (size_t i = 0; i < shape.size(); ++i) { if (i) ss << ","; ss << shape[i]; }
        ss << "],\"chunks\":[";
        for (size_t i = 0; i < chunks.size(); ++i) { if (i) ss << ","; ss << chunks[i]; }
        ss << "],\"dtype\":\"" << dtype << "\",";
        if (deflate_factor > 0)
            ss << "\"compressor\":{\"id\":\"zlib\",\"level\":" << deflate_factor << "},";
        else
            ss << "\"compressor\":null,";
        ss << "\"fill_value\":0,\"order\":\"C\",\"filters\":null,\"dimension_separator\":\".\"}";
        std::ofstream f(array_dir + "/.zarray", std::ios::binary);
        f << ss.str();
    }

    // Writes one raw (already-gathered, contiguous) buffer as a Zarr v2 chunk
    // file, optionally zlib-compressed. Byte-level and dtype-agnostic -- shared
    // by all three FourD<BitDepth> instantiations rather than being a class
    // member, since nothing here actually depends on BitDepth once the caller has
    // already sized `bytes` correctly.
    void write_zarr_chunk_raw(const std::string& array_dir,
        const std::vector<std::string>& chunk_index, const unsigned char* bytes, size_t n_bytes,
        int deflate_factor)
    {
        std::ostringstream name;
        for (size_t i = 0; i < chunk_index.size(); ++i) { if (i) name << "."; name << chunk_index[i]; }
        std::string chunk_path = array_dir + "/" + name.str();

        if (deflate_factor > 0)
        {
            uLong bound = compressBound((uLong)n_bytes);
            std::vector<unsigned char> compressed(bound);
            uLongf dest_len = bound;
            compress2(compressed.data(), &dest_len, bytes, (uLong)n_bytes, deflate_factor);
            std::ofstream f(chunk_path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(compressed.data()), (std::streamsize)dest_len);
        }
        else
        {
            std::ofstream f(chunk_path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(bytes), (std::streamsize)n_bytes);
        }
    }

    // Zarr v2 requires every chunk file to be the FULL declared chunk shape, even
    // at array edges (unlike HDF5's hyperslabs, which handle partial boundary
    // chunks natively) -- so partial scan-edge blocks (n_valid_rows < chunk_y, or
    // this_nx < chunk_x) get padded with zeros (matching fill_value:0 in the
    // metadata) before compression, sized to exactly what a reader expects for
    // that chunk index.
    template <typename T>
    void write_zarr_4D_chunk(const std::string& array_dir, const T* gathered,
        size_t n_valid_rows, size_t this_nx, size_t chunk_y, size_t chunk_x, size_t det_len,
        long long chunk_id_y, long long chunk_id_x, int deflate_factor)
    {
        std::vector<T> padded((size_t)(chunk_y * chunk_x * det_len * det_len), 0);
        size_t src_row_elems = this_nx * det_len * det_len;
        size_t dst_row_elems = chunk_x * det_len * det_len;
        for (size_t ly = 0; ly < n_valid_rows; ++ly)
            std::copy(gathered + ly * src_row_elems, gathered + ly * src_row_elems + src_row_elems,
                      padded.data() + ly * dst_row_elems);

        write_zarr_chunk_raw(array_dir,
            {std::to_string(chunk_id_y), std::to_string(chunk_id_x), "0", "0"},
            reinterpret_cast<const unsigned char*>(padded.data()), padded.size() * sizeof(T),
            deflate_factor);
    }
}

template<int BitDepth>
void FourD<BitDepth>::run(){
     py::gil_scoped_release release;

     reset();

     if (decluster)
     {
         if (camera != CAMERA::CHEETAH && camera != CAMERA::CHEETAH_PIXELTRIG)
             throw std::runtime_error("FourD decluster=True is currently only supported for .tpx3 (CHEETAH) files.");
         // electron_count_lut_file (a saved 2D map, see ClusterResolver.hpp) is an
         // alternative to tot_per_electron's single ToT/ratio formula -- loading it
         // here means electron_count_lut is populated before the tot_per_electron
         // check below, so a run relying solely on the map (tot_per_electron left at
         // its default 0.0) is not rejected.
         if (!electron_count_lut_file.empty())
             electron_count_lut = load_electron_count_lut_file(electron_count_lut_file);
         if (tot_per_electron <= 0.0 && electron_count_lut.empty())
             throw std::invalid_argument("FourD.tot_per_electron must be set (calibrated from your own cluster ToT-sum histogram), or electron_count_lut_file/electron_count_lut must be set, before enabling decluster=True.");
         if (scan_bin != 1)
             throw std::invalid_argument("FourD decluster=True currently requires scan_bin=1.");
         if (BitDepth != 32)
             throw std::runtime_error("FourD decluster=True is currently only supported for bitdepth=32.");
     }
     // Pixel-triggered (smart-scan pattern_file) acquisitions always use the
     // real-row-tracked chunk-flush path (enable_FourD_pixeltrig_flush) regardless
     // of decluster, since the ordinary raw-decode-progress flush trigger assumes
     // trigger order tracks real scan row 1:1 -- true for a raster scan, false for
     // a sparse custom pattern. That tracking (like the declustered path it mirrors)
     // assumes scan_bin=1.
     if (camera == CAMERA::CHEETAH_PIXELTRIG && scan_bin != 1)
         throw std::invalid_argument("FourD with a pattern_file (smart-scan) currently requires scan_bin=1.");
     // RAW (decluster=False) pixel-triggered FourD previously had a confirmed,
     // severe completeness bug (21% of scan positions undercounted, a 7% total-
     // event shortfall) that was blocked behind a runtime guard here rather than
     // shipped broken. Root-caused and fixed: process_data()'s regular raster-
     // progress-driven flush trigger (further down, guarded by !decluster) was
     // NOT ALSO excluding CHEETAH_PIXELTRIG, so it fired independently of and
     // redundantly with advance_fourD_pixeltrig_chunk's own dedicated flush
     // mechanism (Cheetah_pixeltrig.hpp) -- both drove the same shared
     // write_and_clean()/next_chunk_id counter, racing and corrupting each
     // other's chunk bookkeeping. Fixed by extending that trigger's existing
     // camera exclusion (see its own comment). A separate, small residual (0.12%,
     // later measured at 0.054%) also existed on the DECLUSTERED path specifically
     // -- see declustered_chunk_data's declaration in FourD.h: that one has since
     // been eliminated entirely (every chunk-group now gets a permanent buffer
     // slot for the whole run, flushed only once decluster_thread has fully
     // joined, instead of a small fixed-size round-robin window that could drop a
     // late cluster's 4D voxel credit).

     // Run camera dependent pipeline
     switch (camera)
     {
         case CAMERA::ADVAPIX:
         {   
            using namespace ADVAPIX_ADDITIONAL;
            ADVAPIX<EVENT, BUFFER_SIZE, N_BUFFER> cam(
                nx, 
                ny,  
                dt,
                &b_cumulative,
                rep,
                processor_line,
                preprocessor_line,
                mode,
                file_path,
                socket
            );
           
            cam.enable_FourD(&Dose_image, &chunk_data, det_bin,scan_bin, chunksize,mtx);
            cam.run();
            process_data();
            cam.terminate();
            break;
        }
        case CAMERA::CHEETAH:
        {
            using namespace CHEETAH_ADDITIONAL;
            CHEETAH<EVENT, BUFFER_SIZE, N_BUFFER> cam(
                nx, 
                ny, 
                dt,
                &b_cumulative,
                rep,
                processor_line,
                preprocessor_line,
                mode,
                file_path,
                socket
            );

               
            if (decluster)
            {
                // if constexpr: enable_FourD_declustered only has a uint32_t chunk_data
                // overload (see Timepix.hpp) -- this branch must not be compiled at all
                // for FourD<8>/FourD<16> (only skipped at runtime), or those
                // instantiations would fail to compile for lack of a matching overload.
                // decluster=True is already validated above to require BitDepth==32
                // before any code can reach here, so this is not a behavior restriction,
                // just what makes the template compile for the other two instantiations.
                if constexpr (BitDepth == 32)
                {
                    allocate_declustered_chunks();
                    cam.enable_FourD_declustered(dtime, dspace, cluster_range, tot_per_electron,
                        &Dose_image, &declustered_chunk_data, &declustered_chunk_mtx, det_bin, chunksize, n_threads,
                        &clustersize_histogram, &energy_histogram, &clustersize_tot_histogram,
                        electron_count_lut.empty() ? nullptr : &electron_count_lut);
                }
            }
            else
            {
                cam.enable_FourD(&Dose_image, &chunk_data, det_bin,scan_bin, chunksize,mtx);
            }
            cam.run();
            process_data();
            cam.terminate();
            if constexpr (BitDepth == 32)
            {
                if (decluster) flush_declustered_chunks();
            }
            break;
        }
        case CAMERA::ELECTRON:
        {
            using namespace ELECTRON_ADDITIONAL;
            ELECTRON<EVENT, BUFFER_SIZE, N_BUFFER> cam(
                nx,
                ny,
                n_cam,
                &b_cumulative,
                rep,
                processor_line,
                preprocessor_line,
                mode,
                file_path,
                socket
            );
            cam.enable_FourD(&Dose_image, &chunk_data, det_bin,scan_bin, chunksize,mtx);
            cam.run();
            process_data();
            cam.terminate();
            break;
        }
        case CAMERA::CHEETAH_PIXELTRIG:
        {
            using namespace CHEETAH_ADDITIONAL;
            CHEETAH_pixeltrig<EVENT, BUFFER_SIZE, N_BUFFER> cam(
                nx,
                ny,
                &b_cumulative,
                rep,
                processor_line,
                preprocessor_line,
                mode,
                file_path,
                socket,
                pattern_file
            );

            if (decluster)
            {
                // Same if constexpr rationale as the CHEETAH case above:
                // enable_FourD_declustered only has a uint32_t overload, and
                // decluster=True is already validated above to require BitDepth==32
                // before reaching here.
                if constexpr (BitDepth == 32)
                {
                    allocate_declustered_chunks();
                    cam.enable_FourD_declustered(dtime, dspace, cluster_range, tot_per_electron,
                        &Dose_image, &declustered_chunk_data, &declustered_chunk_mtx, det_bin, chunksize, n_threads,
                        &clustersize_histogram, &energy_histogram, &clustersize_tot_histogram,
                        electron_count_lut.empty() ? nullptr : &electron_count_lut);
                }
            }
            else
            {
                cam.enable_FourD(&Dose_image, &chunk_data, det_bin, scan_bin, chunksize, mtx);
                // Pixel-triggered raw FourD needs its own exact chunk buffering --
                // line_processor()'s raw-decode-progress trigger (used by raster
                // mode's own count_chunked_N writes into chunk_data's 2-slot
                // buffer) assumes trigger order tracks real scan row 1:1, which a
                // sparse custom pattern violates. The pixel-trigger-specific flush
                // mechanism this used to fall back on (advance_fourD_pixeltrig_chunk,
                // Cheetah_pixeltrig.hpp) had the same eviction hazard
                // allocate_declustered_chunks() above fixes for the declustered
                // path: an event whose own chip lagged the aggregate flush trigger
                // could target an already-flushed, recycled chunk-group and lose
                // its 4D voxel write. Same fix, generalized across bitdepths: every
                // chunk-group gets its own permanent slot for the whole run.
                allocate_full_chunk_buffer();
                cam.enable_FourD_pixeltrig_flush(&full_chunk_data, &full_chunk_mtx);
            }
            cam.run();
            process_data();
            cam.terminate();
            if constexpr (BitDepth == 32)
            {
                if (decluster) flush_declustered_chunks();
            }
            if (!decluster) flush_full_chunk_buffer();
            break;
        }
        case CAMERA::MERLIN:
        {
            if (n_cam == 512){ 
                using namespace MERLIN_512_U8;
                MERLIN<N_CAM,BUFFER_SIZE,HEAD_SIZE,N_BUFFER,PIXEL> cam(
                    nx, 
                    ny, 
                    &b_cumulative,
                    rep,
                    processor_line,
                    preprocessor_line,
                    mode,
                    file_path,
                    socket
                );
                cam.enable_compress(&Dose_image,&chunk_data, chunksize,det_bin,mtx);
                cam.run();
                process_data();
                cam.terminate();
                break;
            }
            else if (n_cam == 256){
                using namespace MERLIN_256_U8;
                MERLIN<N_CAM,BUFFER_SIZE,HEAD_SIZE,N_BUFFER,PIXEL> cam(
                    nx, 
                    ny, 
                    &b_cumulative,
                    rep,
                    processor_line,
                    preprocessor_line,
                    mode,
                    file_path,
                    socket
                );
                cam.enable_compress(&Dose_image,&chunk_data,chunksize,det_bin,mtx);
                cam.run();
                process_data();
                cam.terminate();
                break;
            }
            else std::runtime_error("Invalid detector size for Merlin");
            break;
        }
    }
    rc_quit = true;
    save_dose_image();
    if (format == "hdf5") h5file.close();
}

template<int BitDepth>
void FourD<BitDepth>::allocate_chunk()
{
    diff_pattern_size = n_cam/det_bin*n_cam/det_bin;
    diff_pattern_length = n_cam/det_bin;
    double buffer_bits = chunksize/scan_bin*nx/scan_bin*diff_pattern_size*bitdepth;
    double ondisk_chunk_bits = chunksize/scan_bin*chunksize_x/scan_bin*diff_pattern_size*bitdepth;
    std::cout << "Accumulation buffer size (full scan row x " << chunksize << " lines): "
               << buffer_bits/8000000. << "MB" << std::endl;
    std::cout << "On-disk chunk size (" << chunksize << "x" << chunksize_x << " scan block): "
               << ondisk_chunk_bits/8000000. << "MB" << std::endl;

    chunk_data[0].assign(chunksize/scan_bin*nx/scan_bin*diff_pattern_size,0);
    chunk_data[1].assign(chunksize/scan_bin*nx/scan_bin*diff_pattern_size,0);

}

template<int BitDepth>
void FourD<BitDepth>::reset()
{
    rc_quit = false;
    fr_freq = 0;

    // Initializations
    nxy = nx * ny;
    id_image = 0;
    fr_total = nxy * rep;
    fr_count = 0;

    // Allocate memory for image arrays
    if (b_first_run) {
        Dose_image.assign(nxy/(scan_bin*scan_bin), 0);
        b_first_run = false;
    }

    next_chunk_id = 0;

    // Data Processing Progress
    *processor_line = 0;
    *preprocessor_line = 0;

}


template<int BitDepth>
void FourD<BitDepth>::line_processor(
    size_t &img_num,
    size_t &first_frame,
    size_t &end_frame,
    ProgressMonitor *prog_mon,
    size_t &fr_total_u,
    BoundedThreadPool *pool
)
{
    int idxx = 0;
    // process newly finished lines, if there are any
    if ((int)(prog_mon->fr_count / nx) < *preprocessor_line)
    {
        *processor_line = (int)(prog_mon->fr_count) / nx;
        if (*processor_line%ny==0)
            id_image = *processor_line / ny % 2;
        idxx = (int)(prog_mon->fr_count) % nxy;
        *prog_mon += nx;

        // !decluster: when decluster=True, flush timing is driven entirely by the
        // declustered cluster callback itself (Timepix.hpp's
        // enable_FourD_declustered), not by this raw-decode-progress trigger --
        // see that function's comment for why (declustering can lag decode
        // enough that this trigger firing would flush/recycle a buffer slot
        // before slower declustering has finished writing into it).
        //
        // camera != CHEETAH_PIXELTRIG: the REAL root cause of the confirmed
        // ~7% event loss / ~21% undercounted-positions bug for raw (non-
        // decluster) pixel-trigger FourD -- found via
        // measure_pixeltrig_drift.ipynb after two red herrings (chip-to-chip
        // trigger drift was tiny, and a from-scratch rewrite of the eviction-
        // timing formula changed nothing). This trigger and
        // advance_fourD_pixeltrig_chunk's dedicated one (Cheetah_pixeltrig.hpp)
        // were BOTH active at once for pixel-trigger mode -- *processor_line
        // here is fed by the same raw-decode-progress tracker the comment
        // above already calls out as unreliable for a custom scan pattern
        // (assumes trigger order tracks real scan row 1:1). Both triggers call
        // write_and_clean(), which advances its own single, shared sequential
        // next_chunk_id counter -- so the two trackers raced and corrupted
        // each other's bookkeeping, flushing chunks out of step with what
        // advance_fourD_pixeltrig_chunk's own (correct) accounting expected.
        // Pixel-trigger mode must rely SOLELY on its own dedicated mechanism,
        // exactly like the decluster case already does, for exactly the same
        // reason (a second, independently-timed trigger touching the same
        // shared counter is unsafe) -- this is that same exclusion, extended.
        if (!decluster && camera != CAMERA::CHEETAH_PIXELTRIG && b_save_4D && *processor_line%chunksize == 0 && *processor_line > 0){
            write_and_clean((int)(chunksize/scan_bin));
        }

        progress_percent = prog_mon->progress_percent;


        int update_line = idxx / nx; 
        if ((prog_mon->report_set) && (update_line)>0)
        {
            fr_freq = prog_mon->fr_freq;
            prog_mon->reset_flags();
        }
    }

    // end of image handler
    if (prog_mon->fr_count >= end_frame)
    {
        if (b_continuous) {
            prog_mon->fr_total += nxy;
            fr_total_u += nxy;
        }
        if (prog_mon->fr_count != fr_total_u)
        {
            img_num++;
            first_frame = img_num * nxy;
            end_frame = (img_num + 1) * nxy;
        }

        if (!decluster && camera != CAMERA::CHEETAH_PIXELTRIG && b_save_4D){
            // camera != CHEETAH_PIXELTRIG: same exclusion and reasoning as the
            // regular per-chunksize trigger above -- pixel-trigger mode
            // (decluster or not) already gets its own final-group flush via
            // finalize_fourD_declustered(), called explicitly after
            // cam.terminate() in run(). Letting this ALSO fire raced it
            // against that dedicated mechanism's own bookkeeping.
            //
            // This flush is NOT redundant with the regular per-chunksize trigger
            // above, even when ny is an exact multiple of chunksize -- empirically
            // confirmed (fourd_conversion.ipynb) that the regular trigger's guard,
            // `(fr_count/nx) < *preprocessor_line`, never becomes true for the
            // scan's true final line: preprocessor_line stops advancing once
            // decode finishes (there is no further line to detect), so by the time
            // fr_count's own line-equivalent catches up to that same final value,
            // the strict "<" never holds for that last transition. So the final
            // chunk group -- full or partial -- is only ever flushed here.
            //
            // When decluster=True, the equivalent final-group flush instead
            // happens via Timepix.hpp's finalize_fourD_declustered(), called
            // explicitly after cam.terminate() in run() -- this raw-progress
            // trigger is skipped entirely for that mode (see the note on the
            // per-chunksize trigger above).
            int total_rows = (int)(ny/scan_bin);
            int rows_per_chunk = (int)(chunksize/scan_bin);
            int remaining_rows = total_rows % rows_per_chunk;
            int rows_to_flush = (remaining_rows > 0) ? remaining_rows : rows_per_chunk;
            write_and_clean(rows_to_flush);
        }
    }

    // end of recon handler
    if (((prog_mon->fr_count >= fr_total_u) && (!b_continuous)) || rc_quit)
    {
        pool->wait_for_completion();
        p_prog_mon = nullptr;
        b_cumulative = false;
        b_continuous = false;
        *processor_line = -1;
    }
}

template<int BitDepth>
void FourD<BitDepth>::write_and_clean(int n_valid_rows)
{
    int chunk_id = next_chunk_id++;
    int mod_chunk_id = chunk_id%2;
    std::lock_guard<std::mutex> lock(mtx[mod_chunk_id]);
    write_chunk(chunk_data[mod_chunk_id], (hsize_t)chunk_id, (hsize_t)n_valid_rows);
    chunk_data[mod_chunk_id].assign(chunksize/scan_bin*nx/scan_bin*diff_pattern_size,0);

}

// Declustered 4D conversion only (scan_bin=1 always, enforced in run() above) --
// see declustered_chunk_data's declaration in FourD.h for why every chunk-group
// gets its own permanent slot here instead of reusing chunk_data's 2-slot
// round-robin buffer. Called once, right before cam.enable_FourD_declustered(),
// while ny/chunksize/diff_pattern_size are already known.
template<int BitDepth>
void FourD<BitDepth>::allocate_declustered_chunks()
{
    int n_chunks_total = (int)(((size_t)ny + chunksize - 1) / chunksize);
    // Each slot must hold a FULL chunk-group -- chunksize rows x nx columns x
    // diff_pattern_size detector pixels -- not just one row (scan_bin=1 is always
    // enforced for the declustered path, matching allocate_chunk()'s own identical
    // formula for chunk_data above).
    declustered_chunk_data.assign(n_chunks_total, std::vector<uint32_t>(chunksize*nx*diff_pattern_size, 0));
    declustered_chunk_mtx = std::vector<std::mutex>(n_chunks_total);
}

// Writes every chunk-group to disk, in order, in one pass -- only called after
// cam.terminate() guarantees decluster_thread has fully joined, so every cluster
// in the file has already been resolved and credited into declustered_chunk_data.
// No recycling, no "already flushed" case: each slot is written here exactly once.
template<int BitDepth>
void FourD<BitDepth>::flush_declustered_chunks()
{
    int n_chunks_total = (int)declustered_chunk_data.size();
    int total_rows = (int)ny;
    for (int i = 0; i < n_chunks_total; ++i)
    {
        int remaining = total_rows - i * (int)chunksize;
        int n_valid_rows = std::min(remaining, (int)chunksize);
        write_chunk(declustered_chunk_data[i], (hsize_t)i, (hsize_t)n_valid_rows);
    }
    declustered_chunk_data.clear();
    declustered_chunk_data.shrink_to_fit();
}

// Raw (non-declustered) pixel-triggered FourD counterpart to
// allocate_declustered_chunks() above -- see full_chunk_data's declaration in
// FourD.h for why. Bitdepth-generic via ChunkElemT (uint8_t/16/32).
template<int BitDepth>
void FourD<BitDepth>::allocate_full_chunk_buffer()
{
    int n_chunks_total = (int)(((size_t)ny + chunksize - 1) / chunksize);
    full_chunk_data.assign(n_chunks_total, std::vector<ChunkElemT>(chunksize*nx*diff_pattern_size, 0));
    full_chunk_mtx = std::vector<std::mutex>(n_chunks_total);
}

// Counterpart to flush_declustered_chunks() above, for full_chunk_data.
template<int BitDepth>
void FourD<BitDepth>::flush_full_chunk_buffer()
{
    int n_chunks_total = (int)full_chunk_data.size();
    int total_rows = (int)ny;
    for (int i = 0; i < n_chunks_total; ++i)
    {
        int remaining = total_rows - i * (int)chunksize;
        int n_valid_rows = std::min(remaining, (int)chunksize);
        write_chunk(full_chunk_data[i], (hsize_t)i, (hsize_t)n_valid_rows);
    }
    full_chunk_data.clear();
    full_chunk_data.shrink_to_fit();
}

template<int BitDepth>
void FourD<BitDepth>::init_4D_file_32() {

    if (format == "zarr") { init_4D_file_zarr("<u4"); return; }

    if (save_metadata)
    {
        const char* __DATASET_NAME = "shape";
        hsize_t __dims[1] = {4};
        H5::DataSpace __dataspace(1, __dims);
        H5::DataSet __dataset = h5file.createDataSet(__DATASET_NAME,H5::PredType::NATIVE_UINT16, __dataspace);
        // Buffer type must actually match the NATIVE_UINT16 predtype declared below --
        // this used to be `hsize_t shape[4]` (8-byte elements) written with a 2-byte
        // predtype, so HDF5 read the memory as four uint16_t's starting from the first
        // hsize_t's low bytes: the first value came through by coincidence (small
        // enough to fit in 16 bits, little-endian), the other three always read back as
        // 0 regardless of the real dimensions -- a real, independent bug from the
        // axis-order one below, only found once this "shape" dataset was actually
        // checked against the real dataset's shape (see fourd_conversion.ipynb).
        uint16_t shape[4] = {
            (uint16_t)(ny/scan_bin), (uint16_t)(nx/scan_bin),
            (uint16_t)diff_pattern_length, (uint16_t)diff_pattern_length};
        __dataset.write(shape, H5::PredType::NATIVE_UINT16);
    }

    b_save_4D = true;

    const char* DATASET_NAME_4 = "4D";

    // Axis order fixed to match how the data is actually written: axis 0 advances
    // by `chunksize` rows every flush (the scan-Y/line axis), axis 1 is scan-X. This
    // used to be swapped here (nx/ny) relative to both the true write order AND the
    // "shape" dataset above (which already used the correct ny,nx order) -- silent
    // for the square scans (nx==ny) tested so far, but would mislabel a non-square
    // scan's data.
    hsize_t dims_4[4] = {static_cast<hsize_t>(ny/scan_bin),static_cast<hsize_t>(nx/scan_bin),diff_pattern_length,diff_pattern_length};
    H5::DataSpace dataspace_4(4, dims_4);

    H5::DSetCreatPropList prop_4;
    // Chunk shape: chunksize_x replaces the old full scan-row-width chunk (was
    // nx/scan_bin here), so a small-ROI-across-several-lines read only has to touch
    // chunks that actually overlap the ROI instead of every full-width row group.
    // Detector dimensions stay full-extent per chunk (confirmed normal read wants
    // the whole diffraction pattern per scan point).
    hsize_t chunk_dims_4[4] = {chunksize/scan_bin,chunksize_x/scan_bin,diff_pattern_length, diff_pattern_length};
    prop_4.setChunk(4, chunk_dims_4);
    prop_4.setDeflate(deflate_factor); // 0-9, 9 is maximum compression (slower), 0 is no compression
    dataset4D = h5file.createDataSet(DATASET_NAME_4,H5::PredType::NATIVE_UINT32, dataspace_4,prop_4);
    }


template<int BitDepth>
void FourD<BitDepth>::init_4D_file_16() {

    if (format == "zarr") { init_4D_file_zarr("<u2"); return; }

    if (save_metadata)
    {
        const char* __DATASET_NAME = "shape";
        hsize_t __dims[1] = {4};
        H5::DataSpace __dataspace(1, __dims);
        H5::DataSet __dataset = h5file.createDataSet(__DATASET_NAME,H5::PredType::NATIVE_UINT16, __dataspace);
        // Buffer type must actually match the NATIVE_UINT16 predtype declared below --
        // this used to be `hsize_t shape[4]` (8-byte elements) written with a 2-byte
        // predtype, so HDF5 read the memory as four uint16_t's starting from the first
        // hsize_t's low bytes: the first value came through by coincidence (small
        // enough to fit in 16 bits, little-endian), the other three always read back as
        // 0 regardless of the real dimensions -- a real, independent bug from the
        // axis-order one below, only found once this "shape" dataset was actually
        // checked against the real dataset's shape (see fourd_conversion.ipynb).
        uint16_t shape[4] = {
            (uint16_t)(ny/scan_bin), (uint16_t)(nx/scan_bin),
            (uint16_t)diff_pattern_length, (uint16_t)diff_pattern_length};
        __dataset.write(shape, H5::PredType::NATIVE_UINT16);
    }

    b_save_4D = true;

    const char* DATASET_NAME_4 = "4D";

    // See init_4D_file_32 for why the axis order and chunk shape changed here.
    hsize_t dims_4[4] = {static_cast<hsize_t>(ny/scan_bin),static_cast<hsize_t>(nx/scan_bin),diff_pattern_length,diff_pattern_length};
    H5::DataSpace dataspace_4(4, dims_4);

    H5::DSetCreatPropList prop_4;
    hsize_t chunk_dims_4[4] = {chunksize/scan_bin,chunksize_x/scan_bin,diff_pattern_length, diff_pattern_length};
    prop_4.setChunk(4, chunk_dims_4);
    prop_4.setDeflate(deflate_factor); // 0-9, 9 is maximum compression (slower), 0 is no compression
    dataset4D = h5file.createDataSet(DATASET_NAME_4,H5::PredType::NATIVE_UINT16, dataspace_4,prop_4);
    }
 
template<int BitDepth>
void FourD<BitDepth>::init_4D_file_8() {

    if (format == "zarr") { init_4D_file_zarr("<u1"); return; }

    try{
        if (save_metadata)
        {
            const char* __DATASET_NAME = "shape";
            hsize_t __dims[1] = {4};
            H5::DataSpace __dataspace(1, __dims);
            H5::DataSet __dataset = h5file.createDataSet(__DATASET_NAME,H5::PredType::NATIVE_UINT16, __dataspace);
            // Same buffer-type/predtype mismatch bug fixed in init_4D_file_32/16
            // during Part 1 (hsize_t is 8 bytes, NATIVE_UINT16 is 2) -- missed here
            // at the time since this 8-bit path wasn't covered by that pass's
            // testing; caught now while touching this exact block.
            uint16_t shape[4] = {
                (uint16_t)(ny/scan_bin), (uint16_t)(nx/scan_bin),
                (uint16_t)diff_pattern_length, (uint16_t)diff_pattern_length};
            __dataset.write(shape, H5::PredType::NATIVE_UINT16);
        }

        b_save_4D = true;

        const char* DATASET_NAME_4 = "4D";

        // See init_4D_file_32 for why the axis order and chunk shape changed here.
        hsize_t dims_4[4] = {static_cast<hsize_t>(ny/scan_bin),static_cast<hsize_t>(nx/scan_bin),diff_pattern_length,diff_pattern_length};
        H5::DataSpace dataspace_4(4, dims_4);

        H5::DSetCreatPropList prop_4;
        hsize_t chunk_dims_4[4] = {chunksize/scan_bin,chunksize_x/scan_bin,diff_pattern_length, diff_pattern_length};
        prop_4.setChunk(4, chunk_dims_4);
        prop_4.setDeflate(deflate_factor); // 0-9, 9 is maximum compression (slower), 0 is no compression
        dataset4D = h5file.createDataSet(DATASET_NAME_4,H5::PredType::NATIVE_UINT8, dataspace_4,prop_4);
    }
    catch (const H5::FileIException& e) {
        std::cerr << "File error: " << e.getDetailMsg() << std::endl;
        throw;} 
    catch (const H5::DataSetIException& e) {
        std::cerr << "Dataset error: " << e.getDetailMsg() << std::endl;
        throw;} 
    catch (const H5::DataSpaceIException& e) {
        std::cerr << "Dataspace error: " << e.getDetailMsg() << std::endl;
        throw;}
    catch (const H5::PropListIException& e) {
        std::cerr << "Property list error: " << e.getDetailMsg() << std::endl;
        throw;}
    catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        throw;} 
    catch (...) {
    std::cerr << "Unknown error occurred during HDF5 file initialization." << std::endl;
    throw;}
}

template<int BitDepth>
void FourD<BitDepth>::init_4D_file()
{
    if (format != "hdf5" && format != "zarr")
    {
        throw std::invalid_argument("FourD.format must be \"hdf5\" or \"zarr\", got \"" + format + "\"");
    }
    switch (bitdepth)
    {
    case 8:
        init_4D_file_8();
        break;
    case 16:
        init_4D_file_16();
        break;
    case 32:
        init_4D_file_32();
        break;
    }
}

// Zarr counterpart of init_4D_file_8/16/32's HDF5-specific dataset creation:
// creates the store's root directory + ".zgroup", and the "4D" and "shape"
// arrays' metadata (mirroring the two HDF5 datasets of the same name exactly, so
// switching `format` doesn't change what a caller reads back beyond the
// container format itself).
template<int BitDepth>
void FourD<BitDepth>::init_4D_file_zarr(const std::string& dtype_str)
{
    std::filesystem::create_directories(zarr_root);
    {
        std::ofstream f(zarr_root + "/.zgroup", std::ios::binary);
        f << "{\"zarr_format\":2}";
    }

    b_save_4D = true;

    write_zarr_array_meta(zarr_root + "/4D",
        {(long long)(ny/scan_bin), (long long)(nx/scan_bin),
         (long long)diff_pattern_length, (long long)diff_pattern_length},
        {(long long)(chunksize/scan_bin), (long long)(chunksize_x/scan_bin),
         (long long)diff_pattern_length, (long long)diff_pattern_length},
        dtype_str, deflate_factor);

    if (save_metadata)
    {
        write_zarr_array_meta(zarr_root + "/shape", {4}, {4}, "<u2", 0);
        // "shape" is 1-D (four elements), so its one chunk's index is just "0" --
        // not "0.0.0.0" (that would be a 4-D array's first chunk).
        uint16_t shape[4] = {
            (uint16_t)(ny/scan_bin), (uint16_t)(nx/scan_bin),
            (uint16_t)diff_pattern_length, (uint16_t)diff_pattern_length};
        std::ofstream f(zarr_root + "/shape/0", std::ios::binary);
        f.write(reinterpret_cast<const char*>(shape), sizeof(shape));
    }
}

// Slices the buffered [n_valid_rows][full nx width][det,det] data (buffer layout is
// always full-row-width, regardless of the on-disk chunk width -- see chunksize_x in
// FourD.h) into chunksize_x-wide scan-x blocks and writes each as one real HDF5
// chunk. This is the only place chunk shape differs from the old full-row-width
// scheme; the accumulation buffer itself, and the per-event hot path that fills it
// (Timepix.hpp's count_chunked_*), are unchanged.
//
// y_offset is derived from chunk_id * (chunksize/scan_bin), NOT from n_valid_rows --
// chunk groups are always spaced by the FULL chunksize in the dataset's Y
// coordinate, even when the final group is only partially filled (n_valid_rows <
// chunksize/scan_bin), since chunk_id counts flushes, not filled rows.
template<int BitDepth>
void FourD<BitDepth>::write_chunk(const std::vector<uint32_t>& chunk, hsize_t chunk_id, hsize_t n_valid_rows) {
    hsize_t nx_full = nx/scan_bin;
    hsize_t nx_chunk = chunksize_x/scan_bin;
    hsize_t y_offset = chunk_id * (chunksize/scan_bin);
    if (y_offset + n_valid_rows > ny/scan_bin) return; // defensive bounds guard

    hsize_t n_xblocks = (nx_full + nx_chunk - 1) / nx_chunk;
    for (hsize_t xb = 0; xb < n_xblocks; ++xb)
    {
        hsize_t x0 = xb * nx_chunk;
        hsize_t this_nx = std::min(nx_chunk, nx_full - x0);

        std::vector<uint32_t> block(n_valid_rows * this_nx * diff_pattern_size);
        for (hsize_t ly = 0; ly < n_valid_rows; ++ly)
        {
            const uint32_t* src = chunk.data() + (ly * nx_full + x0) * diff_pattern_size;
            uint32_t* dst = block.data() + (ly * this_nx) * diff_pattern_size;
            std::copy(src, src + this_nx * diff_pattern_size, dst);
        }

        if (format == "zarr")
        {
            write_zarr_4D_chunk<uint32_t>(zarr_root + "/4D", block.data(), n_valid_rows, this_nx,
                chunksize/scan_bin, chunksize_x/scan_bin, diff_pattern_length,
                (long long)chunk_id, (long long)xb, deflate_factor);
            continue;
        }

        hsize_t offset4D[4] = { y_offset, x0, 0, 0 };
        hsize_t block_dims4D[4] = { n_valid_rows, this_nx, diff_pattern_length, diff_pattern_length };
        H5::DataSpace filespace4D = dataset4D.getSpace();
        filespace4D.selectHyperslab(H5S_SELECT_SET, block_dims4D, offset4D);
        H5::DataSpace memspace4D(4, block_dims4D);
        dataset4D.write(block.data(), H5::PredType::NATIVE_UINT32, memspace4D, filespace4D);
    }
}

template<int BitDepth>
void FourD<BitDepth>::write_chunk(const std::vector<uint16_t>& chunk, hsize_t chunk_id, hsize_t n_valid_rows) {
    hsize_t nx_full = nx/scan_bin;
    hsize_t nx_chunk = chunksize_x/scan_bin;
    hsize_t y_offset = chunk_id * (chunksize/scan_bin);
    if (y_offset + n_valid_rows > ny/scan_bin) return;

    hsize_t n_xblocks = (nx_full + nx_chunk - 1) / nx_chunk;
    for (hsize_t xb = 0; xb < n_xblocks; ++xb)
    {
        hsize_t x0 = xb * nx_chunk;
        hsize_t this_nx = std::min(nx_chunk, nx_full - x0);

        std::vector<uint16_t> block(n_valid_rows * this_nx * diff_pattern_size);
        for (hsize_t ly = 0; ly < n_valid_rows; ++ly)
        {
            const uint16_t* src = chunk.data() + (ly * nx_full + x0) * diff_pattern_size;
            uint16_t* dst = block.data() + (ly * this_nx) * diff_pattern_size;
            std::copy(src, src + this_nx * diff_pattern_size, dst);
        }

        if (format == "zarr")
        {
            write_zarr_4D_chunk<uint16_t>(zarr_root + "/4D", block.data(), n_valid_rows, this_nx,
                chunksize/scan_bin, chunksize_x/scan_bin, diff_pattern_length,
                (long long)chunk_id, (long long)xb, deflate_factor);
            continue;
        }

        hsize_t offset4D[4] = { y_offset, x0, 0, 0 };
        hsize_t block_dims4D[4] = { n_valid_rows, this_nx, diff_pattern_length, diff_pattern_length };
        H5::DataSpace filespace4D = dataset4D.getSpace();
        filespace4D.selectHyperslab(H5S_SELECT_SET, block_dims4D, offset4D);
        H5::DataSpace memspace4D(4, block_dims4D);
        dataset4D.write(block.data(), H5::PredType::NATIVE_UINT16, memspace4D, filespace4D);
    }
}

template<int BitDepth>
void FourD<BitDepth>::write_chunk(const std::vector<uint8_t>& chunk, hsize_t chunk_id, hsize_t n_valid_rows) {
    hsize_t nx_full = nx/scan_bin;
    hsize_t nx_chunk = chunksize_x/scan_bin;
    hsize_t y_offset = chunk_id * (chunksize/scan_bin);
    if (y_offset + n_valid_rows > ny/scan_bin) return;

    hsize_t n_xblocks = (nx_full + nx_chunk - 1) / nx_chunk;
    for (hsize_t xb = 0; xb < n_xblocks; ++xb)
    {
        hsize_t x0 = xb * nx_chunk;
        hsize_t this_nx = std::min(nx_chunk, nx_full - x0);

        std::vector<uint8_t> block(n_valid_rows * this_nx * diff_pattern_size);
        for (hsize_t ly = 0; ly < n_valid_rows; ++ly)
        {
            const uint8_t* src = chunk.data() + (ly * nx_full + x0) * diff_pattern_size;
            uint8_t* dst = block.data() + (ly * this_nx) * diff_pattern_size;
            std::copy(src, src + this_nx * diff_pattern_size, dst);
        }

        if (format == "zarr")
        {
            write_zarr_4D_chunk<uint8_t>(zarr_root + "/4D", block.data(), n_valid_rows, this_nx,
                chunksize/scan_bin, chunksize_x/scan_bin, diff_pattern_length,
                (long long)chunk_id, (long long)xb, deflate_factor);
            continue;
        }

        hsize_t offset4D[4] = { y_offset, x0, 0, 0 };
        hsize_t block_dims4D[4] = { n_valid_rows, this_nx, diff_pattern_length, diff_pattern_length };
        H5::DataSpace filespace4D = dataset4D.getSpace();
        filespace4D.selectHyperslab(H5S_SELECT_SET, block_dims4D, offset4D);
        H5::DataSpace memspace4D(4, block_dims4D);
        dataset4D.write(block.data(), H5::PredType::NATIVE_UINT8, memspace4D, filespace4D);
    }
}

template<int BitDepth>
void FourD<BitDepth>::save_dose_image(){

    // Dose_image is always computed and available via the .Dose_image Python
    // property regardless -- save_metadata=false only suppresses writing it to
    // disk, so the output file/store contains only "4D" (see FourD.h's comment on
    // save_metadata: single-array containers are what generic readers, e.g.
    // hyperspy's hs.load(fn, lazy=True), expect).
    if (!save_metadata) return;

    if (format == "zarr") { save_dose_image_zarr(); return; }

    const char*  DATASET_NAME = "dose_image";

    hsize_t dims[2] = {nx/scan_bin , ny/scan_bin};
    H5::DataSpace dataspace(2, dims);
    H5::DataSet dataset_ = h5file.createDataSet(DATASET_NAME, H5::PredType::NATIVE_UINT64, dataspace);
    dataset_.write(Dose_image.data(), H5::PredType::NATIVE_UINT64);
}

template<int BitDepth>
void FourD<BitDepth>::save_dose_image_zarr(){
    // Covered by init_4D_file_zarr()'s ".zgroup"/directory creation when that was
    // called first (the normal case), but save_dose_image() always runs at the end
    // of run() even if init_4D_file() never was -- so make sure the store root
    // exists here too rather than assuming it does.
    std::filesystem::create_directories(zarr_root);
    {
        std::ofstream f(zarr_root + "/.zgroup", std::ios::binary);
        f << "{\"zarr_format\":2}";
    }
    write_zarr_array_meta(zarr_root + "/dose_image",
        {(long long)(nx/scan_bin), (long long)(ny/scan_bin)},
        {(long long)(nx/scan_bin), (long long)(ny/scan_bin)},
        "<u8", 0);
    std::ofstream f(zarr_root + "/dose_image/0.0", std::ios::binary);
    f.write(reinterpret_cast<const char*>(Dose_image.data()), (std::streamsize)(Dose_image.size() * sizeof(uint64_t)));
}


template class FourD<8>;
template class FourD<16>;
template class FourD<32>;