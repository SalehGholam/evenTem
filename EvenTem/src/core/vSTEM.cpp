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

#include "vSTEM.h"

void vSTEM::run(){
    py::gil_scoped_release release;
    reset();

    if (decluster)
    {
        if (camera != CAMERA::CHEETAH && camera != CAMERA::CHEETAH_PIXELTRIG)
            throw std::runtime_error("vSTEM decluster=True is currently only supported for .tpx3 (CHEETAH) files.");
        // electron_count_lut_file (a saved 2D map, see ClusterResolver.hpp) is an
        // alternative to tot_per_electron's single ToT/ratio formula -- loading it
        // here means electron_count_lut is populated before the tot_per_electron
        // check below, so a run relying solely on the map (tot_per_electron left at
        // its default 0.0) is not rejected.
        if (!electron_count_lut_file.empty())
            electron_count_lut = load_electron_count_lut_file(electron_count_lut_file);
        if (tot_per_electron <= 0.0 && electron_count_lut.empty())
            throw std::invalid_argument("vSTEM.tot_per_electron must be set (calibrated from your own cluster ToT-sum histogram), or electron_count_lut_file/electron_count_lut must be set, before enabling decluster=True.");
    }

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
            if (use_mask) cam.enable_mask_vSTEM(&detector_mask,&vSTEM_stack);
            else if (detector.n_detectors > 1)
            {
                cam.enable_multi_vSTEM(&detector.radia_sqr,&offsets,&vSTEM_stack);
            }
            else  cam.enable_vSTEM(&detector.radia_sqr[0],&offsets[0],&vSTEM_stack);
            // else  cam.enable_atomic_vSTEM(&detector.radia_sqr[0],&offsets[0],&atomic_vSTEM_image);
            cam.run();
            startime = std::chrono::high_resolution_clock::now();
            process_data();
            cam.terminate();
            processing_rate = cam.get_processing_rate();
            // from_atomic();
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
            // Multi-process file-splitting (see vSTEM.h) -- defaults (0, 0, -1)
            // reproduce the exact old whole-file behavior when unset.
            cam.file_byte_offset = this->file_byte_offset;
            cam.line_number_offset = this->line_number_offset;
            cam.stop_at_line = this->stop_at_line;
            cam.seed_dt = this->seed_dt;
            cam.seed_rise_t = this->seed_rise_t;
            cam.seed_rise_fall = this->seed_rise_fall;
            cam.seed_line_count = this->seed_line_count;
            cam.seed_chip_id = this->seed_chip_id;
            if (decluster)
            {
                cam.enable_vSTEM_declustered(dtime,dspace,cluster_range,tot_per_electron,
                    &detector.radia_sqr[0],&offsets[0],&vSTEM_stack,n_threads,
                    &clustersize_histogram,&energy_histogram,&clustersize_tot_histogram,
                    electron_count_lut.empty() ? nullptr : &electron_count_lut);
            }
            else if (use_mask) cam.enable_mask_vSTEM(&detector_mask,&vSTEM_stack);
            else if (detector.n_detectors > 1)
            {
                cam.enable_multi_vSTEM(&detector.radia_sqr,&offsets,&vSTEM_stack);
            }
            else  cam.enable_vSTEM(&detector.radia_sqr[0],&offsets[0],&vSTEM_stack);
            cam.run();
            process_data();
            cam.terminate();
            if (decluster || line_number_offset > 0 || stop_at_line >= 0)
            {
                // line_processor() above copies vSTEM_stack into vSTEM_image
                // incrementally, one full row at a time, but only once the
                // AGGREGATE decode progress (*preprocessor_line, driven by the
                // slowest chip) has advanced PAST that row -- used as the "this
                // row is now complete" signal. Two situations break that signal:
                //
                // - decluster: clusters are resolved and credited into
                //   vSTEM_stack asynchronously, on a separate, slower thread
                //   that lags behind decode. By the time any cluster is
                //   actually credited, line_processor has typically already
                //   copied still-zero vSTEM_stack data into vSTEM_image for
                //   every line, and never revisits it -- so vSTEM_image
                //   silently ends up all zero.
                // - multi-process file-splitting (line_number_offset/
                //   stop_at_line): a worker's own decode deliberately STOPS
                //   at exactly its boundary row (that's what stop_at_line
                //   means) -- so the aggregate progress marker never crosses
                //   PAST that row within this worker, and the incremental
                //   copy never fires for it, even though some chips (the ones
                //   already ahead of the slowest one at the split point) have
                //   legitimately already written real hits there via vstem().
                //   Confirmed via mp_vstem.ipynb: this is what was silently
                //   discarding a worker's own share of its boundary row,
                //   masquerading as a "small residual" until traced here.
                //
                // cam.terminate() above guarantees all decode/decluster work
                // has now fully finished writing into vSTEM_stack, so redo the
                // copy once, authoritatively, from the now-complete data
                // (summing completed repetitions the same way b_cumulative
                // accumulation would) -- this naturally covers every row this
                // worker actually touched, partial boundary rows included.
                std::fill(vSTEM_image.begin(), vSTEM_image.end(), 0);
                for (int r = 0; r < rep; ++r)
                    for (int i = 0; i < nxy; ++i)
                        vSTEM_image[i] += vSTEM_stack[r][i];
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
            if (detector.n_detectors > 1)
            {
                cam.enable_multi_vSTEM(&detector.radia_sqr,&offsets,&vSTEM_stack);
            }
            else  cam.enable_vSTEM(&detector.radia_sqr[0],&offsets[0],&vSTEM_stack);
            // else  cam.enable_atomic_vSTEM(&detector.radia_sqr[0],&offsets[0],&atomic_vSTEM_image);
            cam.run();
            process_data();
            cam.terminate();
            // from_atomic();
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
            // Multi-process file-splitting (see vSTEM.h) -- defaults (0, 0,
            // -1, all-(-1)) reproduce the exact old whole-file behavior when
            // unset.
            cam.file_byte_offset = this->file_byte_offset;
            cam.line_number_offset = this->line_number_offset;
            cam.stop_at_line = this->stop_at_line;
            cam.seed_rise_fall = this->seed_rise_fall;
            cam.seed_probe_count_chip = this->seed_probe_count_chip;
            cam.seed_chip_id = this->seed_chip_id;
            if (decluster)
            {
                cam.enable_vSTEM_declustered(dtime,dspace,cluster_range,tot_per_electron,
                    &detector.radia_sqr[0],&offsets[0],&vSTEM_stack,n_threads,
                    &clustersize_histogram,&energy_histogram,&clustersize_tot_histogram,
                    electron_count_lut.empty() ? nullptr : &electron_count_lut);
            }
            else if (detector.n_detectors > 1)
            {
                cam.enable_multi_vSTEM(&detector.radia_sqr,&offsets,&vSTEM_stack);
            }
            else  cam.enable_vSTEM(&detector.radia_sqr[0],&offsets[0],&vSTEM_stack);
            cam.run();
            process_data();
            cam.terminate();
            if (decluster || line_number_offset > 0 || stop_at_line >= 0)
            {
                // Same staging-buffer race as the CHEETAH case above, same fix --
                // plus, for pixel-trigger specifically, a worker's own decode
                // deliberately stops at exactly its own trigger-count boundary
                // (stop_at_line, reinterpreted as a trigger count here -- see
                // vSTEM.h), so the incremental copy may not have caught up to
                // whatever vSTEM_stack it already wrote for the tail end of this
                // worker's slice. cam.terminate() guarantees decode has fully
                // finished, so redo the copy once, authoritatively, from the
                // now-complete vSTEM_stack.
                std::fill(vSTEM_image.begin(), vSTEM_image.end(), 0);
                for (int r = 0; r < rep; ++r)
                    for (int i = 0; i < nxy; ++i)
                        vSTEM_image[i] += vSTEM_stack[r][i];
            }
            break;
        }
        case CAMERA::MERLIN:
        {
            if (bitdepth == 8){
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else if (n_cam == 514){
                    using namespace MERLIN_514_U8;
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else std::runtime_error("Invalid detector size for Merlin");
                break;
            }
            else if (bitdepth == 16){
                if (n_cam == 512){ 
                    using namespace MERLIN_512_U16;
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else if (n_cam == 514){
                    using namespace MERLIN_514_U16;
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else if (n_cam == 256){
                    using namespace MERLIN_256_U16;
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else std::runtime_error("Invalid detector size for Merlin");
            }
            else std::runtime_error("Invalid bitdepth for Merlin");
        }
        case CAMERA::HDF5:
        {
            using namespace HDF5_ADDITIONAL;
            HDF5<N_CAM,BUFFER_SIZE,HEAD_SIZE,N_BUFFER,PIXEL> cam(
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
            compute_detector();
            cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
            cam.run();
            process_data();
            cam.terminate();
            break;
        }
        case CAMERA::NUMPY:
        {
            if (bitdepth == 8) {
                if (n_cam == 64){ 
                    using namespace FRAME_64_U8;
                    NUMPY<N_CAM,BUFFER_SIZE,HEAD_SIZE,N_BUFFER,PIXEL> cam(
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else if (n_cam == 128){ 
                    using namespace FRAME_128_U8;
                    NUMPY<N_CAM,BUFFER_SIZE,HEAD_SIZE,N_BUFFER,PIXEL> cam(
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else if (n_cam == 192){ 
                    using namespace FRAME_192_U8;
                    NUMPY<N_CAM,BUFFER_SIZE,HEAD_SIZE,N_BUFFER,PIXEL> cam(
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else if (n_cam == 256){ 
                    using namespace FRAME_256_U8;
                    NUMPY<N_CAM,BUFFER_SIZE,HEAD_SIZE,N_BUFFER,PIXEL> cam(
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else if (n_cam == 512){ 
                    using namespace FRAME_512_U8;
                    NUMPY<N_CAM,BUFFER_SIZE,HEAD_SIZE,N_BUFFER,PIXEL> cam(
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else std::runtime_error("Invalid detector size for Numpy");
            }
            else if (bitdepth == 16) {
                if (n_cam == 64){ 
                    using namespace FRAME_64_U16;
                    NUMPY<N_CAM,BUFFER_SIZE,HEAD_SIZE,N_BUFFER,PIXEL> cam(
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else if (n_cam == 128){ 
                    using namespace FRAME_128_U16;
                    NUMPY<N_CAM,BUFFER_SIZE,HEAD_SIZE,N_BUFFER,PIXEL> cam(
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else if (n_cam == 256){ 
                    using namespace FRAME_256_U16;
                    NUMPY<N_CAM,BUFFER_SIZE,HEAD_SIZE,N_BUFFER,PIXEL> cam(
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else if (n_cam == 512){ 
                    using namespace FRAME_512_U16;
                    NUMPY<N_CAM,BUFFER_SIZE,HEAD_SIZE,N_BUFFER,PIXEL> cam(
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
                    compute_detector();
                    cam.enable_vSTEM(&detector.detector_image,&vSTEM_stack, allow_torch);
                    cam.run();
                    process_data();
                    cam.terminate();
                    break;
                }
                else std::runtime_error("Invalid detector size for Numpy");
            }
            else std::runtime_error("Invalid bitdepth for Numpy");
        }
    }
    rc_quit = true;
}

void vSTEM::reset()
{
    rc_quit = false;
    fr_freq = 0;

    std::fill(vSTEM_image.begin(), vSTEM_image.end(), 0);

    // Initializations
    nxy = nx * ny;
    id_image = 0;
    // Multi-process file-splitting (see LiveProcessor.h's fr_count_seed):
    // fr_total must be the ABSOLUTE ending line boundary for this worker's own
    // slice (stop_at_line*nx), not the slice's relative size, so it stays
    // directly comparable to *preprocessor_line (which CHEETAH always reports
    // in absolute, whole-scan terms). fr_count_seed is the matching absolute
    // starting point. Both default to the exact old whole-file values
    // (nxy*rep, 0) when stop_at_line/line_number_offset are unset.
    //
    // CHEETAH_PIXELTRIG is the one exception: stop_at_line/line_number_offset
    // are reinterpreted there as raw TRIGGER counts (see vSTEM.h), not scan
    // rows, so nx*stop_at_line is not the right formula -- current_line there
    // is only an ESTIMATE (probe_count/nx, integer division), not a true row
    // counter the way it is for raster, so it cannot represent an exact
    // single-trigger stop point (its granularity is whole nx-sized buckets).
    //
    // An earlier version of this fix tried to predict current_line's floored
    // final value (nx*(stop_at_line/nx)) and use that as fr_total, so this
    // sweep's own "done" check would line up with the real stop. That was
    // wrong: flooring means the bucket boundary is always <= the real
    // trigger target, so this sweep would (and, measured directly, actually
    // did) reach "done" and force-quit *processor_line=-1 BEFORE
    // CHEETAH_pixeltrig::process_tdc()'s own exact probe_count>=stop_at_line
    // check ever got to fire -- silently dropping every trigger between the
    // bucket floor and the real boundary. The dropped range's width (always
    // < nx, confirmed to vary run-to-run) gave it away: two independent
    // threads (this sweep's process_data(), and the camera's own decode
    // thread) were racing to be the one that stops decode first, and the
    // coarser (bucket-granularity) one usually won.
    //
    // The real fix lives in CHEETAH_pixeltrig::process_tdc() instead: it now
    // writes *p_processor_line=-1 itself, directly, the instant
    // probe_count==stop_at_line exactly -- so this sweep must never be able
    // to win that race. Keeping fr_total at the ordinary whole-file value
    // guarantees that: current_line cannot get anywhere near ny for a sparse
    // pattern (confirmed by direct measurement), so this sweep can never
    // reach "done" on its own for a split worker -- termination comes
    // entirely from the camera's own direct signal. fr_count_seed stays 0:
    // the sweep always starts from position 0 of the full nxy space, never a
    // mid-array offset (unlike raster's contiguous row-range split). The
    // authoritative post-terminate() re-copy above (triggered by
    // line_number_offset>0||stop_at_line>=0) is what actually guarantees
    // correctness for a split worker's slice, not this sweep.
    if (camera == CAMERA::CHEETAH_PIXELTRIG)
    {
        fr_total = nxy * rep;
        fr_count_seed = 0;
    }
    else
    {
        fr_total = (stop_at_line >= 0) ? (int)((size_t)nx * stop_at_line) : nxy * rep;
        fr_count_seed = (size_t)nx * line_number_offset;
    }
    fr_count = 0;

    if (auto_offset) offsets = {{(float)n_cam/2, (float)n_cam/2}};

    detector.set_radia(inner_radia, outer_radia);

    vSTEM_image.assign(nxy, 0);
    vSTEM_stack.assign(rep+1, std::vector<size_t>(nxy, 0));
    for (int i = 0; i < nxy; i++)
    {
        atomic_vSTEM_image.push_back(std::atomic<int>(0));
    }

    // Data Processing Progress
    *processor_line = 0;
    *preprocessor_line = 0;
}


void vSTEM::line_processor(
    size_t &img_num,
    size_t &first_frame,
    size_t &end_frame,
    ProgressMonitor *prog_mon,
    size_t &fr_total_u,
    BoundedThreadPool *pool
)
{
    int pp_id = 0;
    // process newly finished lines, if there are any
    if ((int)(prog_mon->fr_count / nx) < *preprocessor_line) 
    {
        *processor_line = (int)(prog_mon->fr_count) / nx;
        if (*processor_line%ny==0) id_image = *processor_line / ny;
        pp_id = (int)(prog_mon->fr_count) % nxy;

        for (size_t i = 0; i < (size_t)nx; i++)
        {
            int idxx_p_i = pp_id + i;
            if (b_cumulative) vSTEM_image[idxx_p_i] += vSTEM_stack[id_image][idxx_p_i];
            if (b_continuous) vSTEM_image[idxx_p_i] = vSTEM_stack[id_image][idxx_p_i];

        }

        *prog_mon += nx;
    }

    // Mirrors FourD::line_processor's identical copy -- see Roi.cpp's line_processor
    // for why (raw C++ console output doesn't reliably reach Jupyter; this lets
    // Python poll `.progress` from another thread instead).
    progress_percent = prog_mon->progress_percent;

    // end of line handler
    int update_line = pp_id / nx;
    if ((prog_mon->report_set) && (update_line)>0)
    {
        fr_freq = prog_mon->fr_freq;
        prog_mon->reset_flags();

        current_time = std::chrono::high_resolution_clock::now();
        elapsed_seconds = current_time - startime;
        this->reached_pp_id.push_back(pp_id);
        this->elapsed_seconds_vec.push_back(elapsed_seconds.count());
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

std::vector<int> vSTEM::compute_detector(){
    if (auto_offset) offsets = {{(float)n_cam/2, (float)n_cam/2}};
    detector.set_radia(inner_radia, outer_radia);
    detector.compute_detector(n_cam, n_cam, offsets);
    return detector.detector_image;
}

std::vector<int> vSTEM::get_detector(){
    if (!use_mask) return compute_detector();
    else return detector_mask;
}

void vSTEM::set_detector_mask(py::array_t<int> array){
    py::buffer_info buf_info = array.request();
    if (buf_info.ndim != 1) throw std::runtime_error("Input should be a 1-D array");
    int* data_ptr = static_cast<int*>(buf_info.ptr);
    size_t size = buf_info.size;
    detector_mask = std::vector<int>(data_ptr, data_ptr + size);
    use_mask = true;
};

void vSTEM::set_offsets(std::vector<std::array<float, 2>> _offsets)
{
    auto_offset = false;
    offsets = _offsets;
}

void vSTEM::from_atomic(){
    for (int i = 0; i < nxy; i++)
    {
        vSTEM_image[i] = atomic_vSTEM_image[i].load();
    }
}

std::vector<std::tuple<uintmax_t, int, uint64_t, std::vector<uint64_t>, std::vector<int>, std::vector<int>, int>> vSTEM::find_checkpoints(int n_splits)
{
    if (camera != CAMERA::CHEETAH)
        throw std::runtime_error("vSTEM.find_checkpoints() is currently only supported for .tpx3 (CHEETAH) files.");

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
    return cam.find_line_checkpoints(n_splits);
}

std::vector<std::tuple<uintmax_t, int, std::vector<int>, std::vector<int>, int>> vSTEM::find_checkpoints_pixeltrig(int n_splits)
{
    if (camera != CAMERA::CHEETAH_PIXELTRIG)
        throw std::runtime_error("vSTEM.find_checkpoints_pixeltrig() is currently only supported for pixel-trigger (smart-scan) .tpx3 files.");

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
    return cam.find_trigger_checkpoints(n_splits);
}