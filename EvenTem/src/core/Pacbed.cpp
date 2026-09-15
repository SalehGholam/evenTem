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

#include "Pacbed.h"

void Pacbed::run(){
    py::gil_scoped_release release;
    reset();

    if (decluster)
    {
        if (camera != CAMERA::CHEETAH && camera != CAMERA::CHEETAH_PIXELTRIG)
            throw std::runtime_error("Pacbed decluster=True is currently only supported for .tpx3 (CHEETAH) files.");
        // electron_count_lut_file (a saved 2D map, see ClusterResolver.hpp) is an
        // alternative to tot_per_electron's single ToT/ratio formula -- loading it
        // here means electron_count_lut is populated before the tot_per_electron
        // check below, so a run relying solely on the map (tot_per_electron left at
        // its default 0.0) is not rejected.
        if (!electron_count_lut_file.empty())
            electron_count_lut = load_electron_count_lut_file(electron_count_lut_file);
        if (tot_per_electron <= 0.0 && electron_count_lut.empty())
            throw std::invalid_argument("Pacbed.tot_per_electron must be set (calibrated from your own cluster ToT-sum histogram), or electron_count_lut_file/electron_count_lut must be set, before enabling decluster=True.");
    }

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
        if (use_mask) cam.enable_Pacbed_mask(&scan_mask, &Pacbed_image);
        else cam.enable_Pacbed(&Pacbed_image);
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
            cam.enable_Pacbed_declustered(dtime,dspace,cluster_range,tot_per_electron,
                &Pacbed_image, n_threads,
                &clustersize_histogram,&energy_histogram,&clustersize_tot_histogram,
                electron_count_lut.empty() ? nullptr : &electron_count_lut,
                use_mask ? &scan_mask : nullptr);
        }
        else if (use_mask) cam.enable_Pacbed_mask(&scan_mask, &Pacbed_image);
        else cam.enable_Pacbed(&Pacbed_image);
        cam.run();
        process_data();
        cam.terminate();
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
        if (use_mask) cam.enable_Pacbed_mask(&scan_mask, &Pacbed_image);
        else cam.enable_Pacbed(&Pacbed_image);
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
            cam.enable_Pacbed_declustered(dtime,dspace,cluster_range,tot_per_electron,
                &Pacbed_image, n_threads,
                &clustersize_histogram,&energy_histogram,&clustersize_tot_histogram,
                electron_count_lut.empty() ? nullptr : &electron_count_lut,
                use_mask ? &scan_mask : nullptr);
        }
        else if (use_mask) cam.enable_Pacbed_mask(&scan_mask, &Pacbed_image);
        else cam.enable_Pacbed(&Pacbed_image);
        cam.run();
        process_data();
        cam.terminate();
        break;
    }
    case CAMERA::MERLIN:
    {
        if (bitdepth == 8)
        {
            if (n_cam == 512)
            {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else if (n_cam == 514)
            {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else if (n_cam == 256)
            {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else std::runtime_error("Invalid detector size for Merlin");
        }
        else if (bitdepth == 16)
        {
            if (n_cam == 512) 
            {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else if (n_cam == 514) 
            {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else if (n_cam == 256) 
            {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
        }
        else std::runtime_error("Invalid detector size for Merlin");
        break;
    }
    case CAMERA::NUMPY:
    {
        if (bitdepth == 8) 
        {
            if (n_cam == 64) {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else if (n_cam == 128) {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else if (n_cam == 192) {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else if (n_cam == 256) {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else if (n_cam == 512) {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else std::runtime_error("Invalid detector size for Numpy");
        }
        else if (bitdepth == 16)
        {
            if (n_cam == 64) {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else if (n_cam == 128) {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else if (n_cam == 256) {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else if (n_cam == 512) {
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
                cam.enable_Pacbed(&Pacbed_image);
                cam.run();
                process_data();
                cam.terminate();
            }
            else std::runtime_error("Invalid detector size for Numpy");
        }
        else std::runtime_error("Invalid bitdepth for Numpy");
        break;
    }
    }
    rc_quit = true;
}

void Pacbed::reset()
{
    rc_quit = false;
    fr_freq = 0;

    // Initializations
    nxy = nx * ny;
    id_image = 0;
    fr_total = nxy * rep;
    fr_count = 0;

    Pacbed_image.assign(n_cam*n_cam, 0);
    
    // Data Processing Progress
    *processor_line = 0;
    *preprocessor_line = 0;

}


void Pacbed::line_processor(
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

        // Mirrors Roi/vSTEM/Var/FourD's identical copy -- see Roi.cpp's line_processor
        // for why (raw C++ console output doesn't reliably reach Jupyter; this lets
        // Python poll `.progress` from another thread instead).
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

void Pacbed::set_scan_mask(py::array_t<int> array){
    py::buffer_info buf_info = array.request();
    if (buf_info.ndim != 1) throw std::runtime_error("Input should be a 1-D array");
    int* data_ptr = static_cast<int*>(buf_info.ptr);
    size_t size = buf_info.size;
    scan_mask = std::vector<int>(data_ptr, data_ptr + size);
    use_mask = true;
};
