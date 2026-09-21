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

#include "Roi.h"

void Roi::run(){
     py::gil_scoped_release release;
     reset();

     if (decluster)
     {
         if (camera != CAMERA::CHEETAH && camera != CAMERA::CHEETAH_PIXELTRIG)
             throw std::runtime_error("Roi decluster=True is currently only supported for .tpx3 (CHEETAH) files.");
         // electron_count_lut_file (a saved 2D map, see ClusterResolver.hpp) is an
         // alternative to tot_per_electron's single ToT/ratio formula -- loading it
         // here means electron_count_lut is populated before the tot_per_electron
         // check below, so a run relying solely on the map (tot_per_electron left at
         // its default 0.0) is not rejected.
         if (!electron_count_lut_file.empty())
             electron_count_lut = load_electron_count_lut_file(electron_count_lut_file);
         if (tot_per_electron <= 0.0 && electron_count_lut.empty())
             throw std::invalid_argument("Roi.tot_per_electron must be set (calibrated from your own cluster ToT-sum histogram), or electron_count_lut_file/electron_count_lut must be set, before enabling decluster=True.");
     }
     if (tot_mode)
     {
         if (camera != CAMERA::CHEETAH && camera != CAMERA::CHEETAH_PIXELTRIG)
             throw std::runtime_error("Roi tot_mode=True is currently only supported for .tpx3 (CHEETAH) files.");
         if (decluster)
             throw std::runtime_error("Roi tot_mode and decluster cannot both be enabled -- tot_mode sums each raw hit's own ToT (no clustering); decluster resolves clusters into electron counts. Pick one.");
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
            if (b_ROI_4D)
            {
                if (roi_bitdepth == 8) cam.enable_roi_4D(Roi_4D_8,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                else if (roi_bitdepth == 16) cam.enable_roi_4D(Roi_4D_16,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                else if (roi_bitdepth == 32) cam.enable_roi_4D(Roi_4D_32,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
            }
            else if (use_mask)
            {
                cam.enable_roi_mask(&roi_mask,&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern);
            }
            else
            {
                cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
            }
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
            // Multi-process file-splitting (see Roi.h) -- defaults (0, 0, -1)
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
                std::vector<std::vector<int>> *_p_electron_lut = electron_count_lut.empty() ? nullptr : &electron_count_lut;
                if (roi_bitdepth == 8) cam.enable_roi_declustered(dtime,dspace,cluster_range,tot_per_electron,Roi_4D_8,b_ROI_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin,n_threads,&clustersize_histogram,&energy_histogram,&clustersize_tot_histogram,_p_electron_lut);
                else if (roi_bitdepth == 16) cam.enable_roi_declustered(dtime,dspace,cluster_range,tot_per_electron,Roi_4D_16,b_ROI_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin,n_threads,&clustersize_histogram,&energy_histogram,&clustersize_tot_histogram,_p_electron_lut);
                else if (roi_bitdepth == 32) cam.enable_roi_declustered(dtime,dspace,cluster_range,tot_per_electron,Roi_4D_32,b_ROI_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin,n_threads,&clustersize_histogram,&energy_histogram,&clustersize_tot_histogram,_p_electron_lut);
            }
            else if (b_ROI_4D)
            {
                if (roi_bitdepth == 8) cam.enable_roi_4D(Roi_4D_8,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                else if (roi_bitdepth == 16) cam.enable_roi_4D(Roi_4D_16,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                else if (roi_bitdepth == 32) cam.enable_roi_4D(Roi_4D_32,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                if (tot_mode) cam.b_tot = true;
            }
            else if (use_mask)
            {
                cam.enable_roi_mask(&roi_mask,&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern);
            }
            else
            {
                cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                if (tot_mode) cam.b_tot = true;
            }
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
                std::vector<std::vector<int>> *_p_electron_lut = electron_count_lut.empty() ? nullptr : &electron_count_lut;
                if (roi_bitdepth == 8) cam.enable_roi_declustered(dtime,dspace,cluster_range,tot_per_electron,Roi_4D_8,b_ROI_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin,n_threads,&clustersize_histogram,&energy_histogram,&clustersize_tot_histogram,_p_electron_lut);
                else if (roi_bitdepth == 16) cam.enable_roi_declustered(dtime,dspace,cluster_range,tot_per_electron,Roi_4D_16,b_ROI_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin,n_threads,&clustersize_histogram,&energy_histogram,&clustersize_tot_histogram,_p_electron_lut);
                else if (roi_bitdepth == 32) cam.enable_roi_declustered(dtime,dspace,cluster_range,tot_per_electron,Roi_4D_32,b_ROI_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin,n_threads,&clustersize_histogram,&energy_histogram,&clustersize_tot_histogram,_p_electron_lut);
            }
            else if (b_ROI_4D)
            {
                if (roi_bitdepth == 8) cam.enable_roi_4D(Roi_4D_8,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                else if (roi_bitdepth == 16) cam.enable_roi_4D(Roi_4D_16,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                else if (roi_bitdepth == 32) cam.enable_roi_4D(Roi_4D_32,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                if (tot_mode) cam.b_tot = true;
            }
            else if (use_mask)
            {
                cam.enable_roi_mask(&roi_mask,&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern);
            }
            else
            {
                cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                if (tot_mode) cam.b_tot = true;
            }
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
            if (b_ROI_4D)
            {
                if (roi_bitdepth == 8) cam.enable_roi_4D(Roi_4D_8,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                else if (roi_bitdepth == 16) cam.enable_roi_4D(Roi_4D_16,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                else if (roi_bitdepth == 32) cam.enable_roi_4D(Roi_4D_32,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
            }
            else if (use_mask)
            {
                cam.enable_roi_mask(&roi_mask,&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern);
            }
            else
            {
                cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
            }
            cam.run();
            process_data();
            cam.terminate();
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data();
                    cam.terminate();
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data();
                    cam.terminate();
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data();
                    cam.terminate();
                }
                else std::runtime_error("Invalid detector size for Merlin");
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data();
                    cam.terminate();
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data();
                    cam.terminate();
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data();
                    cam.terminate();
                }
                else std::runtime_error("Invalid detector size for Merlin");
            }
            else std::runtime_error("Invalid bitdepth for Merlin");
            break;
        }
        case CAMERA::NUMPY:
        {
            if (bitdepth == 8) 
            {
                if (n_cam == 512)
                {
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data();
                    cam.terminate();
                }
                else if (n_cam == 256)
                {
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data();
                    cam.terminate();
                }
                else if (n_cam == 128)
                {
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data();
                    cam.terminate();
                }
                else if (n_cam == 192)
                {
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data();
                    cam.terminate();
                }
                else if (n_cam == 64)
                {
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data();
                    cam.terminate();
                }
                else std::runtime_error("Invalid detector size for Numpy");
            }
            else if (bitdepth == 16) 
            {
                if (n_cam == 512)
                {
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data();
                    cam.terminate();
                }
                else if (n_cam == 256)
                {
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data();
                    cam.terminate();
                }
                else if (n_cam == 128)
                {
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
                    cam.run();
                    process_data(); 
                    cam.terminate();
                }
                else if (n_cam == 64)
                {
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
                    if (b_ROI_4D)
                    {
                        // cam.enable_roi_4D(&Roi_4D,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right,det_bin);
                    }
                    else
                    {
                        cam.enable_roi(&Roi_scan_image_stack,&Roi_diffraction_pattern_stack,&Roi_scan_image,&Roi_diffraction_pattern, lower_left, upper_right);
                    }
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


void Roi::set_roi(int x , int y, int width, int height)
{
    lower_left[0] = x;
    lower_left[1] = ny - (y+height); // y is from top to bottom, but the image is from bottom to top, so we need to flip
    upper_right[0] = x + width;
    upper_right[1] = ny-y;

    if (x < 0 || y < 0)
    {
        throw std::invalid_argument("ROI x & y parameters must be positive");
    }
    if (width <= 0 && height <= 0)
    {
        throw std::invalid_argument("ROI width or height parameters must be > 0");
    }
    if (x + width > nx || y + height > ny)
    {
        throw std::invalid_argument("ROI must be within the image dimensions");
    }

    L_0 = upper_right[0] - lower_left[0];
    L_1 = upper_right[1] - lower_left[1];

    finish_line = (y+height+1) + ny*(rep-1);
}

std::array<int,4> Roi::get_roi()
{
    std::array<int,4> roi = {lower_left[0], lower_left[1], upper_right[0], upper_right[1]};
    return roi;
}

void Roi::set_roi_mask(std::vector<py::array_t<int>> arrays){
    for (auto& array : arrays) {
        py::buffer_info buf_info = array.request();
        if (buf_info.ndim != 1) throw std::runtime_error("Input should be a 1-D array");
        int* data_ptr = static_cast<int*>(buf_info.ptr);
        size_t size = buf_info.size;
        roi_mask.push_back(std::vector<int>(data_ptr, data_ptr + size));
    }
    roi_mask.push_back(std::vector<int>(nx*ny,0));
    use_mask = true;
    L_1 = ny;
    L_0 = nx;
    finish_line = ny*rep;
};

void Roi::set_bitdepth(int _bitdepth)
{
    if (_bitdepth == 8) roi_bitdepth = 8;
    else if (_bitdepth == 16) roi_bitdepth = 16;
    else if (_bitdepth == 32) roi_bitdepth = 32;
    else
    {
        throw std::invalid_argument("Invalid bitdepth, only 8 , 16 or 32 are supported");
    }
}

void Roi::reset()
{
    rc_quit = false;
    fr_freq = 0;

    // Initializations
    nxy = nx * ny;
    id_image = 0;
    // Multi-process file-splitting: fr_total/fr_count_seed must reflect THIS
    // worker's own absolute slice boundary (stop_at_line/line_number_offset),
    // exactly mirroring vSTEM::reset()'s identical fix. Without this, a
    // worker whose slice ends before finish_line (the ROI's own row-
    // restricted early exit -- see set_roi()) never satisfies either
    // termination check once the reader stops supplying new lines, and
    // process_data()'s "while (*processor_line != -1)" loop spins forever.
    // Both default to the old whole-file values (nxy*rep, 0) when
    // stop_at_line/line_number_offset are unset.
    fr_total = (stop_at_line >= 0) ? (int)((size_t)nx * stop_at_line) : nxy * rep;
    fr_count_seed = (size_t)nx * line_number_offset;
    fr_count = 0;

    Roi_diffraction_pattern.assign(n_cam*n_cam, 0);
    Roi_diffraction_pattern_stack.assign(rep+1, std::vector<uint64_t>(n_cam*n_cam, 0));
    Roi_scan_image.assign(L_1*L_0,0);
    Roi_scan_image_stack.assign(rep+1,std::vector<uint64_t>(L_1*L_0,0));

    if (b_ROI_4D)
    {
        if (roi_bitdepth == 8 ) Roi_4D_8->init(L_1,L_0,n_cam,det_bin);
        else if (roi_bitdepth == 16)  Roi_4D_16->init(L_1,L_0,n_cam,det_bin);
        else if (roi_bitdepth == 32)  Roi_4D_32->init(L_1,L_0,n_cam,det_bin);
    }

    // Data Processing Progress
    *processor_line = 0;
    *preprocessor_line = 0;

}


void Roi::line_processor(
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

        // Mirrors FourD::line_processor's identical copy -- exposed as the
        // read-only `.progress` (0-100) property. Raw stdout/stderr writes from a
        // C extension (like ProgressMonitor's own console bar) don't reliably
        // reach a Jupyter cell's output; this lets Python poll actual progress
        // from another thread instead (run() releases the GIL) and render it
        // with a real notebook-native progress bar (tqdm, ipywidgets, ...).
        progress_percent = prog_mon->progress_percent;

        // end of line handler
        int update_line = idxx / nx;
        if ((prog_mon->report_set) && (update_line)>0)
        {
            fr_freq = prog_mon->fr_freq;
            prog_mon->reset_flags();
        }

        if (*processor_line > finish_line){
            rc_quit = true;
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
}

std::vector<std::tuple<uintmax_t, int, uint64_t, std::vector<uint64_t>, std::vector<int>, std::vector<int>, int>> Roi::find_checkpoints(int n_splits, bool allow_sidecar)
{
    if (camera != CAMERA::CHEETAH)
        throw std::runtime_error("Roi.find_checkpoints() is currently only supported for .tpx3 (CHEETAH) files.");

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
    return cam.find_line_checkpoints(n_splits, allow_sidecar);
}

