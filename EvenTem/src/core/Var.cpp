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

#include "Var.h"

void Var::run(){
     py::gil_scoped_release release;
     reset();

     if (decluster)
     {
         if (camera != CAMERA::CHEETAH && camera != CAMERA::CHEETAH_PIXELTRIG)
             throw std::runtime_error("Var decluster=True is currently only supported for .tpx3 (CHEETAH) files.");
         // electron_count_lut_file (a saved 2D map, see ClusterResolver.hpp) is an
         // alternative to tot_per_electron's single ToT/ratio formula -- loading it
         // here means electron_count_lut is populated before the tot_per_electron
         // check below, so a run relying solely on the map (tot_per_electron left at
         // its default 0.0) is not rejected.
         if (!electron_count_lut_file.empty())
             electron_count_lut = load_electron_count_lut_file(electron_count_lut_file);
         if (tot_per_electron <= 0.0 && electron_count_lut.empty())
             throw std::invalid_argument("Var.tot_per_electron must be set (calibrated from your own cluster ToT-sum histogram), or electron_count_lut_file/electron_count_lut must be set, before enabling decluster=True.");
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
            cam.enable_var(&Var_data, offset, inner_radius, outer_radius);
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
                cam.enable_var_declustered(dtime,dspace,cluster_range,tot_per_electron,
                    &Var_data, offset, inner_radius, outer_radius, n_threads,
                    &clustersize_histogram,&energy_histogram,&clustersize_tot_histogram,
                    electron_count_lut.empty() ? nullptr : &electron_count_lut);
            }
            else cam.enable_var(&Var_data, offset, inner_radius, outer_radius);
            cam.run();
            process_data();
            cam.terminate();
            if (decluster)
            {
                // Same race as vSTEM's equivalent fix: line_processor() above copies
                // Var_data into Var_image incrementally, driven by raw-decode
                // progress, but declustering credits Var_data asynchronously on a
                // slower, lagging thread -- so Var_image ends up copied before any
                // cluster is actually credited. cam.terminate() guarantees
                // declustering has now finished, so redo the copy once from the
                // final repetition's slot (Var_data alternates by repetition, not
                // cumulative -- matches Var::line_processor's own id_image choice).
                int final_id_image = (rep - 1) % 2;
                for (int i = 0; i < nxy; ++i)
                    Var_image[i] = (float)Var_data[final_id_image][i];
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
            cam.enable_var(&Var_data, offset, inner_radius, outer_radius);
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
                cam.enable_var_declustered(dtime,dspace,cluster_range,tot_per_electron,
                    &Var_data, offset, inner_radius, outer_radius, n_threads,
                    &clustersize_histogram,&energy_histogram,&clustersize_tot_histogram,
                    electron_count_lut.empty() ? nullptr : &electron_count_lut);
            }
            else cam.enable_var(&Var_data, offset, inner_radius, outer_radius);
            cam.run();
            process_data();
            cam.terminate();
            if (decluster)
            {
                // Same staging-buffer race as the CHEETAH case above, same fix.
                int final_id_image = (rep - 1) % 2;
                for (int i = 0; i < nxy; ++i)
                    Var_image[i] = (float)Var_data[final_id_image][i];
            }
            break;
        }
     }

    rc_quit = true;
 }

void Var::set_offset(std::array<float, 2> _offset)
{
    auto_offset = false;
    offset = _offset;
}



void Var::reset()
{
    rc_quit = false;
    fr_freq = 0;

    std::fill(Var_image.begin(), Var_image.end(), 0);

    // Initializations
    nxy = nx * ny;
    id_image = 0;
    fr_total = nxy * rep;
    fr_count = 0;

    if (auto_offset) offset = {(float)n_cam/2, (float)n_cam/2};

    // Allocate memory for image arrays
    Var_image.assign(nxy, 0);
    for (int i=0; i<2; i++)
    {
        Var_data[i].assign(nxy, 0);
    }

    // Data Processing Progress
    *processor_line = 0;
    *preprocessor_line = 0;

}


void Var::line_processor(
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

        // Mirrors FourD::line_processor's identical copy -- see Roi.cpp's line_processor
        // for why (raw C++ console output doesn't reliably reach Jupyter; this lets
        // Python poll `.progress` from another thread instead).
        progress_percent = prog_mon->progress_percent;

        for (size_t i = 0; i < (size_t)nx; i++)
        {
            int idxx_p_i = idxx + i;
            if ((idxx_p_i >= 0) | (nx > 1))
            {
            Var_image[idxx_p_i] = (float)Var_data[id_image][idxx_p_i];
            }
        // end of line handler
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
}

