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

#ifndef TIMEPIX_H
#define TIMEPIX_H

#ifdef __GNUC__
#define PACK(__Declaration__) __Declaration__ __attribute__((__packed__))
#endif

#ifdef _MSC_VER
#define PACK(__Declaration__) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
#endif

#define _USE_MATH_DEFINES
#include <cmath>
#include <atomic>
#include <vector>
#include <array>
#include <thread>
#include <chrono>
#include <functional>
#include <algorithm>
#include <cstdlib>

#include <fstream>
#include "SocketConnector.h"
#include "FileConnector.h"
#include "FileConnectorMmap.h"
#include "Declusterer.hpp"
#include "dtype_Electron.hpp"
#include "Logger.hpp"
#include "Roi4D.hpp"
#include "AtomicWrapper.hpp"

#define mmap_buffer_size 4194304 

template <typename event, int buffer_size, int n_buffer>
class TIMEPIX
{
protected:

    #ifdef GPRI_OPTION_ENABLED
        enum class FunctionType 
        {
            vstem,
            multi_vstem,
            mask_vstem,
            com,
            GPRI,
            count_chunked_8,
            count_chunked_16,
            count_chunked_32,
            pacbed,
            pacbed_mask,
            var,
            roi,
            roi_mask,
            roi_4D,
            roi_ToT,
            write_electron,
            write_declusterer_buffer,
            information,
            atomic_vstem,
            tcBF
        };
    #else
        enum class FunctionType
        {
            vstem,
            multi_vstem,
            mask_vstem,
            com,
            count_chunked_8,
            count_chunked_16,
            count_chunked_32,
            pacbed,
            pacbed_mask,
            var,
            roi,
            roi_mask,
            roi_4D,
            roi_ToT,
            write_electron,
            write_declusterer_buffer,
            information,
            atomic_vstem,
            tcBF
        };
    #endif


    // ----------------------------------------------------------------------------------------------- 
    // process methods
    // -----------------------------------------------------------------------------------------------

    inline void vstem(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        int _d2 = (_kx - x_offset)*(_kx - x_offset) + (_ky - y_offset)*(_ky - y_offset);
        if (_d2 >= in_radius_sqr && _d2 <= out_radius_sqr)  // inclusive inner radius, matching multi_vstem/AnnularDetector::compute_detector -- see CPP_EVENTEM_BUGS.md #6
        {   
            (*p_stem_data)[_id_image][_probe_position]++;
        }
    };

    inline void atomic_vstem(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        (*p_atomic_stem_data)[_probe_position]++;
        atomic_counter++;
    };

    inline void multi_vstem(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        for (int i = 0; i < n_detectors; i++)
        {
            int _d2 = (_kx - offsets[i][0])*(_kx - offsets[i][0]) + (_ky - offsets[i][1])*(_ky - offsets[i][1]);
            if (_d2 >= radia_sqr[i][0] && _d2 <= radia_sqr[i][1])
            {
                (*p_stem_data)[_id_image][_probe_position]++;
            }
        }
    };

    inline void mask_vstem(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
            (*p_stem_data)[_id_image][_probe_position] += detector_mask[_ky*n_cam+_kx];
    };

    inline void com(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        (*p_dose_data)[_id_image][_probe_position]++;
        (*p_sumy_data)[_id_image][_probe_position] += _ky;
        (*p_sumx_data)[_id_image][_probe_position] += _kx;
    };

    inline void com_masked(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        (*p_dose_data)[_id_image][_probe_position] += com_mask[_ky*n_cam+_kx];
        (*p_sumy_data)[_id_image][_probe_position] += _ky*com_mask[_ky*n_cam+_kx];
        (*p_sumx_data)[_id_image][_probe_position] += _kx*com_mask[_ky*n_cam+_kx];
    };

    inline void tcBF(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        int m = (*p_tcbf_detector_mask)[_ky*n_cam+_kx];
        if (m != -1)
        {
            (*p_tcbf_BF_image)[_probe_position]++;
            (*p_tcBF_stack)[m][_probe_position]++;
        }
    };

    #ifdef GPRI_OPTION_ENABLED
    inline void GPRI(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
    };
    #endif


    // FIXED (see CPP_EVENTEM_BUGS.md #5): voxel_offset below now indexes ky*n_cam+kx,
    // matching Pacbed/Roi. Breaking change for anyone relying on the old kx*n_cam+ky
    // orientation of a FourD cube exported before this fix.

    inline void count_chunked_8(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        uint64_t _x_pp = _probe_position%nx;
        uint64_t _y_pp = _probe_position/nx;

        uint64_t _bin_probe_position = (_y_pp/fourD_scan_bin)*nx_scan_bin + _x_pp/fourD_scan_bin;

        (*p_counts_data)[_bin_probe_position]++;
        uint64_t chunk_span = (uint64_t)chunksize_scan_bin*nx_scan_bin;
        uint64_t voxel_offset = (_bin_probe_position%chunk_span)*diff_pattern_size + (_ky/fourD_det_bin*n_cam/fourD_det_bin+_kx/fourD_det_bin);  // ky*n_cam+kx, matching Pacbed/Roi -- see CPP_EVENTEM_BUGS.md #5 (was transposed: kx*n_cam+ky)
        if (p_full_chunk_data_8)
        {
            int chunk_id = (int)(_bin_probe_position/chunk_span);
            std::lock_guard<std::mutex> lock((*p_full_chunk_mtx)[chunk_id]);
            (*p_full_chunk_data_8)[chunk_id][voxel_offset]++;
            return;
        }
        int _id_chunk = (int)(_bin_probe_position/chunk_span)%2;
        std::lock_guard<std::mutex> lock(mtx[_id_chunk]);
        (*p_fourDchunk_data_8)[_id_chunk][voxel_offset]++;
    };

    inline void count_chunked_16(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        uint64_t _x_pp = _probe_position%nx;
        uint64_t _y_pp = _probe_position/nx;

        uint64_t _bin_probe_position = (_y_pp/fourD_scan_bin)*nx_scan_bin + _x_pp/fourD_scan_bin;

        (*p_counts_data)[_bin_probe_position]++;
        uint64_t chunk_span = (uint64_t)chunksize_scan_bin*nx_scan_bin;
        uint64_t voxel_offset = (_bin_probe_position%chunk_span)*diff_pattern_size + (_ky/fourD_det_bin*n_cam/fourD_det_bin+_kx/fourD_det_bin);  // ky*n_cam+kx, matching Pacbed/Roi -- see CPP_EVENTEM_BUGS.md #5 (was transposed: kx*n_cam+ky)
        if (p_full_chunk_data_16)
        {
            int chunk_id = (int)(_bin_probe_position/chunk_span);
            std::lock_guard<std::mutex> lock((*p_full_chunk_mtx)[chunk_id]);
            (*p_full_chunk_data_16)[chunk_id][voxel_offset]++;
            return;
        }
        int _id_chunk = (int)(_bin_probe_position/chunk_span)%2;
        std::lock_guard<std::mutex> lock(mtx[_id_chunk]);
        (*p_fourDchunk_data_16)[_id_chunk][voxel_offset]++;
    };

    inline void count_chunked_32(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        uint64_t _x_pp = _probe_position%nx;
        uint64_t _y_pp = _probe_position/nx;

        uint64_t _bin_probe_position = (_y_pp/fourD_scan_bin)*nx_scan_bin + _x_pp/fourD_scan_bin;

        (*p_counts_data)[_bin_probe_position]++;
        uint64_t chunk_span = (uint64_t)chunksize_scan_bin*nx_scan_bin;
        uint64_t voxel_offset = (_bin_probe_position%chunk_span)*diff_pattern_size + (_ky/fourD_det_bin*n_cam/fourD_det_bin+_kx/fourD_det_bin);  // ky*n_cam+kx, matching Pacbed/Roi -- see CPP_EVENTEM_BUGS.md #5 (was transposed: kx*n_cam+ky)
        if (p_full_chunk_data_32)
        {
            int chunk_id = (int)(_bin_probe_position/chunk_span);
            std::lock_guard<std::mutex> lock((*p_full_chunk_mtx)[chunk_id]);
            (*p_full_chunk_data_32)[chunk_id][voxel_offset]++;
            return;
        }
        int _id_chunk = (int)(_bin_probe_position/chunk_span)%2;
        std::lock_guard<std::mutex> lock(mtx[_id_chunk]);
        (*p_fourDchunk_data_32)[_id_chunk][voxel_offset]++;
    };

    inline void pacbed(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        (*p_pacbed_data)[_ky*n_cam+_kx]++;
    };

    // Masked-scan-pattern PACBED: same accumulation as pacbed() above, but only for
    // scan positions where pacbed_scan_mask is nonzero -- e.g. accumulate the
    // diffraction average only over a particle (or only over vacuum), rather than the
    // whole scan. pacbed_scan_mask is flat, nx*ny, row-major (index = ry*nx+rx, no
    // y-flip) -- same convention _probe_position already arrives in here (see
    // mask_vstem's identical, unflipped indexing above).
    inline void pacbed_mask(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        if (pacbed_scan_mask[_probe_position])
            (*p_pacbed_data)[_ky*n_cam+_kx]++;
    };

    inline void var(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        int _d2 = (_kx - offset[0])*(_kx - offset[0]) + (_ky - offset[1])*(_ky - offset[1]);
        if (_d2 > inner_radius_sqr_var && _d2 <= outer_radius_sqr_var)
        {
            (*p_var_data)[_id_image][_probe_position] += (_kx-offset[0])*(_kx-offset[0])+(_ky-offset[1])*(_ky-offset[1]);
        }
    };

    inline void roi(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        int _x = _probe_position%nx;
        int _y = ny - floor(_probe_position/nx);  // flip by ny, matching Roi::set_roi() -- see CPP_EVENTEM_BUGS.md #3

        if (_x >= lower_left[0] && _x < upper_right[0] && _y > lower_left[1] && _y <= upper_right[1])
        {
            (*p_roi_diffraction_pattern_stack)[_id_image][_ky*n_cam+_kx]++;
            (*p_roi_scan_image_stack)[_id_image][(L_1 - (_y-lower_left[1])) * L_0 + (_x-lower_left[0])]++;
            (*p_roi_diffraction_pattern)[_ky*n_cam+_kx]++;
            (*p_roi_scan_image)[(L_1 - (_y-lower_left[1])) * L_0 + (_x-lower_left[0])]++;
        } 
    };

    // ToT-sum mode (Roi.tot_mode=True, see Roi::run()): every raw hit contributes
    // its own ToT to scan image + diffraction pattern (and, in roi_4D_ToT below,
    // the 4D cube) instead of +1 -- "summed deposited charge per probe" rather
    // than "hit count per probe". Fixed to also sum ToT into the scan image below
    // (previously incremented by +1 there while summing ToT into the diffraction
    // pattern -- an inconsistency that never mattered because this function was
    // dead code before tot_mode existed: b_tot=true and functionType=roi never
    // co-occurred under any previously-shipped feature).
    inline void roi_ToT(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        int _x = _probe_position%nx;
        int _y = ny - floor(_probe_position/nx);  // flip by ny, matching Roi::set_roi() -- see CPP_EVENTEM_BUGS.md #3

        if (_x >= lower_left[0] && _x < upper_right[0] && _y > lower_left[1] && _y <= upper_right[1])
        {
            (*p_roi_diffraction_pattern_stack)[_id_image][_ky*n_cam+_kx] += this->tot;
            (*p_roi_scan_image_stack)[_id_image][(L_1 - (_y-lower_left[1])) * L_0 + (_x-lower_left[0])] += this->tot;
            (*p_roi_diffraction_pattern)[_ky*n_cam+_kx] += this->tot;
            (*p_roi_scan_image)[(L_1 - (_y-lower_left[1])) * L_0 + (_x-lower_left[0])] += this->tot;
        }
    };

    inline void roi_mask(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        if (mask_roi[_id_image][_probe_position] == 1)
        {
            (*p_roi_diffraction_pattern_stack)[_id_image][_ky*n_cam+_kx]++;
            (*p_roi_scan_image_stack)[_id_image][_probe_position]++;
            (*p_roi_diffraction_pattern)[_ky*n_cam+_kx] += 1;
            (*p_roi_scan_image)[_probe_position]++;
        } 
    };
  
    inline void roi_4D(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        int _x = _probe_position%nx;
        int _y = ny - floor(_probe_position/nx);  // flip by ny, matching Roi::set_roi() -- see CPP_EVENTEM_BUGS.md #3

        if (_x >= lower_left[0] && _x < upper_right[0] && _y > lower_left[1] && _y <= upper_right[1] )
        {
            (*p_roi_diffraction_pattern)[_ky*n_cam+_kx]++;
            (*p_roi_scan_image)[(L_1 - (_y-lower_left[1])) * L_0 + (_x-lower_left[0])]++;
            p_roi_4D->increment(L_1 - (_y-lower_left[1]), _x-lower_left[0], _ky/det_bin, _kx/det_bin);
        }
    };

    // ToT-sum counterpart to roi_4D above (see Roi.tot_mode / roi_ToT's comment) --
    // credits this->tot instead of +1 to the diffraction pattern, scan image, AND
    // the 4D cube (via increment_by).
    inline void roi_4D_ToT(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        int _x = _probe_position%nx;
        int _y = ny - floor(_probe_position/nx);  // flip by ny, matching Roi::set_roi() -- see CPP_EVENTEM_BUGS.md #3

        if (_x >= lower_left[0] && _x < upper_right[0] && _y > lower_left[1] && _y <= upper_right[1] )
        {
            (*p_roi_diffraction_pattern)[_ky*n_cam+_kx] += this->tot;
            (*p_roi_scan_image)[(L_1 - (_y-lower_left[1])) * L_0 + (_x-lower_left[0])] += this->tot;
            p_roi_4D->increment_by(L_1 - (_y-lower_left[1]), _x-lower_left[0], _ky/det_bin, _kx/det_bin, (uint64_t)this->tot);
        }
    };

    inline void write_electron(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        // FIX ELECTRON NOT LOCAL VAR NOW
        electron.kx = _kx/det_bin_electron;
        electron.ky = _ky/det_bin_electron;
        electron.rx = (_probe_position%nx)/scan_bin_electron;
        electron.ry = (_probe_position/nx)/scan_bin_electron;
        electron.id_image = _id_image;

        if (electron.rx < (x_crop/scan_bin_electron) && electron.ry < (y_crop/scan_bin_electron))
        {
            (*p_file).write((const char *)&electron, sizeof(electron));
        }
    };

    inline void write_declusterer_buffer(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image, uint64_t _toa, uint16_t _tot)
    {
        uint16_t _rx = _probe_position%nx;
        uint16_t _ry = _probe_position/nx;
        declusterer.buffer[declusterer.buffer_id_filling]->push_back({_kx, _ky, _rx, _ry, _id_image,_toa,_tot});
    };


    inline void information(uint64_t _probe_position, uint16_t _kx, uint16_t _ky, uint16_t _id_image)
    {
        (*p_information_image)[_probe_position] += -log2((*p_probability_distribution)[_ky*n_cam+_kx]);
        (*p_count_image)[_probe_position]++;
    };

protected:
    
    FileConnector file;
    FileConnectorMmap mmap;
    std::thread read_thread;
    std::thread proc_thread;
    int n_buf = n_buffer;
    int n_buffer_filled=0;
    int n_buffer_processed=0;
    uint16_t id_image = 0 ;
    uint64_t current_line = 0;
    uint64_t probe_position = 0;
    uint64_t probe_position_total = 0;
    uint16_t kx;
    uint16_t ky;
    uint64_t n_events_processed = 0;

    inline void read_file()
    {
        int buffer_id;
        size_t size = sizeof(buffer[0]);
        while ((!this->repetitions_reached) && (*p_processor_line!=-1))
        {
            if (n_buffer_filled < (n_buffer + n_buffer_processed))
            {
                buffer_id = n_buffer_filled % n_buffer;
                file.read_data((char *)&(buffer[buffer_id]), size);
                ++n_buffer_filled;
            }
            else
            {
                read_wait++;
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
        file.close_file();
    };

    inline void read_mmap()
    {
        int buffer_id;
        size_t map_size = mmap_buffer_size * sizeof(event) ;
        while ((!this->repetitions_reached) && (*p_processor_line!=-1))
        {
            if (n_buffer_filled < (n_buffer + n_buffer_processed))
            {
                buffer_id = n_buffer_filled % n_buffer;
                mmap.read_data(mmap_buffer[buffer_id], map_size);
                ++n_buffer_filled;
            }
            else
            {
                read_wait++;
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
        file.close_file();
    };

    inline void read_socket()
    {
        int buffer_id;
        while (*p_processor_line != -1)
        {
            if (n_buffer_filled < (n_buffer + n_buffer_processed))
            {
                buffer_id = n_buffer_filled % n_buffer;
                socket.read_data((char *)&(buffer[buffer_id]), sizeof(buffer[buffer_id]));
                ++n_buffer_filled;
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
       
    };

    void reset()
    {
        n_buffer_filled = 0;
        n_buffer_processed = 0;
        current_line = 0;
    };

public:
//-------------------------------------------------------------------------------------------------
    void enable_multi_vSTEM(std::vector<std::array<float, 2>> *_p_radia_sqr,std::vector<std::array<float, 2>> *_p_offsets,std::vector<std::vector<size_t>> *_p_stem_data)
    {
        p_stem_data = _p_stem_data;
        radia_sqr = (*_p_radia_sqr);
        offsets = *_p_offsets;
        n_detectors = radia_sqr.size();
        process.push_back(std::bind(&TIMEPIX::multi_vstem, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::multi_vstem;
        ++n_proc;
    }

    void enable_vSTEM(std::array<float, 2> *_p_radius_sqr,std::array<float, 2> *_p_offset,std::vector<std::vector<size_t>> *_p_stem_data)
    {
        p_stem_data = _p_stem_data;
        in_radius_sqr = (int)(*_p_radius_sqr)[0];
        out_radius_sqr = (int)(*_p_radius_sqr)[1];
        x_offset = (int)(*_p_offset)[0];
        y_offset = (int)(*_p_offset)[1];
        process.push_back(std::bind(&TIMEPIX::vstem, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::vstem;
        ++n_proc;
    }

    void enable_atomic_vSTEM(std::array<float, 2> *_p_radius_sqr,std::array<float, 2> *_p_offset,std::vector<atomwrapper<int>> *_p_atomic_stem_data)
    {
        p_atomic_stem_data = _p_atomic_stem_data;
        in_radius_sqr = (int)(*_p_radius_sqr)[0];
        out_radius_sqr = (int)(*_p_radius_sqr)[1];
        x_offset = (int)(*_p_offset)[0];
        y_offset = (int)(*_p_offset)[1];
        functionType = FunctionType::atomic_vstem;
        process.push_back(std::bind(&TIMEPIX::atomic_vstem, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        ++n_proc;
    }

    void enable_mask_vSTEM(std::vector<int> *_p_detector_mask,std::vector<std::vector<size_t>> *_p_stem_data)
    {
        p_stem_data = _p_stem_data;
        detector_mask = *_p_detector_mask;
        process.push_back(std::bind(&TIMEPIX::mask_vstem, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::mask_vstem;
        ++n_proc;
    }

    void enable_Ricom(std::vector<std::vector<size_t>> (*_p_dose_data),std::vector<std::vector<size_t>> (*_p_sumx_data),std::vector<std::vector<size_t>> (*_p_sumy_data))
    {
        p_dose_data = _p_dose_data;
        p_sumx_data = _p_sumx_data;
        p_sumy_data = _p_sumy_data;

        process.push_back(std::bind(&TIMEPIX::com, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::com;
        ++n_proc; 
    }

    void enable_Ricom_masked(std::vector<int> *_p_com_mask,std::vector<std::vector<size_t>> (*_p_dose_data),std::vector<std::vector<size_t>> (*_p_sumx_data),std::vector<std::vector<size_t>> (*_p_sumy_data))
    {
        p_dose_data = _p_dose_data;
        p_sumx_data = _p_sumx_data;
        p_sumy_data = _p_sumy_data;

        com_mask = *_p_com_mask;
        process.push_back(std::bind(&TIMEPIX::com_masked, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::com;
        ++n_proc; 
    }


    void enable_FourD(std::vector<uint64_t> *_p_counts_data, std::vector<uint8_t> (*_p_fourD_data)[2], size_t det_bin, size_t scan_bin, size_t _chunksize, std::mutex* _mtx)
    {
        p_counts_data = _p_counts_data;
        p_fourDchunk_data_8 = _p_fourD_data;
        process.push_back(std::bind(&TIMEPIX::count_chunked_8, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::count_chunked_8;
        is_fourD_chunked_type = true;
        ++n_proc;

        fourD_det_bin = det_bin;
        fourD_scan_bin = scan_bin;
        chunksize_scan_bin = _chunksize/scan_bin;
        n_cam_det_bin = n_cam/det_bin;
        nx_scan_bin = nx/scan_bin;
        diff_pattern_size = n_cam/fourD_det_bin*n_cam/fourD_det_bin;
        this->mtx = _mtx;
    }
    void enable_FourD(std::vector<uint64_t> *_p_counts_data, std::vector<uint16_t> (*_p_fourD_data)[2], size_t det_bin, size_t scan_bin, size_t _chunksize, std::mutex* _mtx)
    {
        p_counts_data = _p_counts_data;
        p_fourDchunk_data_16 = _p_fourD_data;
        process.push_back(std::bind(&TIMEPIX::count_chunked_16, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::count_chunked_16;
        is_fourD_chunked_type = true;
        ++n_proc;

        fourD_det_bin = det_bin;
        fourD_scan_bin = scan_bin;
        chunksize_scan_bin = _chunksize/scan_bin;
        n_cam_det_bin = n_cam/det_bin;
        nx_scan_bin = nx/scan_bin;
        diff_pattern_size = n_cam/fourD_det_bin*n_cam/fourD_det_bin;
        this->mtx = _mtx;
    }
    void enable_FourD(std::vector<uint64_t> *_p_counts_data, std::vector<uint32_t> (*_p_fourD_data)[2], size_t det_bin, size_t scan_bin, size_t _chunksize, std::mutex* _mtx)
    {
        p_counts_data = _p_counts_data;
        p_fourDchunk_data_32 = _p_fourD_data;
        process.push_back(std::bind(&TIMEPIX::count_chunked_32, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::count_chunked_32;
        is_fourD_chunked_type = true;
        ++n_proc;

        fourD_det_bin = det_bin;
        fourD_scan_bin = scan_bin;
        chunksize_scan_bin = _chunksize/scan_bin;
        n_cam_det_bin = n_cam/det_bin;
        nx_scan_bin = nx/scan_bin;
        diff_pattern_size = n_cam/fourD_det_bin*n_cam/fourD_det_bin;
        this->mtx = _mtx;
    }

    // Companion to enable_FourD() above, opt-in only for pixel-triggered (smart-
    // scan / custom pattern-file) acquisitions: FourD's on-disk chunk buffer
    // (chunk_data, set by enable_FourD above) is a 2-slot round-robin buffer,
    // normally kept in sync via line_processor()'s raw-decode-progress trigger --
    // which assumes trigger order tracks real scan row 1:1. True for a full
    // raster scan (CHEETAH), but not for a sparse custom pattern: measured on a
    // real 40856-trigger/262144-position smart-scan file, the real target row
    // drifted +428 rows ahead of the trigger-count-based proxy by the end of the
    // file, and (separately) an individual chip lagging the aggregate progress
    // could see its own events target an already-recycled slot, silently
    // dropping their 4D voxel credit -- the raw-path counterpart to the
    // declustered-FourD residual documented in IMPROVEMENTS.md, fixed the same
    // way: _p_full_chunk_data gives every chunk-group its own permanent slot for
    // the whole run (see FourD::allocate_full_chunk_buffer/flush_full_chunk_buffer),
    // so count_chunked_8/16/32 below write into it directly by absolute chunk id
    // instead of chunk_data's mod-2 slot, and advance_fourD_pixeltrig_chunk
    // (Cheetah_pixeltrig.hpp) never needs to reject/drop an event at all. No-op
    // unless called -- CHEETAH's raster FourD path is unaffected since it never
    // calls this.
    void enable_FourD_pixeltrig_flush(std::vector<std::vector<uint8_t>> *_p_full_chunk_data, std::vector<std::mutex> *_p_full_chunk_mtx)
    { p_full_chunk_data_8 = _p_full_chunk_data; p_full_chunk_mtx = _p_full_chunk_mtx; }

    void enable_FourD_pixeltrig_flush(std::vector<std::vector<uint16_t>> *_p_full_chunk_data, std::vector<std::mutex> *_p_full_chunk_mtx)
    { p_full_chunk_data_16 = _p_full_chunk_data; p_full_chunk_mtx = _p_full_chunk_mtx; }

    void enable_FourD_pixeltrig_flush(std::vector<std::vector<uint32_t>> *_p_full_chunk_data, std::vector<std::mutex> *_p_full_chunk_mtx)
    { p_full_chunk_data_32 = _p_full_chunk_data; p_full_chunk_mtx = _p_full_chunk_mtx; }

    // Charge-weighted declustered 4D conversion: raw hits are first grouped into
    // physical-electron clusters by the same Declusterer used everywhere else in
    // this project, then each cluster is resolved to N>=0 electrons from its total
    // deposited charge (see ClusterResolver.hpp) and credited -- N, not +1 -- at
    // its ToT-weighted centroid, into the same chunk_data buffer/mutex pair the
    // raw path uses. Requires scan_bin=1 (validated by the caller, FourD::run())
    // -- combining scan binning with the cluster-driven flush-timing scheme below
    // adds a dimension of complexity not needed for what was asked; can be
    // extended later if required.
    //
    // Flush timing is the one genuinely different thing here versus every other
    // declustered mode (Roi/vSTEM/Var): those write directly into their final
    // output arrays with no intermediate buffer, so timing doesn't matter. FourD's
    // chunk_data buffer is recycled (cleared and reused for a later chunk-group)
    // every `chunksize` lines -- if flushing stayed tied to raw decode progress
    // while this callback credits clusters asynchronously and later (declustering
    // measured to lag decode by roughly 8-10x elsewhere in this project), a late
    // cluster could target a buffer slot already flushed and recycled for a
    // different chunk-group -- silent data loss or a wrong-chunk write, not just
    // stale zeros. So this callback tracks its own progress (fourD_declustered_
    // chunk_id) from each cluster's own scan line -- advanced on EVERY cluster,
    // even ones that resolve to 0 electrons, since a chunk-group containing only
    // noise would otherwise never trigger a flush -- and calls the flush callback
    // itself when crossing a chunksize boundary, instead of relying on
    // line_processor()'s raw-progress trigger (which FourD::line_processor() gates
    // off entirely when decluster=True, see FourD.cpp).
    void enable_FourD_declustered(uint64_t _dtime, uint16_t _dspace, int _cluster_range, double _tot_per_electron,
    std::vector<uint64_t> *_p_counts_data,
    std::vector<std::vector<uint32_t>> *_p_declustered_chunk_data, std::vector<std::mutex> *_p_declustered_chunk_mtx,
    size_t det_bin, size_t _chunksize, int _n_threads,
    std::vector<int> *_p_clustersize_histogram, std::vector<int> *_p_energy_histogram,
    std::vector<std::vector<int>> *_p_joint_histogram,
    const std::vector<std::vector<int>> *_p_electron_lut = nullptr)
    {
        p_counts_data = _p_counts_data;
        fourD_det_bin = det_bin;
        fourD_scan_bin = 1;
        chunksize_scan_bin = (int)_chunksize;
        n_cam_det_bin = n_cam/det_bin;
        nx_scan_bin = (int)nx;
        diff_pattern_size = n_cam/fourD_det_bin*n_cam/fourD_det_bin;

        declusterer.init(_dtime, _dspace, _cluster_range, (int)nx, (int)ny, /*scan_bin*/1, /*det_bin*/1,
            roi_declustered_dummy_file, _n_threads, _p_clustersize_histogram, _p_energy_histogram);
        declusterer.set_disable_file_write(true);
        declusterer.set_joint_histogram(_p_joint_histogram);

        declusterer.set_cluster_callback([this, _tot_per_electron, _p_electron_lut, _p_declustered_chunk_data, _p_declustered_chunk_mtx]
        (const cluster_event& seed, const ClusterInfo& info)
        {
            int _y = (int)seed.ry;
            int _x = (int)seed.rx;

            // Guard against a corrupted/out-of-range row or column (e.g. a noise hit
            // with garbage ToA/position near a ToA-overflow reset) landing outside
            // the real scan. Letting it through was a real crash before this bounds
            // check existed (found via this feature's own full-512x512-scan
            // verification run in fourd_conversion.ipynb -- n_threads=8 crashed
            // partway through). Drop it, same treatment as an unresolved (N<=0)
            // cluster.
            if (_y < 0 || _y >= (int)this->ny || _x < 0 || _x >= (int)this->nx_scan_bin) return;

            int64_t n_electrons = _p_electron_lut
                ? resolve_electron_count_lut(info.tot_sum, info.size, *_p_electron_lut)
                : resolve_electron_count(info.tot_sum, _tot_per_electron);
            if (n_electrons <= 0) return;

            int _kx = (int)std::lround(info.kx);
            int _ky = (int)std::lround(info.ky);
            if (_kx < 0) _kx = 0; else if (_kx >= (int)this->n_cam) _kx = (int)this->n_cam - 1;
            if (_ky < 0) _ky = 0; else if (_ky >= (int)this->n_cam) _ky = (int)this->n_cam - 1;

            (*this->p_counts_data)[(uint64_t)_y * (uint64_t)this->nx_scan_bin + (uint64_t)_x] += (uint64_t)n_electrons;

            // Every chunk-group has its OWN dedicated, never-recycled slot for the
            // whole run (see FourD.h's declustered_chunk_data) -- no eviction, no
            // ordering assumption between clusters, no "already flushed, drop the
            // voxel credit" case at all. This replaces an earlier fixed-size
            // (1, then 2 slot) round-robin buffer that silently dropped the 4D
            // voxel credit (keeping only the Dose_image one) for any cluster whose
            // resolution lagged decode by more chunk-groups than the buffer had
            // slots -- the source of the declustered-FourD pixel-trigger residual
            // documented in IMPROVEMENTS.md. Flushed to disk in one pass by
            // FourD::flush_declustered_chunks(), only after cam.terminate()
            // guarantees every cluster in the file has already been resolved.
            int cluster_chunk_id = _y / this->chunksize_scan_bin;
            int local_y = _y - cluster_chunk_id * this->chunksize_scan_bin;
            uint64_t within_chunk_pos = (uint64_t)local_y * (uint64_t)this->nx_scan_bin + (uint64_t)_x;

            std::lock_guard<std::mutex> lock((*_p_declustered_chunk_mtx)[cluster_chunk_id]);
            (*_p_declustered_chunk_data)[cluster_chunk_id][within_chunk_pos * this->diff_pattern_size +
                (uint64_t)(_ky/(int)this->fourD_det_bin*(int)this->n_cam/(int)this->fourD_det_bin + _kx/(int)this->fourD_det_bin)]
                += (uint32_t)n_electrons;
        });

        decluster = true;
        this->b_tot = true;
        functionType = FunctionType::write_declusterer_buffer;
        decluster_thread = std::thread(&Declusterer::run, &declusterer);
        ++n_proc;
    }

    // Must be called after cam.terminate() (guarantees declustering has fully
    // finished) -- flushes whatever's left in the current, not-yet-flushed
    // chunk-group (full or partial; the regular per-cluster flush above only
    // fires when crossing INTO a later group, so the scan's final group never
    // gets an internal trigger of its own). Also reused as-is for pixel-triggered
    // raw FourD (enable_FourD_pixeltrig_flush above) -- same two tracking fields,
    // same "flush whatever's left" logic, regardless of which of the two callers
    // advanced them.
    void finalize_fourD_declustered()
    {
        // Loops (rather than a single flush) because the cluster callback below
        // can now leave UP TO 2 chunk-groups pending simultaneously (using both
        // of chunk_data's existing round-robin slots as a real 2-chunk buffer,
        // not just 1) -- see that callback's own comment for why. A single-shot
        // flush here would silently drop whichever chunk-group wasn't the most
        // recently pending one.
        int total_rows = (int)ny;
        int n_chunks_total = (total_rows + chunksize_scan_bin - 1) / chunksize_scan_bin;
        while (fourD_declustered_chunk_id < n_chunks_total)
        {
            int remaining = total_rows - fourD_declustered_chunk_id * chunksize_scan_bin;
            int this_flush = std::min(remaining, chunksize_scan_bin);
            if (p_fourD_flush_callback) p_fourD_flush_callback(this_flush);
            ++fourD_declustered_chunk_id;
        }
    }

    void enable_Pacbed(std::vector<size_t> *_p_pacbed_data)
    {
        p_pacbed_data = _p_pacbed_data;
        process.push_back(std::bind(&TIMEPIX::pacbed, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::pacbed;
        ++n_proc;
    }

    // Masked-scan-pattern PACBED (see pacbed_mask() above) -- _p_scan_mask is a flat
    // nx*ny 0/1 array selecting which scan positions contribute to the average.
    void enable_Pacbed_mask(std::vector<int> *_p_scan_mask, std::vector<size_t> *_p_pacbed_data)
    {
        p_pacbed_data = _p_pacbed_data;
        pacbed_scan_mask = *_p_scan_mask;
        process.push_back(std::bind(&TIMEPIX::pacbed_mask, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::pacbed_mask;
        ++n_proc;
    }

    // Charge-weighted, declustered PACBED: same idea as enable_var_declustered/
    // enable_vSTEM_declustered above, crediting N (not +1) into the diffraction-plane
    // accumulator at the cluster's ToT-weighted centroid, instead of +1 per raw hit at
    // its raw position. _p_scan_mask (optional, nullptr = whole scan) restricts which
    // clusters count by their SEED hit's scan position, same nx*ny/no-y-flip
    // convention as pacbed_mask() above.
    void enable_Pacbed_declustered(uint64_t _dtime, uint16_t _dspace, int _cluster_range, double _tot_per_electron,
    std::vector<size_t> *_p_pacbed_data,
    int _n_threads, std::vector<int> *_p_clustersize_histogram, std::vector<int> *_p_energy_histogram,
    std::vector<std::vector<int>> *_p_joint_histogram,
    const std::vector<std::vector<int>> *_p_electron_lut = nullptr,
    const std::vector<int> *_p_scan_mask = nullptr)
    {
        p_pacbed_data = _p_pacbed_data;

        // roi_declustered_dummy_file is shared across every *_declustered enable_*
        // method here -- none of them ever open or write through it (they all set
        // disable_file_write(true)); Declusterer::init() just needs an ofstream&.
        declusterer.init(_dtime, _dspace, _cluster_range, (int)nx, (int)ny, /*scan_bin*/1, /*det_bin*/1,
            roi_declustered_dummy_file, _n_threads, _p_clustersize_histogram, _p_energy_histogram);
        declusterer.set_disable_file_write(true);
        declusterer.set_joint_histogram(_p_joint_histogram);

        declusterer.set_cluster_callback([this, _tot_per_electron, _p_electron_lut, _p_scan_mask]
        (const cluster_event& seed, const ClusterInfo& info)
        {
            int64_t n_electrons = _p_electron_lut
                ? resolve_electron_count_lut(info.tot_sum, info.size, *_p_electron_lut)
                : resolve_electron_count(info.tot_sum, _tot_per_electron);
            if (n_electrons <= 0) return;

            if (_p_scan_mask)
            {
                uint64_t _probe_position = (uint64_t)seed.ry * this->nx + (uint64_t)seed.rx;
                if (_probe_position >= _p_scan_mask->size() || (*_p_scan_mask)[_probe_position] == 0) return;
            }

            int _kx = (int)std::lround(info.kx);
            int _ky = (int)std::lround(info.ky);
            if (_kx < 0) _kx = 0; else if (_kx >= this->n_cam) _kx = this->n_cam-1;
            if (_ky < 0) _ky = 0; else if (_ky >= this->n_cam) _ky = this->n_cam-1;

            (*this->p_pacbed_data)[_ky*this->n_cam+_kx] += (size_t)n_electrons;
        });

        decluster = true;
        this->b_tot = true; // required so parse_event_w_tot (not parse_event) runs, giving real ToT values
        functionType = FunctionType::write_declusterer_buffer;
        decluster_thread = std::thread(&Declusterer::run, &declusterer);
        ++n_proc;
    }

    void enable_var(std::vector<size_t> (*_p_var_data)[2],std::array<float, 2> _offset, int inner_radius, int outer_radius)
    {
        p_var_data = _p_var_data;
        offset = _offset;
        inner_radius_sqr_var = inner_radius*inner_radius;
        outer_radius_sqr_var = outer_radius*outer_radius;
        process.push_back(std::bind(&TIMEPIX::var, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        // p_images.push_back(p_var_data);
        functionType = FunctionType::var;
        ++n_proc;
        // ++n_images;
    }

    void enable_tcBF(std::vector<int> *_p_detector_mask,std::vector<size_t> *_p_BF_image, std::vector<std::vector<size_t>> *_p_tcBF_stack)
    {
        p_tcbf_detector_mask = _p_detector_mask;
        p_tcbf_BF_image = _p_BF_image;
        p_tcBF_stack = _p_tcBF_stack;
        functionType = FunctionType::tcBF;
        ++n_proc;

    }

    void enable_roi(std::vector<std::vector<uint64_t>> *_p_roi_scan_image_stack,std::vector<std::vector<uint64_t>> *_p_roi_diffraction_pattern_stack,
    std::vector<uint64_t> *_p_roi_scan_image,std::vector<uint64_t> *_p_roi_diffraction_pattern,int _lower_left[2] , int _upper_right[2])
    {
        p_roi_scan_image_stack = _p_roi_scan_image_stack;
        p_roi_diffraction_pattern_stack = _p_roi_diffraction_pattern_stack;
        p_roi_scan_image = _p_roi_scan_image;
        p_roi_diffraction_pattern = _p_roi_diffraction_pattern;
        lower_left[0] = _lower_left[0];
        lower_left[1] = _lower_left[1];
        upper_right[0] = _upper_right[0];
        upper_right[1] = _upper_right[1];
        L_0 = upper_right[0]-lower_left[0];
        L_1 = upper_right[1]-lower_left[1];
        process.push_back(std::bind(&TIMEPIX::roi, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::roi;
        ++n_proc;
        // this->b_tot = true;
    }

    void enable_roi_mask(std::vector<std::vector<int>> *_p_roi_mask,std::vector<std::vector<uint64_t>> *_p_roi_scan_image_stack,std::vector<std::vector<uint64_t>> *_p_roi_diffraction_pattern_stack,
    std::vector<uint64_t> *_p_roi_scan_image,std::vector<uint64_t> *_p_roi_diffraction_pattern)
    {
        p_roi_scan_image_stack = _p_roi_scan_image_stack;
        p_roi_diffraction_pattern_stack = _p_roi_diffraction_pattern_stack;
        p_roi_scan_image = _p_roi_scan_image;
        p_roi_diffraction_pattern = _p_roi_diffraction_pattern;
        mask_roi = *_p_roi_mask;

        process.push_back(std::bind(&TIMEPIX::roi_mask, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::roi_mask;
        ++n_proc;

    }

    template <typename T>
    void enable_roi_4D(std::shared_ptr<Roi4D<T>> _p_roi_4D,std::vector<uint64_t> *_p_roi_scan_image,
    std::vector<uint64_t> *_p_roi_diffraction_pattern, int _lower_left[2] , int _upper_right[2],int _det_bin)
    {
        p_roi_4D = _p_roi_4D;
        det_bin = _det_bin;
        p_roi_scan_image = _p_roi_scan_image;
        p_roi_diffraction_pattern = _p_roi_diffraction_pattern;
        lower_left[0] = _lower_left[0];
        lower_left[1] = _lower_left[1];
        upper_right[0] = _upper_right[0];
        upper_right[1] = _upper_right[1];
        L_0 = upper_right[0]-lower_left[0];
        L_1 = upper_right[1]-lower_left[1];
        process.push_back(std::bind(&TIMEPIX::roi_4D, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::roi_4D;
        ++n_proc;
    }

    // Charge-weighted, declustered ROI extraction: instead of counting every raw pixel
    // activation as one electron (see roi_4D above), raw hits are first grouped into
    // physical-electron clusters by the same Declusterer used for .electron conversion
    // (enable_electron below), then each cluster is resolved to N>=0 electrons from its
    // total deposited charge (see ClusterResolver.hpp) and credited at its ToT-weighted
    // centroid position. b_roi_4D mirrors Roi::b_ROI_4D -- when false, only the scan
    // image / diffraction pattern are accumulated, matching enable_roi_4D's own scope.
    template <typename T>
    void enable_roi_declustered(uint64_t _dtime, uint16_t _dspace, int _cluster_range, double _tot_per_electron,
    std::shared_ptr<Roi4D<T>> _p_roi_4D, bool _b_roi_4D,
    std::vector<uint64_t> *_p_roi_scan_image, std::vector<uint64_t> *_p_roi_diffraction_pattern,
    int _lower_left[2], int _upper_right[2], int _det_bin, int _n_threads,
    std::vector<int> *_p_clustersize_histogram, std::vector<int> *_p_energy_histogram,
    std::vector<std::vector<int>> *_p_joint_histogram,
    const std::vector<std::vector<int>> *_p_electron_lut = nullptr)
    {
        if (_b_roi_4D) p_roi_4D = _p_roi_4D;
        det_bin = _det_bin;
        p_roi_scan_image = _p_roi_scan_image;
        p_roi_diffraction_pattern = _p_roi_diffraction_pattern;
        lower_left[0] = _lower_left[0];
        lower_left[1] = _lower_left[1];
        upper_right[0] = _upper_right[0];
        upper_right[1] = _upper_right[1];
        L_0 = upper_right[0]-lower_left[0];
        L_1 = upper_right[1]-lower_left[1];

        // Declusterer.init() requires an ofstream reference for its (unused here) .electron
        // write path; roi_declustered_dummy_file is never opened, and set_disable_file_write
        // below makes sure Declusterer never attempts to write through it. _n_threads controls
        // how many ring-buffer slots get declustered in parallel (see Declusterer::init) --
        // pass Roi.n_threads > 1 for a real speedup; the default of 1 preserves the old,
        // effectively-serial behavior.
        declusterer.init(_dtime, _dspace, _cluster_range, (int)nx, (int)ny, /*scan_bin*/1, /*det_bin*/1,
            roi_declustered_dummy_file, _n_threads, _p_clustersize_histogram, _p_energy_histogram);
        declusterer.set_disable_file_write(true);
        declusterer.set_joint_histogram(_p_joint_histogram);

        int* p_lower_left = lower_left;
        int* p_upper_right = upper_right;
        declusterer.set_cluster_callback([this, _b_roi_4D, _tot_per_electron, p_lower_left, p_upper_right, _p_electron_lut]
        (const cluster_event& seed, const ClusterInfo& info)
        {
            int64_t n_electrons = _p_electron_lut
                ? resolve_electron_count_lut(info.tot_sum, info.size, *_p_electron_lut)
                : resolve_electron_count(info.tot_sum, _tot_per_electron);
            if (n_electrons <= 0) return;

            // Same scan-position mapping (including the nx/ny convention) as roi_4D()
            // above, so decluster=True/False runs land on identical scan pixels.
            int _x = (int)seed.rx;
            int _y = (int)this->nx - (int)seed.ry;
            if (_x >= p_lower_left[0] && _x < p_upper_right[0] && _y > p_lower_left[1] && _y <= p_upper_right[1])
            {
                int _kx = (int)std::lround(info.kx);
                int _ky = (int)std::lround(info.ky);
                if (_kx < 0) _kx = 0; else if (_kx >= this->n_cam) _kx = this->n_cam-1;
                if (_ky < 0) _ky = 0; else if (_ky >= this->n_cam) _ky = this->n_cam-1;

                (*this->p_roi_diffraction_pattern)[_ky*this->n_cam+_kx] += (uint64_t)n_electrons;
                (*this->p_roi_scan_image)[(this->L_1 - (_y-p_lower_left[1])) * this->L_0 + (_x-p_lower_left[0])] += (uint64_t)n_electrons;
                if (_b_roi_4D) this->p_roi_4D->increment_by(this->L_1 - (_y-p_lower_left[1]), _x-p_lower_left[0], _ky/this->det_bin, _kx/this->det_bin, (uint64_t)n_electrons);
            }
        });

        decluster = true;
        this->b_tot = true; // required so parse_event_w_tot (not parse_event) runs, giving real ToT values
        functionType = FunctionType::write_declusterer_buffer;
        decluster_thread = std::thread(&Declusterer::run, &declusterer);
        ++n_proc;
    }

    // Charge-weighted declustered vSTEM: same idea as enable_roi_declustered above,
    // but crediting N (not +1) to the annular-detector dose image p_stem_data at the
    // cluster's ToT-weighted centroid. Only the single-annulus case (enable_vSTEM) is
    // covered -- enable_multi_vSTEM/enable_mask_vSTEM are out of scope, matching how
    // Roi's own decluster feature only covers enable_roi/enable_roi_4D, not
    // enable_roi_mask. Unlike Roi, there is no y-flip here: vstem()'s raw callback
    // indexes probe_position directly (see vstem() above), so the declustered path
    // reconstructs it the same plain way from the cluster seed's (rx, ry).
    void enable_vSTEM_declustered(uint64_t _dtime, uint16_t _dspace, int _cluster_range, double _tot_per_electron,
    std::array<float, 2> *_p_radius_sqr, std::array<float, 2> *_p_offset, std::vector<std::vector<size_t>> *_p_stem_data,
    int _n_threads, std::vector<int> *_p_clustersize_histogram, std::vector<int> *_p_energy_histogram,
    std::vector<std::vector<int>> *_p_joint_histogram,
    const std::vector<std::vector<int>> *_p_electron_lut = nullptr)
    {
        p_stem_data = _p_stem_data;
        in_radius_sqr = (int)(*_p_radius_sqr)[0];
        out_radius_sqr = (int)(*_p_radius_sqr)[1];
        x_offset = (int)(*_p_offset)[0];
        y_offset = (int)(*_p_offset)[1];

        // roi_declustered_dummy_file is shared across every *_declustered enable_*
        // method here -- none of them ever open or write through it (they all set
        // disable_file_write(true)); Declusterer::init() just needs an ofstream&.
        declusterer.init(_dtime, _dspace, _cluster_range, (int)nx, (int)ny, /*scan_bin*/1, /*det_bin*/1,
            roi_declustered_dummy_file, _n_threads, _p_clustersize_histogram, _p_energy_histogram);
        declusterer.set_disable_file_write(true);
        declusterer.set_joint_histogram(_p_joint_histogram);

        declusterer.set_cluster_callback([this, _tot_per_electron, _p_electron_lut]
        (const cluster_event& seed, const ClusterInfo& info)
        {
            int64_t n_electrons = _p_electron_lut
                ? resolve_electron_count_lut(info.tot_sum, info.size, *_p_electron_lut)
                : resolve_electron_count(info.tot_sum, _tot_per_electron);
            if (n_electrons <= 0) return;

            int _kx = (int)std::lround(info.kx);
            int _ky = (int)std::lround(info.ky);
            if (_kx < 0) _kx = 0; else if (_kx >= this->n_cam) _kx = this->n_cam-1;
            if (_ky < 0) _ky = 0; else if (_ky >= this->n_cam) _ky = this->n_cam-1;

            int _d2 = (_kx - this->x_offset)*(_kx - this->x_offset) + (_ky - this->y_offset)*(_ky - this->y_offset);
            if (_d2 >= this->in_radius_sqr && _d2 <= this->out_radius_sqr)  // inclusive inner radius, see CPP_EVENTEM_BUGS.md #6
            {
                uint64_t _probe_position = (uint64_t)seed.ry * this->nx + (uint64_t)seed.rx;
                (*this->p_stem_data)[seed.id_image][_probe_position] += (size_t)n_electrons;
            }
        });

        decluster = true;
        this->b_tot = true;
        functionType = FunctionType::write_declusterer_buffer;
        decluster_thread = std::thread(&Declusterer::run, &declusterer);
        ++n_proc;
    }

    // Charge-weighted declustered Var: same idea, crediting N * (squared radial
    // deviation) instead of the raw path's per-hit +=, since var()'s accumulator is a
    // running sum-of-squares, not a simple count -- N electrons sharing one resolved
    // position contribute that same per-event term N times. Same no-y-flip note as
    // enable_vSTEM_declustered above.
    void enable_var_declustered(uint64_t _dtime, uint16_t _dspace, int _cluster_range, double _tot_per_electron,
    std::vector<size_t> (*_p_var_data)[2], std::array<float, 2> _offset, int _inner_radius, int _outer_radius,
    int _n_threads, std::vector<int> *_p_clustersize_histogram, std::vector<int> *_p_energy_histogram,
    std::vector<std::vector<int>> *_p_joint_histogram,
    const std::vector<std::vector<int>> *_p_electron_lut = nullptr)
    {
        p_var_data = _p_var_data;
        offset = _offset;
        inner_radius_sqr_var = _inner_radius * _inner_radius;
        outer_radius_sqr_var = _outer_radius * _outer_radius;

        declusterer.init(_dtime, _dspace, _cluster_range, (int)nx, (int)ny, /*scan_bin*/1, /*det_bin*/1,
            roi_declustered_dummy_file, _n_threads, _p_clustersize_histogram, _p_energy_histogram);
        declusterer.set_disable_file_write(true);
        declusterer.set_joint_histogram(_p_joint_histogram);

        declusterer.set_cluster_callback([this, _tot_per_electron, _p_electron_lut]
        (const cluster_event& seed, const ClusterInfo& info)
        {
            int64_t n_electrons = _p_electron_lut
                ? resolve_electron_count_lut(info.tot_sum, info.size, *_p_electron_lut)
                : resolve_electron_count(info.tot_sum, _tot_per_electron);
            if (n_electrons <= 0) return;

            int _kx = (int)std::lround(info.kx);
            int _ky = (int)std::lround(info.ky);
            if (_kx < 0) _kx = 0; else if (_kx >= this->n_cam) _kx = this->n_cam-1;
            if (_ky < 0) _ky = 0; else if (_ky >= this->n_cam) _ky = this->n_cam-1;

            int _d2 = (_kx - this->offset[0])*(_kx - this->offset[0]) + (_ky - this->offset[1])*(_ky - this->offset[1]);
            if (_d2 > this->inner_radius_sqr_var && _d2 <= this->outer_radius_sqr_var)
            {
                uint64_t _probe_position = (uint64_t)seed.ry * this->nx + (uint64_t)seed.rx;
                (*this->p_var_data)[seed.id_image][_probe_position] += (size_t)n_electrons * (uint64_t)_d2;
            }
        });

        decluster = true;
        this->b_tot = true;
        functionType = FunctionType::write_declusterer_buffer;
        decluster_thread = std::thread(&Declusterer::run, &declusterer);
        ++n_proc;
    }

    void enable_electron(std::ofstream& _p_file, bool _decluster, uint64_t _dtime, uint16_t _dspace, int _cluster_range, int _x_crop, int _y_crop,
    int _scan_bin_electron, int _det_bin_electron, int _n_threads, std::vector<int> *_p_clustersize_histogram,std::vector<int> *_p_energy_histogram)
    {
        decluster = _decluster;
        if (decluster) declusterer.init(_dtime, _dspace, _cluster_range,_x_crop,_y_crop,_scan_bin_electron,_det_bin_electron,_p_file,_n_threads,_p_clustersize_histogram,_p_energy_histogram);
        p_file = &_p_file;
        x_crop = _x_crop;
        y_crop = _y_crop;
        scan_bin_electron = _scan_bin_electron;
        det_bin_electron = _det_bin_electron;
        if (decluster) functionType = FunctionType::write_declusterer_buffer;
        else functionType = FunctionType::write_electron;
        if (decluster) decluster_thread = std::thread(&Declusterer::run, &declusterer);
        ++n_proc;        
        this->b_tot = true;
    } 

    #ifdef GPRI_OPTION_ENABLED
    void enable_GPRI(std::vector<std::shared_ptr<std::vector<int>>> *_p_k_indices_vec,int detector_bin, int scan_bin,
    std::vector<std::vector<uint64_t>> *_p_N_electrons_map_scangrid)
    {
        GPRI_enabled = true;
        p_k_indices_vec = _p_k_indices_vec;
        GPRI_detector_bin = detector_bin;
        GPRI_scan_bin = scan_bin;
        GPRI_nxy_scan_bin = nxy/(scan_bin*scan_bin);
        GPRI_cam_bin = n_cam/detector_bin;
        p_N_electrons_map_scangrid = _p_N_electrons_map_scangrid;
    
        process.push_back(std::bind(&TIMEPIX::GPRI, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::GPRI;
        ++n_proc;
    }
    #endif

    void enable_information(std::vector<float> *_p_information_image, std::vector<float> *_p_probability_distribution, std::vector<float> *_p_count_image)
    {
        p_information_image = _p_information_image;
        p_probability_distribution = _p_probability_distribution;
        p_count_image = _p_count_image;
        process.push_back(std::bind(&TIMEPIX::information, this,std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        functionType = FunctionType::information;
        ++n_proc;
    }

    //-------------------------------------------------------------------------------------------------

    void terminate()
    {
        while ((!read_thread.joinable()) || (!proc_thread.joinable())) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        read_thread.join();
        proc_thread.join();
        if (decluster)
        {
            declusterer.still_reading = false;
            while (declusterer.still_processing || declusterer.still_writing) {std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
            declusterer.terminate();
            decluster_thread.join();
        }
        this->endtime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        processing_rate = n_events_processed / ((endtime - starttime) / 1e9);
        std::cout << n_events_processed << " events processed at " << processing_rate/1000000 << " M events/s " << std::endl; 
        if (decluster) std::cout << declusterer.n_electrons_kept << " electrons kept" << std::endl;
        std::cout << "reading waited " << read_wait << " times, processing waited " << process_wait << " times"<< std::endl;
        std::cout << "atomic counter: " << atomic_counter << std::endl;
        delete [] buffer;
    };

    float get_processing_rate(){
        return processing_rate;
    }

    //-------------------------------------------------------------------------------------------------
    // Variables
    //-------------------------------------------------------------------------------------------------
    int nx;
    int ny;
    int nxy;
    int n_cam;
    bool *b_cumulative;
    // bool b_continuous;
    int repetitions;
    bool b_tot = false;
    bool repetitions_reached = false;

    // Multi-process file-splitting support (CHEETAH/.tpx3 only -- see
    // Cheetah.hpp). Opt-in, default values preserve the normal whole-file
    // behavior exactly. Lets one worker process only a byte-range/line-range
    // slice of a file -- seeking straight to file_byte_offset instead of reading
    // from the start, seeding line_number_offset as the real (absolute) starting
    // scan line instead of 0, and stopping once stop_at_line is reached instead
    // of running to the true end of the scan -- so N independent worker
    // processes can each cover a disjoint band of scan lines in parallel and
    // have their partial output images summed together afterward (the bands are
    // spatially disjoint, so summing is the same as placing each band's own
    // pixels -- no partial overlap to reconcile).
    uintmax_t file_byte_offset = 0;
    int line_number_offset = 0;
    int stop_at_line = -1; // -1 = use the normal ny*repetitions end

    // Multi-process file-splitting: dt/rise_t calibration seed. A worker
    // resuming mid-file starts with no TDC history -- parse_event()'s
    // probe_position formula ((toa - rise_t[chip_id]*2) / dt) needs both, and
    // until a real TDC rise+fall pair completes for each chip after resuming,
    // dt/rise_t fall back to CHEETAH's constructor guess (the same fallback
    // that produces the small, already-accepted residual at real scan
    // position (0,0) for a normal whole-file run) -- filtering out nearly
    // every real hit of that worker's entire first line via the
    // `_probe_position < nx` gate, confirmed via mp_vstem.ipynb (~90% loss on
    // exactly the first row of every interior/last slice, zero mismatch
    // everywhere else). find_line_checkpoints() already computes the
    // correctly-calibrated dt/rise_t as a byproduct of finding the checkpoint
    // byte position; seeding them here lets a resumed worker skip that
    // warm-up window entirely. seed_dt == 0 means "no seed provided, use the
    // old fallback-then-calibrate behavior" (dt is never legitimately 0
    // otherwise).
    uint64_t seed_dt = 0;
    std::vector<uint64_t> seed_rise_t = {0, 0, 0, 0};
    // Multi-process file-splitting: the REAL root cause of the boundary-row
    // undercount (found via diag_target_row_seen/diag_stop_dropped
    // instrumentation -- neither the stop-line guard nor dt/rise_t
    // calibration explained it). process_buffer() only ever dispatches a hit
    // to parse_event() while rise_fall[chip_id] is true (i.e. that chip is
    // currently between its own TDC rise and fall -- "inside" an active
    // line). reset() force-resets rise_fall[i]=false for all 4 chips
    // uniformly, but at an intermediate checkpoint only the SLOWEST chip
    // (the one whose fall defined the checkpoint) genuinely has rise_fall
    // false there -- the other 3 chips are, by definition, already ahead,
    // meaning they already rose for the target line and are still actively
    // emitting real hits for it when the checkpoint byte is reached. Forcing
    // their rise_fall to false silently drops every one of those hits until
    // each chip's own NEXT rise (for the line after the target one) turns it
    // back on -- explaining why only the checkpoint's own target row loses
    // ~90% of its data while every later row is byte-for-byte exact.
    // seed_rise_fall (0/1 per chip) restores each chip's true state.
    std::vector<int> seed_rise_fall = {0, 0, 0, 0};
    // Multi-process file-splitting: a SECOND real bug found via the N=8/16
    // benchmark sweep in mp_vstem_benchmark.ipynb, after seed_rise_fall alone
    // fixed the N=4 case. The 3 chips ahead of the slowest one at a
    // checkpoint are not always exactly 1 line ahead -- sometimes 2 or more
    // (varies by which chip is slowest and by scan position, not a fixed
    // pattern). reset() used to force EVERY chip's line_count to
    // line_number_offset uniformly -- correct only for the slowest chip. A
    // chip that's really 2+ lines ahead then has its own early hits (until
    // its next real TDC fall corrects line_count[chip_id]) misattributed to
    // rows line_number_offset/line_number_offset+1/... instead of its true,
    // higher row -- a substantial (several-percent) undercount at the
    // checkpoint's target row plus a small sum-preserving shift at the row(s)
    // just after it, exactly what was observed. -1 per chip means "not
    // seeded, use line_number_offset" (keeps old behavior for a normal
    // worker-0/whole-file run, where line_number_offset is already correct
    // for every chip).
    std::vector<int> seed_line_count = {-1, -1, -1, -1};

    // Multi-process file-splitting, pixel-trigger (smart-scan) mode
    // (CHEETAH_pixeltrig only): the per-chip analogue of seed_line_count, but
    // for probe_count_chip[] (each chip's own trigger count) rather than
    // line_count[] -- pixel-trigger mode resolves scan position directly as
    // pattern[probe_count_chip[chip_id]], with no dt/rise_t dwell-time
    // calibration involved at all (unlike raster mode), so seed_dt/
    // seed_rise_t simply don't apply here. -1 per chip means "not seeded, use
    // line_number_offset" (same fallback convention as seed_line_count).
    std::vector<int> seed_probe_count_chip = {-1, -1, -1, -1};

    // Multi-process file-splitting, pixel-trigger (smart-scan) mode only:
    // which chip's data stream a resumed worker's very first packets belong
    // to. Cheetah_pixeltrig::chip_id is set only by header packets, which can
    // be arbitrarily far from any given checkpoint (confirmed directly: 40+
    // consecutive TDC/event packets with zero headers in between, right at a
    // real checkpoint byte offset) -- so a worker seeing no header of its own
    // for a long stretch after resuming has no way to know which chip its
    // first packets belong to, and silently misattributes them all to
    // whatever chip_id happens to default to until its own first header
    // finally arrives. -1 (unseeded) keeps the old default (0) for a normal
    // worker-0/whole-file run, where a header always appears almost
    // immediately.
    int seed_chip_id = -1;

    uint64_t toa;
    uint16_t tot;

    float processing_rate;

    //------------------------
    // VSTEM
    std::vector<atomwrapper<int>> *p_atomic_stem_data;
    std::vector<std::vector<size_t>> *p_stem_data;
    int d2;
    int in_radius_sqr;
    int out_radius_sqr;
    int x_offset;
    int y_offset;

    // masked VSTEM
    std::vector<int> detector_mask;

    // multi VSTEM
    int n_detectors;
    std::vector<std::array<float, 2>> radia_sqr;
    std::vector<std::array<float, 2>> offsets;

    // ricom 
    // std::vector<size_t> (*p_dose_data)[2];
    // std::vector<size_t> (*p_sumx_data)[2];
    // std::vector<size_t> (*p_sumy_data)[2];
    std::vector<std::vector<size_t>> *p_dose_data;
    std::vector<std::vector<size_t>> *p_sumx_data;
    std::vector<std::vector<size_t>> *p_sumy_data;
    std::vector<int> com_mask;

    // tcBF
    std::vector<int> *p_tcbf_detector_mask;
    std::vector<size_t> *p_tcbf_BF_image;
    std::vector<std::vector<size_t>> *p_tcBF_stack;


    // GPRI
    #ifdef GPRI_OPTION_ENABLED
    bool GPRI_enabled = false;
    int GPRI_detector_bin;
    int GPRI_scan_bin; 
    uint16_t GPRI_cam_bin;
    uint64_t GPRI_nxy_scan_bin;
    std::vector<std::shared_ptr<std::vector<int>>> *p_k_indices_vec;
    std::vector<std::vector<uint64_t>> *p_N_electrons_map_scangrid;
    #endif


    // FourD
    std::vector<uint64_t> *p_counts_data;
    std::vector<uint8_t> (*p_fourDchunk_data_8)[2];
    std::vector<uint16_t> (*p_fourDchunk_data_16)[2];
    std::vector<uint32_t> (*p_fourDchunk_data_32)[2];
    std::mutex* mtx = nullptr; //2 mutexes for 2 chunks

    // Raw pixel-triggered FourD's exact chunk buffer (see
    // enable_FourD_pixeltrig_flush's comment above) -- non-null selects this path
    // in count_chunked_8/16/32 and short-circuits advance_fourD_pixeltrig_chunk's
    // now-unnecessary eviction/rejection logic entirely. Exactly one of these
    // three is ever set for a given run (matching the BitDepth the caller chose).
    std::vector<std::vector<uint8_t>> *p_full_chunk_data_8 = nullptr;
    std::vector<std::vector<uint16_t>> *p_full_chunk_data_16 = nullptr;
    std::vector<std::vector<uint32_t>> *p_full_chunk_data_32 = nullptr;
    std::vector<std::mutex> *p_full_chunk_mtx = nullptr;
    size_t id_chunk;
    size_t fourD_det_bin;
    size_t fourD_scan_bin;
    size_t n_cam_det_bin;
    size_t diff_pattern_size;
    int x_pp;
    int y_pp;
    int bin_probe_position;
    int chunksize_scan_bin;
    int nx_scan_bin;

    // Declustered FourD only: flush timing for this mode is driven by the
    // declustered cluster stream itself (see enable_FourD_declustered below), not
    // by line_processor()'s raw-decode-progress trigger the way the raw path is --
    // declustering can lag decode significantly, and FourD's chunk_data buffer is
    // recycled every chunksize lines, so tying flush timing to raw progress could
    // let a buffer be flushed/reused before slower declustering has finished
    // writing into it (see the plan notes this was designed against). Both fields
    // are only touched from Declusterer's single dedicated writer thread (clusters
    // arrive in strict order there), so no additional synchronization is needed
    // beyond the existing per-slot `mtx` already guarding the buffer itself.
    std::function<void(int)> p_fourD_flush_callback = nullptr;
    int fourD_declustered_chunk_id = 0;

    // PACBED
    std::vector<size_t> *p_pacbed_data;
    std::vector<int> pacbed_scan_mask;

    // Variance
    std::array<float, 2> offset;
    std::vector<size_t> (*p_var_data)[2];
    int inner_radius_sqr_var;
    int outer_radius_sqr_var;

    // ROI
    std::shared_ptr<Roi4DBase> p_roi_4D;
    std::vector<uint64_t> *p_roi_scan_image;
    std::vector<uint64_t> *p_roi_diffraction_pattern;
    std::vector<std::vector<uint64_t>> *p_roi_scan_image_stack;
    std::vector<std::vector<uint64_t>> *p_roi_diffraction_pattern_stack;
    int lower_left[2];
    int upper_right[2];
    int L_0;
    int L_1;
    int det_bin; 
    std::vector<std::vector<int>> mask_roi;

    //Information
    std::vector<float> *p_information_image;
    std::vector<float> *p_probability_distribution;
    std::vector<float> *p_count_image;

    //Electron
    std::ofstream* p_file;
    dtype_Electron electron;
    int x_crop;
    int y_crop;
    int scan_bin_electron;
    int det_bin_electron;

    // ROI declustered extraction (see enable_roi_declustered): never opened, only exists
    // to satisfy Declusterer::init()'s ofstream& parameter when file writing is disabled.
    std::ofstream roi_declustered_dummy_file;

    // declustering
    bool decluster = false;
    Declusterer declusterer;
    std::thread decluster_thread;

    // atomic counter
    std::atomic<uint64_t> atomic_counter = 0;
    //-----------------------------------------------------------------------------------------------------------------------

    int *p_processor_line;
    int *p_preprocessor_line;
    int mode;
    std::string file_path;
    SocketConnector socket;

    // std::array<std::array<event, buffer_size>, n_buffer> buffer; // stack, should be faster but limited to aproxx 1MB buffer, either too small buff for efficient read or too little buff
    std::array<event, buffer_size> *buffer = new std::array<event, buffer_size>[n_buffer]; // heap
    std::array<void* ,n_buffer> mmap_buffer = std::array<void*, n_buffer>(); // mmap buffer

    std::vector<std::function<void(uint64_t, uint16_t, uint16_t, uint16_t)>> process;
    TIMEPIX<event, buffer_size, n_buffer>::FunctionType functionType;
    // Set once, here, whenever functionType becomes count_chunked_8/16/32 (see
    // the 3 enable_FourD overloads below) -- lets CHEETAH_pixeltrig::parse_event()
    // check a single bool instead of 3 enum comparisons + 2 boolean ORs on
    // every single event, the overwhelming majority of which aren't FourD at
    // all (found while profiling the pixel-trigger/raster per-event throughput
    // gap: this check ran unconditionally for every functionType, paying its
    // full cost even for plain vSTEM/Roi/etc. runs that never need it).
    bool is_fourD_chunked_type = false;

    int n_proc = 0;
    // std::vector<std::vector<size_t> (*)[2]> p_images; 
    // int n_images = 0;

    uint64_t starttime;
    uint64_t endtime;

    //bemchmarking
    std::chrono::high_resolution_clock::time_point BM_start;
    std::chrono::high_resolution_clock::time_point BM_stop;
    float BM_duration_sum = 0;
    int BM_count = 0;

    float BM_duration_sum_2 = 0;
    int BM_count_2 = 0;

    int read_wait = 0;
    int process_wait = 0;

    TIMEPIX(
        int &nx,
        int &ny,
        bool *b_cumulative,
        int repetitions,
        int *p_processor_line,
        int *p_preprocessor_line,
        int &mode,
        std::string &file_path,  
        SocketConnector socket
    ) : 
        nx(nx),
        ny(ny),
        b_cumulative(b_cumulative),
        repetitions(repetitions),
        p_processor_line(p_processor_line),
        p_preprocessor_line(p_preprocessor_line),
        mode(mode),
        file_path(file_path), 
        socket(socket)
    {
        nxy = nx*ny;
        n_proc = 0;
        // n_images = 0;
    }
};
#endif // TIMEPIX_H