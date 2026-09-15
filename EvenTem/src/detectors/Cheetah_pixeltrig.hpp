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

#ifndef CHEETAH_PIXELTRIG_H
#define CHEETAH_PIXELTRIG_H

#ifdef __GNUC__
#define PACK(__Declaration__) __Declaration__ __attribute__((__packed__))
#endif

#ifdef _MSC_VER
#define PACK(__Declaration__) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
#endif

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <atomic>
#include <vector>
#include <array>
#include <thread>
#include <chrono>

#include "FileConnector.h"
#include "Timepix.hpp"

template <typename event, int buffer_size, int n_buffer>
class CHEETAH_pixeltrig : public TIMEPIX<event, buffer_size, n_buffer>
{
private:
    // header
    // Cheetah.hpp (raster) defaults this to 0 -- this class didn't, leaving it
    // at whatever garbage was in memory at construction until the first real
    // header packet is seen. Harmless for a worker starting at byte 0 (a
    // header packet always appears almost immediately), but a worker resuming
    // mid-file via a multi-process split checkpoint can see TDC/event packets
    // before its own first header -- during that window, an uninitialized
    // chip_id indexing rise_fall[chip_id]/probe_count_chip[chip_id] (both
    // size 4) is undefined behavior: usually just misattributes a handful of
    // early hits to the wrong chip, but if the garbage value happens to be
    // >=4, it's an out-of-bounds write (reproduced directly: crashed once an
    // unrelated member was added nearby, shifting what memory the overrun
    // landed on). Matches Cheetah.hpp's own default now.
    int chip_id = 0;
    uint64_t tpx_header = 861425748; //(b'TPX3', 'little')

    std::string &pattern_file;
    // uint32_t, not uint64_t: values are linearized scan indices (max nx*ny,
    // comfortably under 2^32) -- halves this array's memory footprint, which
    // matters here since (unlike raster mode's pure-arithmetic position calc)
    // every single event requires a real memory load from this array, keyed
    // by each chip's own independently-advancing trigger count.
    std::vector<uint32_t> pattern;
    uint64_t probe_count;

    // TDC
    uint64_t rise_t[4];
    uint64_t fall_t[4];
    bool rise_fall[4] = {false, false, false, false};
    int line_count[4] = {0, 0, 0, 0};
    int probe_count_chip[4] = {0, 0, 0, 0};

    int most_advanced_line = 0;
    int most_advanced_probe_count = 0;

    //buffer
    int type;

    // event
    uint64_t toa = 0;
    uint64_t pack_44;
    int address_multiplier[4] = {1,-1,-1,1};
    int address_bias_x[4] = {256, 511, 255, 0};
    int address_bias_y[4] = {0, 511, 511, 0};

    // overflow correction
    uint64_t prev_toa;
    std::atomic<uint64_t> toa_offset = 0;
    uint64_t toa_overflow_drop = 4294967296;
    int last_offset_line = 0;

    uint64_t prev_tdc = 0;
    uint64_t tdc_offset = 0;
    uint64_t tdc_overflow_drop = 17179869184; //half of 34359738368 = tdc range
    int last_offset_line_tdc = 0;

    // FourD's on-disk chunk buffer is normally flushed/recycled on a cadence tied
    // to line_processor()'s raw-decode-progress trigger, which assumes trigger
    // order tracks real scan row 1:1 -- true for a raster scan, false for a sparse
    // custom pattern (see enable_FourD_pixeltrig_flush's comment in Timepix.hpp for
    // the measured drift). When FourD has opted in (p_fourD_flush_callback set),
    // this tracks flush progress here instead. Returns false if the event's row
    // belongs to an already-flushed-and-recycled chunk-group -- caller must drop
    // the event's 4D-voxel credit (not its Dose_image credit) in that case, same
    // tradeoff already made for declustered FourD (Timepix.hpp's
    // enable_FourD_declustered). No-op (always true) unless FourD called
    // enable_FourD_pixeltrig_flush.
    //
    // REAL FIX (this replaces two earlier, both-broken attempts): the previous
    // approach tracked flush safety via "the slowest chip's CURRENT target row"
    // (pattern[probe_count_chip[c]]), assuming the pattern visits rows in
    // roughly increasing order -- true often enough to mask the bug in casual
    // testing, but false in general for a custom scan pattern (nothing
    // guarantees monotonic row order -- "scan shape" per the original request
    // just means SOME geometric layout, not necessarily raster order). When a
    // pattern's per-chip CURRENT row estimate jumped past a chunk-group's real
    // extent before every chip had actually finished visiting every position
    // in it, that chunk got flushed-and-evicted early -- confirmed via
    // measure_pixeltrig_drift.ipynb: on a real 40856-trigger/262144-position
    // smart-scan file, this dropped 171,937 events (7% of the total, ~21% of
    // positions undercounted in an earlier, similarly-broken variant) even
    // though the actual chip-to-chip trigger-count drift was tiny (141
    // triggers -- nowhere near large enough on its own to explain that loss).
    // Forcing a single giant "chunk" (no eviction possible) collapsed the
    // mismatch from 55,735 positions down to 60 -- proof the eviction
    // heuristic itself, not chip desync or anything else, was the dominant
    // cause.
    //
    // The actual fix needs no row-order assumption at all: all 4 chips index
    // into the SAME shared pattern array (this->pattern), each with its own
    // independent trigger counter -- so precompute, once, the LAST pattern
    // index at which each chunk-group is EVER referenced
    // (chunk_last_pattern_index). A chunk-group is then provably safe to
    // flush once every chip's own trigger counter has advanced PAST that
    // index -- guaranteeing no chip can ever produce another event for it,
    // regardless of whether the pattern visits rows in order, jumps around,
    // or revisits earlier rows later.
    std::vector<int> chunk_last_pattern_index;
    bool chunk_last_pattern_index_ready = false;

    inline void ensure_chunk_last_pattern_index()
    {
        if (chunk_last_pattern_index_ready) return;
        chunk_last_pattern_index_ready = true;
        int n_chunks = ((int)this->ny + this->chunksize_scan_bin - 1) / this->chunksize_scan_bin;
        chunk_last_pattern_index.assign(n_chunks, -1);
        for (size_t i = 0; i < pattern.size(); ++i)
        {
            int row = (int)(pattern[i] / this->nx);
            int k = row / this->chunksize_scan_bin;
            if (k >= 0 && k < n_chunks) chunk_last_pattern_index[k] = (int)i;
        }
    }

    inline bool advance_fourD_pixeltrig_chunk(uint64_t _probe_position)
    {
        // Full-buffer mode (see FourD::allocate_full_chunk_buffer/
        // enable_FourD_pixeltrig_flush's comment in Timepix.hpp): every
        // chunk-group has its own permanent slot for the whole run, so no event
        // can ever target an "already flushed" one -- count_chunked_N itself
        // picks the correct absolute slot directly from its own probe position,
        // with no eviction/rejection needed here at all.
        if (this->p_full_chunk_data_8 || this->p_full_chunk_data_16 || this->p_full_chunk_data_32) return true;

        if (!this->p_fourD_flush_callback) return true;
        ensure_chunk_last_pattern_index();

        int min_progress = probe_count_chip[0];
        for (int c = 1; c < 4; ++c) if (probe_count_chip[c] < min_progress) min_progress = probe_count_chip[c];

        while (this->fourD_declustered_chunk_id < (int)chunk_last_pattern_index.size() &&
               chunk_last_pattern_index[this->fourD_declustered_chunk_id] < min_progress)
        {
            this->p_fourD_flush_callback(this->chunksize_scan_bin);
            ++this->fourD_declustered_chunk_id;
        }

        // Data placement uses the CURRENT event's own row, which can legitimately
        // be ahead of the flush boundary (its target chunk-group simply hasn't
        // been flushed yet) -- only reject if it's BEHIND the already-flushed one.
        // Rejected events still credit Dose_image (no windowing constraint) via
        // the caller, same small accepted-residual tradeoff as declustered
        // FourD's own cluster-boundary case -- confirmed small in practice
        // (0.12% of events on a real 40856-trigger smart-scan file, once the
        // real bug -- a second, redundant flush trigger firing at the same
        // time as this one, see FourD.cpp's process_data() -- was fixed).
        int event_chunk_id = (int)(_probe_position / this->nx) / this->chunksize_scan_bin;
        return event_chunk_id >= this->fourD_declustered_chunk_id;
    }

    void parse_event(event *packet)
    {
        toa = ((((*packet & 0xFFFF) << 14) + ((*packet >> 30) & 0x3FFF)) << 4) + toa_offset;
        // nxy is a runtime value (not a compile-time power-of-two the compiler
        // could turn into a cheap mask), so this modulo is a real integer
        // division -- computing it once and reusing it (instead of redoing the
        // same division for the array index below) avoids paying that cost
        // twice per event.
        uint64_t _pattern_idx = this->probe_count_chip[chip_id]%this->nxy;
        if (_pattern_idx < pattern.size()-1)
        {
            pack_44 = (*packet >> 44);
            uint64_t _probe_position = pattern[_pattern_idx];
            uint16_t _kx = (address_multiplier[chip_id] * (((pack_44 & 0x0FE00) >> 8) + ((pack_44 & 0x00007) >> 2)) + address_bias_x[chip_id]);
            uint16_t _ky = (address_multiplier[chip_id] * (((pack_44 & 0x001F8) >> 1) + (pack_44 & 0x00003)) + address_bias_y[chip_id]);

            // Was 3 enum comparisons + 2 boolean ORs recomputed on every single
            // event regardless of functionType -- raster mode's parse_event()
            // has no equivalent check at all, since it has no FourD-chunk-drop
            // path. Now a single bool, set once (see is_fourD_chunked_type's
            // declaration in Timepix.hpp) when functionType is actually one of
            // the 3 FourD-chunked modes.
            if (this->is_fourD_chunked_type && !advance_fourD_pixeltrig_chunk(_probe_position))
            {
                // Target chunk-group already flushed and recycled -- credit
                // Dose_image (no windowing constraint) but drop the 4D voxel write
                // that would otherwise corrupt a different, already-written group.
                (*this->p_counts_data)[_probe_position]++;
                ++this->n_events_processed;
                return;
            }

            switch(this->functionType){
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::vstem:
                    this->vstem(_probe_position,_kx,_ky, this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::multi_vstem:
                    this->multi_vstem(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::com:
                    this->com(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::count_chunked_8: 
                    this->count_chunked_8(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::count_chunked_16:
                    this->count_chunked_16(_probe_position,_kx,_ky,this->id_image);
                    break;  
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::count_chunked_32:
                    this->count_chunked_32(_probe_position,_kx,_ky,this->id_image);
                    break;  
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::pacbed:
                    this->pacbed(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::pacbed_mask:
                    this->pacbed_mask(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::var:
                    this->var(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::roi:
                    this->roi(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::roi_mask:
                    // Was missing entirely -- Roi.set_roi_mask() on a pixel-trigger
                    // (smart-scan) camera set functionType=roi_mask correctly (see
                    // Roi.cpp's CHEETAH_PIXELTRIG case, wired the same as CHEETAH),
                    // but this switch had no matching case for it, so the mask branch
                    // silently did nothing -- zero results, no error, no crash.
                    this->roi_mask(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::roi_4D:
                    this->roi_4D(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::write_electron:
                    this->write_electron(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::write_declusterer_buffer:
                    this->write_declusterer_buffer(_probe_position,_kx,_ky,this->id_image,toa,0);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::information:
                    this->information(_probe_position,_kx,_ky,this->id_image);
                    break;
                #ifdef GPRI_OPTION_ENABLED
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::GPRI:
                    this->GPRI(_probe_position,_kx,_ky,this->id_image);
                    break;
                #endif
            }
            ++this->n_events_processed;
        }
    };

    // ToT-aware counterpart to parse_event above, used when b_tot=true (declustered
    // modes only -- see enable_roi_declustered/enable_vSTEM_declustered/
    // enable_var_declustered/enable_FourD_declustered, all of which set b_tot=true
    // and functionType=write_declusterer_buffer before this ever needs to run).
    // Mirrors Cheetah.hpp's parse_event_w_tot exactly for the packet-level ToA fine
    // correction and ToT extraction (hardware packet format, unrelated to trigger
    // mode) -- the only thing that differs from raster mode is _probe_position,
    // still resolved from the pattern file exactly as in parse_event above. Without
    // this, declustering combined with a pattern file would always see tot=0 for
    // every cluster (parse_event hardcodes 0), resolving every cluster to 0
    // electrons -- found and fixed as part of this feature's own verification.
    void parse_event_w_tot(event *packet)
    {
        toa = ((((*packet & 0xFFFF) << 14) + ((*packet >> 30) & 0x3FFF)) << 4) - ((*packet >> 16) & 0xF) + toa_offset;
        this->tot = ((*packet) >> (16 + 4)) & 0x3ff;
        uint64_t _pattern_idx = this->probe_count_chip[chip_id]%this->nxy;
        if (_pattern_idx < pattern.size()-1)
        {
            pack_44 = (*packet >> 44);
            uint64_t _probe_position = pattern[_pattern_idx];
            uint16_t _kx = (address_multiplier[chip_id] * (((pack_44 & 0x0FE00) >> 8) + ((pack_44 & 0x00007) >> 2)) + address_bias_x[chip_id]);
            uint16_t _ky = (address_multiplier[chip_id] * (((pack_44 & 0x001F8) >> 1) + (pack_44 & 0x00003)) + address_bias_y[chip_id]);
            ++this->n_events_processed;

            switch(this->functionType){
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::vstem:
                    this->vstem(_probe_position,_kx,_ky, this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::multi_vstem:
                    this->multi_vstem(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::com:
                    this->com(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::count_chunked_8:
                    this->count_chunked_8(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::count_chunked_16:
                    this->count_chunked_16(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::count_chunked_32:
                    this->count_chunked_32(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::pacbed:
                    this->pacbed(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::pacbed_mask:
                    this->pacbed_mask(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::var:
                    this->var(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::roi:
                    this->roi_ToT(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::roi_4D:
                    // See Cheetah.hpp's identical dispatch for why roi_4D_ToT, not roi_4D.
                    this->roi_4D_ToT(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::write_electron:
                    this->write_electron(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::write_declusterer_buffer:
                    this->write_declusterer_buffer(_probe_position,_kx,_ky,this->id_image,toa*25./16.,this->tot);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::information:
                    this->information(_probe_position,_kx,_ky,this->id_image);
                    break;
                #ifdef GPRI_OPTION_ENABLED
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::GPRI:
                    this->GPRI(_probe_position,_kx,_ky,this->id_image);
                    break;
                #endif
            }
            ++this->n_events_processed;
        }
    };

    void check_toa_overflow()
    {
        if ((prev_toa > toa + toa_overflow_drop) && (this->current_line > 1) && (last_offset_line != this->current_line)) // toa drop bigger than half of toa range --> toa must have overflowed
        {
            toa_offset += 17179869184; 
            last_offset_line = this->current_line;
            std::cout << "toa overflow at line " << this->current_line << std::endl;
        }
        prev_toa = toa;
    };

    inline void schedule_buffer()
    {
        int buffer_id;

        while ((*this->p_processor_line)!=-1)
        {
            if (this->n_buffer_processed < this->n_buffer_filled)
            {
                buffer_id = this->n_buffer_processed % this->n_buf;

                process_buffer(&(this->buffer[buffer_id]));

                if (this->decluster) this->declusterer.set_buffer_read();
 
                ++this->n_buffer_processed;
                *this->p_preprocessor_line = (int)this->current_line;

                check_toa_overflow();
            }
            else
            {
                this->process_wait++;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    };

    inline void process_buffer(std::array<event, buffer_size> *p_buffer)
    {
        // b_tot mirrors Cheetah.hpp's own dispatch: declustered modes set b_tot=true
        // (see enable_*_declustered in Timepix.hpp) and need real ToT values from
        // parse_event_w_tot; every other mode uses the plain, ToT-less parse_event.
        if (this->b_tot)
        {
            for (int j = 0; j < buffer_size; j++)
            {
                type = which_type(&(*p_buffer)[j]);
                if ((type == 2) & rise_fall[chip_id] & (!this->repetitions_reached)) //currently always lose first line because no dwelltime known yet
                {
                    parse_event_w_tot(&(*p_buffer)[j]);
                }
            }
        }
        else
        {
            for (int j = 0; j < buffer_size; j++)
            {
                type = which_type(&(*p_buffer)[j]);
                if ((type == 2) & rise_fall[chip_id] & (!this->repetitions_reached)) //currently always lose first line because no dwelltime known yet
                {
                    parse_event(&(*p_buffer)[j]);
                }
            }
        }
    };

    inline int which_type(event *packet)
    {
        // Event checked FIRST -- the overwhelming majority of packets are hit
        // events, not headers/TDC pulses. The old ordering (header, then TDC,
        // then event last) made every single hit packet fail two comparisons
        // before matching on the third, for no reason -- mirrors Cheetah.hpp's
        // own (already event-first) ordering, which doesn't have this cost.
        // Measured: closed most of a ~2.4x per-event throughput gap between
        // pixel-trigger and raster mode on the same operation (benchmark_
        // pixeltrig_vs_raster.ipynb), previously assumed to just be "more raw
        // data" -- it wasn't; pixel-trigger's real event count was actually
        // lower, it was genuinely slower per event.
        if (*packet >> 60 == 0xb)
        {
            return 2;
        } // event
        else if ((*packet & 0xFFFFFFFF) == tpx_header)
        {
            chip_id = (*packet >> 32) & 0xff;
            return 0;
        } // header
        else if (*packet >> 60 == 0x6)
        {
            process_tdc(packet);
            return 1;
        } // TDC
        else
        {
            return 3;
        } // unknown
    };

    inline void process_tdc(event *packet)
    {
       
        if (((*packet >> 56) & 0x0F) == 15) // TDC1 rise
        {
            rise_fall[chip_id] = true;
            rise_t[chip_id] = ((*packet >> 9) & 0x7FFFFFFFF) + tdc_offset;

            if ((prev_tdc > rise_t[chip_id] + tdc_overflow_drop) && (this->current_line > 1) && (last_offset_line_tdc != this->current_line)) // tdc drop bigger than half of tdc range --> tdc must have overflowed
            {
            tdc_offset += 34359738368; 
            last_offset_line_tdc = this->current_line;
            }

            prev_tdc = rise_t[chip_id];
        }
        else if (((*packet >> 56) & 0x0F) == 10)  // TDC1 fall
        //OUDS or QD scan engine TDC line setting  6 - end of pixel clock
        {
            rise_fall[chip_id] = false;
            fall_t[chip_id] = ((*packet >> 9) & 0x7FFFFFFFF) + tdc_offset;

            if ((prev_tdc > fall_t[chip_id] + tdc_overflow_drop) && (this->current_line > 1) && (last_offset_line_tdc != this->current_line)) // tdc drop bigger than half of tdc range --> tdc must have overflowed
            {
            tdc_offset += 34359738368; 
            last_offset_line_tdc = this->current_line;
            }

            prev_tdc = fall_t[chip_id];

            ++probe_count_chip[chip_id];

            if ((probe_count_chip[chip_id] <= probe_count_chip[0]) & (probe_count_chip[chip_id] <= probe_count_chip[1]) & (probe_count_chip[chip_id] <= probe_count_chip[2]) & (probe_count_chip[chip_id] <= probe_count_chip[3]))
            {
                this->probe_count = probe_count_chip[chip_id];
                this->current_line = this->probe_count/this->nx;
            }
            else if (probe_count_chip[chip_id] >= most_advanced_probe_count)
            {
                most_advanced_probe_count = probe_count_chip[chip_id];
                most_advanced_line = most_advanced_probe_count/this->nx;

                if (most_advanced_probe_count%this->nxy == 0)
                {
                    this->id_image = most_advanced_line / this->ny;
                    // this->flush_image(this->id_image);
                }
            }

            // Multi-process file-splitting: this camera never had a stop-early
            // mechanism at all (only ran to the true end of file, or until
            // externally interrupted) -- needed so a worker's decode thread
            // actually terminates once it reaches its own slice boundary.
            // Uses probe_count (the aggregate/slowest-chip TRIGGER count)
            // directly, not current_line (=probe_count/nx) -- unlike raster
            // mode, current_line doesn't correspond to a real scan row here
            // (a sparse pattern's real trigger total, e.g. 40856, is nowhere
            // near nx*ny=262144, so current_line never reaches anywhere close
            // to ny).
            //
            // REAL FIX (replaces an earlier, broken attempt that used
            // current_line/nx*(stop_at_line/nx) to drive vSTEM's own
            // fr_count sweep to completion): current_line only advances in
            // whole nx-sized buckets, so it cannot represent an exact
            // single-trigger stop point -- flooring to the containing bucket
            // let the sweep declare "done" (and force-quit both decode
            // threads via *p_processor_line=-1) BEFORE probe_count actually
            // reached stop_at_line, silently dropping every trigger between
            // the bucket floor and the real boundary (confirmed directly: the
            // lost range's width was always < nx and always landed in
            // exactly the bucket containing stop_at_line; the exact width
            // varied run-to-run, confirming a genuine race between this
            // decode thread and vSTEM's process_data() thread racing to
            // observe current_line first -- not a deterministic bug).
            //
            // The fix is to stop relying on that sweep for split workers at
            // all: signal termination directly, the instant the real,
            // exact trigger-count boundary is crossed, by writing
            // *p_processor_line=-1 ourselves. Both read_file() and
            // schedule_buffer()'s own loops already key off this same
            // pointer, and it's the same pointer vSTEM::process_data()'s
            // outer while-loop checks -- so this is a strictly more precise
            // version of the same signal, not a new one. vSTEM.cpp keeps
            // fr_total at the old whole-file value for CHEETAH_PIXELTRIG
            // (see its own comment) specifically so its sweep can never
            // race this to the finish first.
            //
            // stop_at_line < 0 (unset, the default/whole-file case) must skip
            // this entirely and keep relying on the old (pre-existing,
            // unrelated to this work) behavior of reading past the file's
            // true EOF -- see vSTEM.cpp's reset() comment.
            if (this->stop_at_line >= 0 && this->probe_count >= (uint64_t)this->stop_at_line && !this->repetitions_reached)
            {
                this->repetitions_reached = true;
                *this->p_processor_line = -1;
            }
        }
    };
    
    void reset()
    {
        TIMEPIX<event, buffer_size, n_buffer>::reset();
        for (int i = 0; i < 4; i++)
        {
            // Multi-process file-splitting: seed each chip's true rise/fall
            // state at the checkpoint -- see seed_rise_fall's declaration in
            // Timepix.hpp (same fix, same reasoning, as the raster path).
            // Defaults to false, identical to the old unconditional reset,
            // when unset.
            rise_fall[i] = (this->seed_rise_fall[i] != 0);
            // Multi-process file-splitting: seed each chip's OWN real trigger
            // count -- see seed_probe_count_chip's declaration in Timepix.hpp
            // for why this must be per-chip, not a single uniform value (same
            // reasoning as the raster path's seed_line_count). -1 (not
            // seeded) falls back to line_number_offset, identical to the old
            // unconditional reset -- correct for worker 0 (true file start,
            // every chip really is at 0) and for any chip
            // seed_probe_count_chip doesn't cover.
            probe_count_chip[i] = (this->seed_probe_count_chip[i] >= 0) ? this->seed_probe_count_chip[i] : this->line_number_offset;
            line_count[i] = 0;
        }
        // line_number_offset is the aggregate/slowest-chip TRIGGER count at
        // the checkpoint (probe_count's own value there) -- current_line is
        // always derived from it (probe_count/nx), never seeded directly, to
        // stay consistent with how process_tdc() computes it going forward.
        this->probe_count = this->line_number_offset;
        most_advanced_probe_count = this->line_number_offset;
        most_advanced_line = most_advanced_probe_count / this->nx;
        this->current_line = this->probe_count / this->nx;
        this->id_image = 0;
        // Multi-process file-splitting: seed which chip's stream the very
        // first packets after the seek point belong to -- see
        // seed_chip_id's declaration in Timepix.hpp. -1 (not seeded) keeps
        // the old default (0), correct for worker 0 (a header always
        // appears almost immediately from the true file start).
        if (this->seed_chip_id >= 0) chip_id = this->seed_chip_id;
    };

    void read_patten_file()
    {
        std::ifstream file(pattern_file);
        if (file.is_open())
        {
            double element;
            while (file >> element)
            {
                pattern.push_back(element);
            }
            file.close();
            std::cout << "Pattern file read: " << pattern.size() << " positions" << std::endl;
        }
        else std::cout << "Unable to open pattern file" << std::endl;   
    };

public:
    void run()
    {
        reset();
        switch (this->mode)
        {
            case 0:
            {
                this->file.path = this->file_path;
                this->file.open_file();
                // Multi-process file-splitting: jump straight to this worker's
                // slice instead of reading from the start. seek_to() only
                // moves the underlying ifstream's read position -- file.pos
                // itself must be updated to match, since read_data() uses it
                // for pos bookkeeping (identical to Cheetah.hpp's own seek).
                if (this->file_byte_offset > 0)
                {
                    this->file.seek_to(this->file_byte_offset);
                    this->file.pos = this->file_byte_offset;
                }
                this->read_thread = std::thread(&CHEETAH_pixeltrig<event, buffer_size, n_buffer>::read_file, this);
                break;
            }
            case 1:
            {
                //socket connection handled through seperate funtions in python binding
                this->read_thread = std::thread(&CHEETAH_pixeltrig<event, buffer_size, n_buffer>::read_socket, this);
                break;
            }
        }
        this->proc_thread = std::thread(&CHEETAH_pixeltrig<event, buffer_size, n_buffer>::schedule_buffer, this);
        this->starttime  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    };

    void terminate()
    {
        TIMEPIX<event, buffer_size, n_buffer>::terminate();
    };

    // Fast pre-pass for multi-process file splitting, mirroring Cheetah.hpp's
    // find_line_checkpoints (same bulk-read/which_type-only technique -- see
    // that method's own comment for the full design rationale). Single
    // meaningful difference: partitions are evenly spaced by raw TRIGGER
    // COUNT against pattern.size() (the real total triggers for one
    // repetition), not by scan line -- current_line (=probe_count/nx) is not
    // a usable proxy for "how far through the file" here the way it is for
    // raster mode, since a sparse pattern's real trigger total can be far
    // smaller than nx*ny (e.g. 40856 vs 262144), so current_line never gets
    // anywhere near ny.
    //
    // No dt/rise_t seed needed at all (unlike raster mode) -- pixel-trigger's
    // position lookup (pattern[probe_count_chip[chip_id]]) has no dwell-time
    // calibration dependency to warm up. Returns n_splits-1 interior
    // checkpoints (byte_offset, start_trigger_count, rise_fall[4],
    // probe_count_chip[4], chip_id) -- feed these to seed_rise_fall/
    // seed_probe_count_chip/seed_chip_id/line_number_offset on the resuming
    // worker (see their declarations in Timepix.hpp). chip_id specifically
    // matters because it's set only by header packets, and a header can be
    // arbitrarily far from any given checkpoint (confirmed directly: 40+
    // consecutive TDC/event packets with no header in between right at a
    // real checkpoint) -- without seeding it, a resumed worker silently
    // misattributes every packet before its own first header to whatever
    // chip_id defaults to.
    std::vector<std::tuple<uintmax_t, int, std::vector<int>, std::vector<int>, int>> find_trigger_checkpoints(int n_splits)
    {
        reset();
        FileConnector local_file;
        local_file.path = this->file_path;
        local_file.open_file();

        std::vector<std::tuple<uintmax_t, int, std::vector<int>, std::vector<int>, int>> checkpoints;
        int total_triggers = (int)(pattern.size() * this->repetitions);
        int split_idx = 1;
        int next_target = (int)(((int64_t)total_triggers * split_idx) / n_splits);

        const size_t chunk_events = 1 << 20; // 1M packets (8MB) per bulk read
        std::vector<event> chunk(chunk_events);

        while (split_idx < n_splits && local_file.pos < local_file.file_size)
        {
            uintmax_t chunk_start_pos = local_file.pos;
            uintmax_t remaining_bytes = local_file.file_size - local_file.pos;
            size_t this_events = std::min(chunk_events, (size_t)(remaining_bytes / sizeof(event)));
            if (this_events == 0) break;
            local_file.read_data((char*)chunk.data(), this_events * sizeof(event));

            for (size_t k = 0; k < this_events && split_idx < n_splits; ++k)
            {
                if (which_type(&chunk[k]) == 1) // TDC (rise or fall; process_tdc() already ran)
                {
                    while (split_idx < n_splits && (int)this->probe_count >= next_target)
                    {
                        checkpoints.push_back({
                            chunk_start_pos + (k + 1) * sizeof(event),
                            next_target,
                            std::vector<int>{rise_fall[0] ? 1 : 0, rise_fall[1] ? 1 : 0, rise_fall[2] ? 1 : 0, rise_fall[3] ? 1 : 0},
                            std::vector<int>{probe_count_chip[0], probe_count_chip[1], probe_count_chip[2], probe_count_chip[3]},
                            chip_id
                        });
                        ++split_idx;
                        next_target = (int)(((int64_t)total_triggers * split_idx) / n_splits);
                    }
                }
            }
        }
        local_file.close_file();
        return checkpoints;
    }

    CHEETAH_pixeltrig(
        int &nx,
        int &ny,
        bool *b_cumulative,
        int repetitions,
        int *p_processor_line,
        int *p_preprocessor_line,
        int &mode,
        std::string &file_path,  
        SocketConnector socket,
        std::string &pattern_file
    ) : TIMEPIX<event, buffer_size, n_buffer>(
            nx,
            ny,
            b_cumulative,
            repetitions,
            p_processor_line,
            p_preprocessor_line,
            mode,
            file_path,
            socket
        ), pattern_file(pattern_file)
        {
            this->n_cam = 512;
            read_patten_file();
        }
};
#endif // CHEETAH_H
