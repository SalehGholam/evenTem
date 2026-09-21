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

#ifndef CHEETAH_H
#define CHEETAH_H

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
#include <set>

#include "FileConnector.h"
#include "Timepix.hpp"
#include "../utils/Tpx3ScanSidecar.hpp"

namespace CHEETAH_ADDITIONAL
{
    const size_t BUFFER_SIZE = 65536;
    const size_t N_BUFFER = 512;
    using EVENT = uint64_t;
}; 

#pragma pack(push, 1)
struct state_after_buffer
{
    uint64_t tdc_offset;
    uint64_t toa_offset;
    int line_count[4];
};
#pragma pack(pop)

template <typename event, int buffer_size, int n_buffer>
class CHEETAH : public TIMEPIX<event, buffer_size, n_buffer>
{
private:

    std::vector<state_after_buffer> state_after_buffer_list;

    // header
    // Was genuinely uninitialized on a fresh object -- harmless for a normal
    // whole-file run (the very first packet is always a header, setting a real
    // chip_id before it's ever used to index rise_fall[]/line_count[]/etc.), but
    // multi-process file-splitting seeks into the MIDDLE of the file, where the
    // first packet(s) can belong to the same chip's ongoing block with no new
    // header immediately following -- an out-of-range garbage chip_id there
    // would index those fixed-size-4 arrays out of bounds. Defaulting to 0 keeps
    // every access in bounds even before this worker's own first real header
    // packet corrects it; found alongside the prev_toa fix above, from the same
    // crash.
    int chip_id = 0;
    uint64_t tpx_header = 861425748; //(b'TPX3', 'little')

    // TDC
    uint64_t rise_t[4];
    uint64_t fall_t[4];
    bool rise_fall[4] = {false, false, false, false};
    int line_count[4] = {0, 0, 0, 0};
    int most_advanced_line = 0;
    uint64_t line_interval;
    uint64_t dt;
    // The caller's own configured dwell time, in NANOSECONDS, captured once
    // at construction (before dt above converts it to ticks and, later, a
    // real scan recalibrates it from hardware TDC pulses) -- kept only to
    // match against a .tpx3scan sidecar's own recorded dwell_time_ns for
    // staleness checking (see find_line_checkpoints/Tpx3ScanSidecar.hpp),
    // nothing else reads it.
    uint64_t nominal_dwell_time_ns = 0;

    int type;

    // event
    uint64_t toa = 0;
    uint64_t pack_44;
    int address_multiplier[4] = {1,-1,-1,1};
    int address_bias_x[4] = {256, 511, 255, 0};
    int address_bias_y[4] = {0, 511, 511, 0};

    // overflow correction
    // prev_toa was missing its "=0" here (unlike every sibling field below) --
    // genuinely uninitialized on a fresh object. Harmless for a normal whole-file
    // run: check_toa_overflow()'s own current_line>1 guard stays false long
    // enough for a real toa value to flow in first, before the comparison
    // against prev_toa could ever matter. But multi-process file-splitting
    // (line_number_offset > 0) seeds current_line already past that guard, so
    // the very first overflow check could compare against garbage -- found via
    // this feature's own testing (a worker slice crashed the kernel outright,
    // consistent with the resulting corrupted toa_offset producing an
    // out-of-range probe_position and an out-of-bounds array write).
    uint64_t prev_toa = 0;
    std::atomic<uint64_t> toa_offset = 0;
    uint64_t toa_overflow_drop = 4294967296;
    int last_offset_line = 0;

    uint64_t prev_tdc = 0;
    uint64_t tdc_offset = 0;
    uint64_t tdc_overflow_drop = 17179869184; //half of 34359738368 = tdc range
    int last_offset_line_tdc = 0;
    std::thread check_overflow_thread;


    void parse_event(event *packet)
    {
        // - ((*packet >> 16) & 0xF): the FToA fine-timing correction term, matching
        // parse_event_w_tot() below -- previously omitted here, so the *same*
        // physical hit reconstructs to a ToA up to 15 ticks (~23ns) different
        // depending on whether declustering (which uses parse_event_w_tot) is
        // active for the run. Near a scan-line dwell-time boundary that's enough
        // to shift which column/line a hit is attributed to, or to accept/reject
        // it at a flyback boundary -- the root cause of the small, consistent
        // ~0.15-0.2% plain-vs-declustered residual documented in pyeventem's
        // CPP_EVENTEM_BUGS.md #4.
        toa = ((((*packet & 0xFFFF) << 14) + ((*packet >> 30) & 0x3FFF)) << 4) - ((*packet >> 16) & 0xF) + toa_offset;
        uint64_t _probe_position = ( toa - (rise_t[chip_id] * 2)) / dt;
        // Guard against a faster chip's line_count racing past the intended
        // ny*repetitions total before the GLOBAL repetitions_reached flag (which
        // only flips once the SLOWEST of the 4 chips catches up) has had a chance
        // to stop dispatch for this chip specifically. Without this, a trailing/
        // phantom line from a chip that finished slightly early wraps via
        // `% this->ny` below and contaminates "virtual row 0" with events that
        // don't belong there -- found by tracing Dose_image[0] jumping sharply
        // while FourD processed the scan's LAST chunk-group, on a real, correctly-
        // sized (matching the file's true scan geometry) run, not a misconfigured
        // test. Each chip's own line_count is checked individually here, not just
        // the aggregate (slowest-chip) repetitions_reached, since chips are not
        // perfectly synchronized at the scan's end.
        {
            // Multi-process file-splitting: an intermediate slice boundary
            // (stop_at_line >= 0, i.e. NOT the last worker) must NOT drop
            // events here. Chips are not interleaved event-by-event in the
            // file -- a faster chip can legitimately reach the boundary line
            // well before the slowest chip does (which is what actually
            // drives the checkpoint byte position via process_tdc()'s
            // aggregate current_line). Those early events occur strictly
            // within THIS worker's own byte range; dropping them here loses
            // them permanently, since the next worker's byte range starts
            // after the checkpoint and structurally cannot see bytes already
            // consumed here. No double-count risk in keeping them: once a
            // chip's own line_count passes the boundary, all its later events
            // (processed by the next worker) are for lines strictly beyond it.
            // Confirmed via mp_vstem.ipynb: dropping them here caused a ~90%
            // undercount concentrated on exactly the first row of every
            // interior/last slice, with every other row byte-for-byte exact.
            // The true end-of-scan guard (stop_at_line < 0, i.e. this IS the
            // last worker or a normal whole-file run) still applies -- a chip
            // racing past the real scan end has nothing valid to wrap onto.
            if (this->stop_at_line < 0 && line_count[chip_id] >= (uint64_t)(this->ny * this->repetitions)) return;
        }
        if (_probe_position < this->nx)
        {
            pack_44 = (*packet >> 44);
            _probe_position += (line_count[chip_id] % this->ny) * this->nx;
            uint16_t _kx = (address_multiplier[chip_id] * (((pack_44 & 0x0FE00) >> 8) + ((pack_44 & 0x00007) >> 2)) + address_bias_x[chip_id]);
            uint16_t _ky = (address_multiplier[chip_id] * (((pack_44 & 0x001F8) >> 1) + (pack_44 & 0x00003)) + address_bias_y[chip_id]);

            switch(this->functionType)
            {
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::vstem:
                    this->vstem(_probe_position,_kx,_ky, this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::multi_vstem:
                    this->multi_vstem(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::mask_vstem:
                    this->mask_vstem(_probe_position,_kx,_ky,this->id_image);
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
                    this->roi_mask(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::roi_4D:
                    this->roi_4D(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::tcBF:
                    // Previously missing entirely -- tcBF was fully wired up (enum
                    // value, processor class, enable_tcBF()) but never actually
                    // dispatched here, so a tcBF run on Cheetah data silently
                    // produced an all-zero result. See CPP_EVENTEM_BUGS.md #7.
                    this->tcBF(_probe_position,_kx,_ky,this->id_image);
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

    void parse_event_w_tot(event *packet)
    {
        toa = ((((*packet & 0xFFFF) << 14) + ((*packet >> 30) & 0x3FFF)) << 4) - ((*packet >> 16) & 0xF) + toa_offset;
        this->tot = ((*packet) >> (16 + 4)) & 0x3ff;
        uint64_t _probe_position = ( toa - (rise_t[chip_id] * 2)) / dt;
        // See parse_event's identical guard above for why: a faster chip's
        // line_count can race past ny*repetitions before the aggregate
        // repetitions_reached flag (gated on the slowest chip) stops it, wrapping
        // a trailing/phantom line back onto "virtual row 0" via `% this->ny` below.
        {
            // Multi-process file-splitting: see parse_event's identical guard
            // above for why an intermediate slice boundary must NOT drop here.
            if (this->stop_at_line < 0 && line_count[chip_id] >= (uint64_t)(this->ny * this->repetitions)) return;
        }
        if (_probe_position < this->nx)
        {
            pack_44 = (*packet >> 44);
            _probe_position += (line_count[chip_id] % this->ny) * this->nx;
            uint16_t _kx = (address_multiplier[chip_id] * (((pack_44 & 0x0FE00) >> 8) + ((pack_44 & 0x00007) >> 2)) + address_bias_x[chip_id]);
            uint16_t _ky = (address_multiplier[chip_id] * (((pack_44 & 0x001F8) >> 1) + (pack_44 & 0x00003)) + address_bias_y[chip_id]);

            switch(this->functionType)
            {
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::vstem:
                    this->vstem(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::multi_vstem:
                    this->multi_vstem(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::mask_vstem:
                    this->mask_vstem(_probe_position,_kx,_ky,this->id_image);
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
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::roi_mask:
                    this->roi_mask(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::roi_4D:
                    // roi_4D_ToT, not roi_4D: this dispatch (parse_event_w_tot) only
                    // ever runs when b_tot=true, which functionType=roi_4D only ever
                    // reaches via Roi.tot_mode=True (declustered Roi uses a separate
                    // cluster callback, not this per-event dispatch, for its 4D cube).
                    this->roi_4D_ToT(_probe_position,_kx,_ky,this->id_image);
                    break;
                case TIMEPIX<event, buffer_size, n_buffer>::FunctionType::tcBF:
                    // See the identical case in parse_event() above -- CPP_EVENTEM_BUGS.md #7.
                    this->tcBF(_probe_position,_kx,_ky,this->id_image);
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
        if ((prev_toa > toa + toa_overflow_drop) && (this->current_line > 1) && (this->current_line > last_offset_line + 1)) // toa drop bigger than half of toa range --> toa must have overflowed
        {
            toa_offset += 17179869184; 
            last_offset_line = this->current_line;
            // std::cout << "toa overflow (image " << this->current_line/this->ny << ", line " << this->current_line%this->ny << ")" << std::endl;
        }
        prev_toa = toa;
    };

    void continous_check_toa_overflow(){
        while ((*this->p_processor_line)!=-1)
        {
            check_toa_overflow();
            std::this_thread::sleep_for(std::chrono::nanoseconds(100));
        }
    }

    inline void schedule_buffer()
    {
        int buffer_id;

        while ((*this->p_processor_line)!=-1)
        {
            if (this->n_buffer_processed < this->n_buffer_filled)
            {
                buffer_id = this->n_buffer_processed % this->n_buf;

                if (!this->repetitions_reached)
                { 
                    process_buffer(&(this->buffer[buffer_id]));
                    // state_after_buffer_list.push_back({tdc_offset, toa_offset, line_count[0], line_count[1], line_count[2], line_count[3]});

                    if (this->decluster) this->declusterer.set_buffer_read();

                    ++this->n_buffer_processed;
                }
                *this->p_preprocessor_line = (int)this->current_line;
                check_toa_overflow();
            }
            else
            {
                this->process_wait++;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    };

    inline void process_buffer(std::array<event, buffer_size> *p_buffer)
    {
        switch (this->b_tot)
        {
           case true:
           for (int j = 0; j < buffer_size; j++)
            {
                type = which_type(&(*p_buffer)[j]);
                if ((type == 2) && rise_fall[chip_id] && (!this->repetitions_reached)) 
                {
                    parse_event_w_tot(&(*p_buffer)[j]);
                }
            }
            break;
            case false:
            for (int j = 0; j < buffer_size; j++)
            {
                type = which_type(&(*p_buffer)[j]);
                if ((type == 2) && rise_fall[chip_id] && (!this->repetitions_reached)) 
                {
                    parse_event(&(*p_buffer)[j]);
                }
            }
        }
        if (this->repetitions_reached)  
        {
            this->probe_position_total = this->nxy*this->repetitions+1;
            this->id_image = this->repetitions;
        }
    };

    inline int which_type(event *packet)
    {
        if (*packet >> 60 == 0xb) // event
        {
            return 2;
        }
        else if ((*packet & 0xFFFFFFFF) == tpx_header) // header
        {
            chip_id = (*packet >> 32) & 0xff;
            check_toa_overflow();
            return 0;
        } 
        else if (*packet >> 60 == 0x6) // TDC
        {
            check_toa_overflow();
            process_tdc(packet);
            return 1;
        } 
        else if (*packet >> 60 == 0x4)
        {
            std::cout << "global time" << std::endl;
            return 3;
        }
        else // unknown
        {
            std::cout << "unknown packet type" << std::endl;
            return 3;
        }
    };

    inline void process_tdc(event *packet)
    {
        if (((*packet >> 56) & 0x0F) == 15) // TDC1 rise
        {
            rise_fall[chip_id] = true;
            rise_t[chip_id] = ((*packet >> 9) & 0x7FFFFFFFF) + tdc_offset;

            if ((prev_tdc > rise_t[chip_id] + tdc_overflow_drop) && (this->current_line > 1) && (this->current_line > last_offset_line_tdc + 1)) // tdc drop bigger than half of tdc range --> tdc must have overflowed
            {
            tdc_offset += 34359738368; 
            last_offset_line_tdc = this->current_line;
            // std::cout << "tdc overflow (image " << this->current_line/this->ny << ", line " << this->current_line%this->ny << ")" << std::endl;
            }

            prev_tdc = rise_t[chip_id];
        }
        else if (((*packet >> 56) & 0x0F) == 10) // TDC1 fall
        {
            // A "fall" with no matching preceding "rise" for this chip (rise_fall[chip_id]
            // already false right here, before this line resets it) is a phantom/orphaned
            // edge -- e.g. the trailing edge of a pulse that began before this acquisition
            // started recording, or before this worker's own slice (multi-process
            // splitting). rise_t[chip_id] then still holds its reset()-time value (0,
            // unless seeded), not a real per-line reference point -- computing
            // line_interval/dt from it below would use fall_t[chip_id]*2 (a large ABSOLUTE
            // timestamp, not a small relative one) as dt itself, making dt wildly too
            // large. Every hit's _probe_position = toa/dt then comes out near 0 for a long
            // stretch of real elapsed time, piling many hits onto the first few pixels
            // while _probe_position never reaches the scan line's true later pixels at all
            // -- exactly a "several pixels with far too many counts, rest of the line zero"
            // failure mode on line 1. Skip line-crediting and the dt recompute entirely
            // when this happens and wait for this chip's first REAL rise/fall pair instead
            // (same "sacrifice a line's data rather than guess wrong" philosophy already
            // used for dt==0 in the CHEETAH constructor above) -- rather than counting this
            // as a completed line: process_buffer() never actually dispatched any hits
            // under it either (that requires rise_fall[chip_id]==true, which never held
            // during this window), so crediting it as one would also permanently offset
            // this chip's line_count by one relative to the other three.
            //
            // line_number_offset > 0 (a resumed multi-process split worker) is
            // deliberately EXCLUDED from this guard: a worker seeking into the middle
            // of the file starts with no real header yet for its actual chip_id (see
            // the genuinely-uninitialized-chip_id comment above), so a real, valid fall
            // belonging to a DIFFERENT chip can get misattributed to chip_id 0's
            // rise_fall/rise_t bookkeeping for the brief window before the next real
            // header corrects it -- rise_fall[chip_id] being false there does not mean
            // "orphaned fall," just "misattributed." Skipping the recompute in that case
            // (measured directly, comparing against the pre-existing small
            // boundary-localized residual already documented in IMPROVEMENTS.md) leaves
            // a stale dt in place for far longer than the old unconditional recompute
            // did, spreading a small residual into a much larger one. So: apply the new
            // guard only for a genuine whole-file/first-worker run, where an orphaned
            // fall really can only mean "no rise has ever happened yet for this chip in
            // this acquisition" -- the exact scenario the guard was written for.
            bool had_valid_rise = rise_fall[chip_id] || (this->line_number_offset > 0);
            rise_fall[chip_id] = false;
            if (had_valid_rise)
            {
                fall_t[chip_id] = ((*packet >> 9) & 0x7FFFFFFFF) + tdc_offset;

                if ((prev_tdc > fall_t[chip_id] + tdc_overflow_drop) && (this->current_line > 1) && (this->current_line > last_offset_line_tdc + 1)) // tdc drop bigger than half of tdc range --> tdc must have overflowed
                {
                tdc_offset += 34359738368;
                last_offset_line_tdc = this->current_line;
                // std::cout << "tdc overflow (image " << this->current_line/this->ny << ", line " << this->current_line%this->ny << ")" << std::endl;
                }

                prev_tdc = fall_t[chip_id];

                ++line_count[chip_id];

                if ((line_count[chip_id] <= line_count[0]) & (line_count[chip_id] <= line_count[1]) & (line_count[chip_id] <= line_count[2]) & (line_count[chip_id] <= line_count[3]))
                {
                    this->current_line = line_count[chip_id];
                }
                else if (line_count[chip_id] >= most_advanced_line)
                {
                    most_advanced_line = line_count[chip_id];
                    if (most_advanced_line%this->ny == 0)
                    {
                        this->id_image = most_advanced_line / this->ny ;
                    }
                }

                line_interval = (fall_t[chip_id] - rise_t[chip_id]) * 2; //factor 2 for difference in time unit of tdc and toa
                dt = line_interval / this->nx; //unit 1.5625 ns
                // std::cout << dt*1.5625 << std::endl;
            }
        }
        // Multi-process file-splitting: stop_at_line lets a worker's slice end
        // before the true scan end (this->ny*repetitions), which the LAST worker
        // still uses via the -1 default. See its declaration in Timepix.hpp.
        uint64_t _effective_stop = (this->stop_at_line >= 0) ? (uint64_t)this->stop_at_line : (uint64_t)(this->ny * this->repetitions);
        if (this->current_line >= _effective_stop) this->repetitions_reached = true;
    };

    void skip_to_buffer(int buffer_id)
    {
        this->n_buffer_processed = buffer_id;
        int min_line = std::min({state_after_buffer_list[buffer_id].line_count[0], state_after_buffer_list[buffer_id].line_count[1], state_after_buffer_list[buffer_id].line_count[2], state_after_buffer_list[buffer_id].line_count[3]});
        this->current_line = min_line;
        toa_offset = state_after_buffer_list[buffer_id].toa_offset;
        tdc_offset = state_after_buffer_list[buffer_id].tdc_offset;
        line_count[0] = state_after_buffer_list[buffer_id].line_count[0];
        line_count[1] = state_after_buffer_list[buffer_id].line_count[1];
        line_count[2] = state_after_buffer_list[buffer_id].line_count[2];
        line_count[3] = state_after_buffer_list[buffer_id].line_count[3];
    }
    
    void reset()
    {
        TIMEPIX<event, buffer_size, n_buffer>::reset();
        for (int i = 0; i < 4; i++)
        {
            // Multi-process file-splitting: seed each chip's true rise/fall
            // state at the checkpoint -- see seed_rise_fall's declaration in
            // Timepix.hpp for why this is the actual fix for the boundary-row
            // undercount. Defaults to false (0), identical to the old
            // unconditional reset, when unset.
            rise_fall[i] = (this->seed_rise_fall[i] != 0);
            // Multi-process file-splitting: seed each chip's OWN real line
            // counter at the checkpoint -- see seed_line_count's declaration
            // in Timepix.hpp for why this must be per-chip, not a single
            // uniform value. -1 (not seeded) falls back to line_number_offset,
            // identical to the old unconditional reset -- correct for worker 0
            // (true file start, every chip really is at line_number_offset=0)
            // and for any chip seed_line_count doesn't cover.
            line_count[i] = (this->seed_line_count[i] >= 0) ? this->seed_line_count[i] : this->line_number_offset;
            // Multi-process file-splitting: seed this chip's last-known TDC
            // rise time too -- see seed_dt/seed_rise_t's declaration in
            // Timepix.hpp for why. Defaults to 0 (matching rise_t's old,
            // uninitialized-in-practice-often-zero state) when unset.
            rise_t[i] = this->seed_rise_t[i];
        }
        // Multi-process file-splitting: see seed_dt's declaration in
        // Timepix.hpp. A seed of 0 means "not provided" -- keep the
        // constructor's own fallback-then-calibrate dt untouched.
        if (this->seed_dt > 0) this->dt = this->seed_dt;
        // Multi-process file-splitting: seed which chip's block this worker's
        // byte_offset resumes into -- see find_line_checkpoints()'s own
        // comment for why this matters (a resuming worker has no header at
        // its own start, so without this every event gets misattributed to
        // chip_id 0 until the worker's own first real header happens to
        // appear). -1 (not seeded) keeps the old default of 0, identical to
        // a normal whole-file run (the very first packet is always a header
        // there, correcting chip_id before it's ever used). Mirrors
        // CHEETAH_pixeltrig::reset()'s identical seed_chip_id use.
        if (this->seed_chip_id >= 0) chip_id = this->seed_chip_id;
        this->current_line = this->line_number_offset;
        // Multi-process file-splitting: most_advanced_line must start at the
        // real maximum across all 4 seeded line_counts, not just
        // line_number_offset (the slowest chip's value) -- otherwise a chip
        // seeded 2+ lines ahead would appear to un-advance id_image tracking
        // for multi-repetition (rep>1) scans. No effect for rep=1.
        most_advanced_line = line_count[0];
        for (int i = 1; i < 4; i++) if (line_count[i] > most_advanced_line) most_advanced_line = line_count[i];
        this->id_image = 0;
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
                // Multi-process file-splitting (see file_byte_offset's declaration
                // in Timepix.hpp): jump straight to this worker's slice instead of
                // reading from the start. seek_to() only moves the underlying
                // ifstream's read position -- file.pos itself must be updated to
                // match, since read_data() uses it for pos bookkeeping and
                // find_line_checkpoints() below reports offsets relative to it.
                if (this->file_byte_offset > 0)
                {
                    this->file.seek_to(this->file_byte_offset);
                    this->file.pos = this->file_byte_offset;
                }
                this->read_thread = std::thread(&CHEETAH<event, buffer_size, n_buffer>::read_file, this);
                break;
            }
            case 1:
            {
                //socket connection handled through seperate funtions in python binding
                this->read_thread = std::thread(&CHEETAH<event, buffer_size, n_buffer>::read_socket, this);
                break;
            }
        }
        this->proc_thread = std::thread(&CHEETAH<event, buffer_size, n_buffer>::schedule_buffer, this);
        this->check_overflow_thread = std::thread(&CHEETAH<event, buffer_size, n_buffer>::continous_check_toa_overflow, this);
        this->starttime  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    };

    void terminate()
    {
        TIMEPIX<event, buffer_size, n_buffer>::terminate();
        while ((!check_overflow_thread.joinable()) ) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        check_overflow_thread.join();
    };

    // Fast pre-pass for multi-process file splitting (see file_byte_offset/
    // line_number_offset/stop_at_line in Timepix.hpp). Reads the file
    // sequentially, ONCE, in bulk chunks -- but for every packet only runs
    // which_type() (a single shift+compare, plus process_tdc() for the sparse
    // TDC packets) and explicitly skips the expensive per-event work
    // (toa/kx/ky decode, array writes) that parse_event()/parse_event_w_tot()
    // would otherwise do for the vast majority of packets (raw hits, not TDC
    // pulses). Single-threaded, synchronous, blocking -- meant to be called
    // once, cheaply, on a throwaway CHEETAH instance before spawning worker
    // processes, not as part of the normal threaded run() pipeline.
    //
    // Returns n_splits-1 interior checkpoints (byte_offset, start_line, dt,
    // rise_t[4], rise_fall[4], line_count[4], chip_id) evenly spaced by scan
    // line, plus nothing for the trivial (0, 0) start or the true end (the
    // first and last workers use file_byte_offset=0/stop_at_line=-1, i.e. the
    // existing defaults, same as a normal whole-file run). byte_offset points
    // to the start of the packet immediately after the TDC-fall that
    // completed the target line -- i.e. exactly where a fresh CHEETAH
    // instance should seek_to() to pick up cleanly from there.
    // dt/rise_t/rise_fall/line_count/chip_id are this same pre-pass's own
    // already-calibrated/tracked values at that exact point -- feed them to
    // the resuming worker as seed_dt/seed_rise_t/seed_rise_fall/
    // seed_line_count/seed_chip_id (see their declaration in Timepix.hpp) so
    // it doesn't have to re-derive them from scratch. seed_rise_fall fixed
    // the N=4 boundary-row undercount; seed_line_count fixes a second bug
    // found only at N=8/16 (some chips are 2+ lines ahead of the slowest one,
    // not just 1 -- see its own comment).
    //
    // chip_id specifically: a resuming worker has no header packet at its own
    // byte_offset (headers only appear at the START of a chip's block, and
    // byte_offset almost always lands MID-block, right after the triggering
    // chip's own TDC-fall) -- without seeding it, the worker defaults
    // chip_id=0 (see its own declaration below) and misattributes every event
    // it decodes to chip 0's rise_t/dt/address_bias/address_multiplier until
    // its own first REAL header happens to appear, however many events later
    // that is. This doesn't crash (chip_id=0 is always in-bounds) but does
    // silently compute wrong positions/timing for those events -- found by
    // tracing a ~19%-of-one-row vSTEM undercount at exactly the split boundary
    // row on a real file, back to this. `which_type()` below already updates
    // `chip_id` as a side effect of scanning every packet (including headers)
    // during this pre-pass, so its value at the exact checkpoint moment is
    // already the correct chip_id for whatever block byte_offset resumes into
    // -- CHEETAH_pixeltrig's own seed_chip_id (find_trigger_checkpoints) is
    // the same fix for the same underlying hazard, already shipped there.
    // Tries a .tpx3scan sidecar first (see ../utils/Tpx3ScanSidecar.hpp) --
    // pyeventem's own persisted "pass 1", real interop rather than a format
    // of our own -- and only falls back to the full from-scratch prescan
    // below when there isn't one, it's stale, or it doesn't parse (the
    // sidecar reader itself never throws; checkpoints_from_index()
    // returning an empty vector for an n_splits it can't fully answer is
    // the other, equally safe "give up, fall back" signal). Genuinely
    // equivalent output either way -- differential-tested directly against
    // this same prescan (allow_sidecar=false forces it) on the real 3.2 GB
    // Full scan test file pyeventem/test_data ships, every field of every
    // checkpoint bit-identical across n_splits in {2, 3, 4, 8, 16, 33} --
    // so a caller never has to know or care which path actually ran.
    //
    // `allow_sidecar=false` forces the prescan unconditionally -- used only
    // by that differential test, to get the trusted reference answer from
    // the very same object/call it's comparing the sidecar path against.
    std::vector<std::tuple<uintmax_t, int, uint64_t, std::vector<uint64_t>, std::vector<int>, std::vector<int>, int>>
    find_line_checkpoints(int n_splits, bool allow_sidecar = true)
    {
        if (allow_sidecar)
        {
            auto index = tpx3scan::read_tpx3scan(this->file_path, this->nx, (double)this->nominal_dwell_time_ns);
            if (index)
            {
                int total_lines = (int)(this->ny * this->repetitions);
                auto derived = tpx3scan::checkpoints_from_index(*index, total_lines, n_splits);
                if ((int)derived.size() == n_splits - 1)
                {
                    std::vector<std::tuple<uintmax_t, int, uint64_t, std::vector<uint64_t>, std::vector<int>, std::vector<int>, int>> out;
                    out.reserve(derived.size());
                    for (const auto &cp : derived)
                    {
                        out.push_back({
                            (uintmax_t)cp.byte_offset,
                            cp.start_line,
                            cp.dt,
                            std::vector<uint64_t>{cp.rise_t[0], cp.rise_t[1], cp.rise_t[2], cp.rise_t[3]},
                            std::vector<int>{cp.rise_fall[0], cp.rise_fall[1], cp.rise_fall[2], cp.rise_fall[3]},
                            std::vector<int>{cp.line_count[0], cp.line_count[1], cp.line_count[2], cp.line_count[3]},
                            cp.chip_id
                        });
                    }
                    return out;
                }
                // Fell short (e.g. a chip never reaches one of the target
                // lines within this sidecar's own recorded segments -- can
                // happen for an n_splits finer than what's practically
                // resolvable near the very end of a short scan) -- fall
                // through to the full prescan below rather than return a
                // partial/wrong-length result.
            }
        }
        return find_line_checkpoints_prescan(n_splits);
    }

    std::vector<std::tuple<uintmax_t, int, uint64_t, std::vector<uint64_t>, std::vector<int>, std::vector<int>, int>> find_line_checkpoints_prescan(int n_splits)
    {
        reset();
        FileConnector local_file;
        local_file.path = this->file_path;
        local_file.open_file();

        std::vector<std::tuple<uintmax_t, int, uint64_t, std::vector<uint64_t>, std::vector<int>, std::vector<int>, int>> checkpoints;
        int total_lines = (int)(this->ny * this->repetitions);
        int split_idx = 1;
        int next_target = (int)(((int64_t)total_lines * split_idx) / n_splits);

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
                    while (split_idx < n_splits && (int)this->current_line >= next_target)
                    {
                        checkpoints.push_back({
                            chunk_start_pos + (k + 1) * sizeof(event),
                            next_target,
                            this->dt,
                            std::vector<uint64_t>{rise_t[0], rise_t[1], rise_t[2], rise_t[3]},
                            std::vector<int>{rise_fall[0] ? 1 : 0, rise_fall[1] ? 1 : 0, rise_fall[2] ? 1 : 0, rise_fall[3] ? 1 : 0},
                            std::vector<int>{line_count[0], line_count[1], line_count[2], line_count[3]},
                            chip_id
                        });
                        ++split_idx;
                        next_target = (int)(((int64_t)total_lines * split_idx) / n_splits);
                    }
                }
            }
        }
        local_file.close_file();
        return checkpoints;
    }

    CHEETAH(
        int &nx,
        int &ny,
        // Dwell time, unit: NANOSECONDS -- matches LiveProcessor::dt and every
        // Python-facing use of it throughout this project (e.g. `roi.dt = 100` for
        // a 100ns dwell; there is no unit conversion anywhere between that and this
        // parameter), matching this project's original (pre-microseconds) API
        // convention. Only affects this constructor's FALLBACK dt (used for events
        // before the real, hardware-calibrated dt is known -- computed from the
        // first line's TDC pulse, which only completes at the END of that line),
        // not the steady-state TDC-derived dt used for every line after the first
        // (that one comes from real hardware timestamps and is unit-independent).
        // An inconsistent value here (e.g. treating the input as microseconds while
        // this conversion expects nanoseconds) makes the fallback dt far too large,
        // so almost every real event during line 1 computes a probe_position that
        // never advances past a handful of pixels -- see IMPROVEMENTS.md/
        // SESSION_OVERVIEW.md for the "several pixels with far too many counts, rest
        // of the line zero" failure mode this produces when the units disagree.
        int &dt,
        bool *b_cumulative,
        int repetitions,
        int *p_processor_line,
        int *p_preprocessor_line,
        int &mode,
        std::string &file_path,
        SocketConnector socket
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
        ), dt((uint64_t)dt*16/25), nominal_dwell_time_ns((uint64_t)dt)
        {
            this->n_cam = 512;
            if (this->dt == 0){
                // Bug (CPP_EVENTEM_BUGS.md #9): `dt = 1000;` here used to write only to
                // the constructor's reference PARAMETER (and, through it, the caller's
                // own int) -- but this->dt (the member actually used as the fallback
                // dwell time everywhere else in this class) was already computed from
                // the parameter's PRE-fallback value in the initializer list above,
                // which runs before this body. So the member stayed 0 forever
                // regardless of this assignment: the fallback never actually took
                // effect. Now sets this->dt directly (in the same tick units as the
                // initializer list's own conversion), and still updates the parameter
                // too so a caller reading `dt` back afterward sees the fallback value
                // applied, matching the pre-fix API surface.
                std::cout << "Dwell time not provided! This means sacrificing the first line" << std::endl;
                dt = 1000;
                this->dt = (uint64_t)dt*16/25;
            }
        };
};
#endif // CHEETAH_H
