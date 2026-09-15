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

#ifndef DECLUSTERER_HPP
#define DECLUSTERER_HPP

#include <cmath>
#include <atomic>
#include <vector>
#include <array>
#include <thread>
#include <chrono>
#include <functional>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>

#include "dtype_Electron.hpp"
#include "BoundedThreadPool.hpp"
#include "ClusterResolver.hpp"

#include "Logger.hpp"


#pragma pack(push, 1)
struct cluster_event
{
    uint16_t kx;
    uint16_t ky;
    uint16_t rx;
    uint16_t ry;
    uint16_t id_image;
    uint64_t toa;
    uint16_t tot;
};
#pragma pack(pop)

class Declusterer
{
private:

    static const int n_buffer = 128;

    uint64_t dtime;
    uint16_t dspace;
    int cluster_range;

    int x_crop;
    int y_crop;
    int scan_bin;
    int det_bin;

    dtype_Electron electron;
    std::ofstream* p_file;

    std::thread decluster_thread;
    std::thread write_thread;

    // Owned via unique_ptr -- previously a raw `new`'d pointer with no matching
    // `delete` anywhere in this class, so every Roi/Electron .run() call with
    // decluster=True leaked one BoundedThreadPool (worker threads included) that was
    // never released until the whole process exited. Constructed once the real
    // thread count is known, in init().
    std::unique_ptr<BoundedThreadPool> pool;

    // Guards the three shared histograms below. decluster() now runs concurrently on
    // up to n_threads worker threads (one per ring-buffer slot -- slots are fully
    // independent of each other, so this is safe to parallelize), and each task
    // accumulates its own local histogram deltas first, merging them into the shared
    // histograms under this lock in one short critical section per buffer -- not
    // held across the expensive O(cluster_range) windowed grouping loop itself, so
    // contention stays low regardless of thread count.
    std::mutex histogram_mutex;

    // Per-ring-slot "this buffer's declustering result is ready to write" flag.
    // Needed because with parallel declustering, buffers can finish out of order
    // (buffer 5 might complete before buffer 3 if they land on different threads
    // with different workloads) -- schedule_writing() still has to emit buffers in
    // their original order (for .electron output and for reproducible ROI
    // accumulation order), so it waits on the specific slot's flag rather than
    // assuming "N buffers declustered so far" means "the next N in order are done".
    std::atomic<bool> buffer_ready[n_buffer];

    void decluster(int _buffer_id)
    {
        int upper_lim;
        int clustersize = 1;
        int _bs = buffer[_buffer_id]->size();
        std::vector<bool> used(_bs, false);
        std::vector<int> energy_buffer(_bs, 0);
        // ToT-weighted running sums of the cluster's member (kx,ky), used below to
        // resolve each kept cluster's charge-weighted centroid position (see
        // ClusterResolver.hpp) once its total energy_buffer[i] is final.
        std::vector<double> wx_buffer(_bs, 0.0);
        std::vector<double> wy_buffer(_bs, 0.0);
        // Final size (in raw hits) of each kept cluster, recorded alongside the
        // clustersize_histogram increment below -- used for the joint histogram.
        std::vector<int> size_buffer(_bs, 0);
        std::shared_ptr<std::vector<int>> lcl_keep = std::make_shared<std::vector<int>>();
        std::shared_ptr<std::vector<ClusterInfo>> lcl_cluster_info = std::make_shared<std::vector<ClusterInfo>>();

        // Local (per-task, no locking needed) histogram deltas -- merged into the
        // shared histograms under histogram_mutex once, at the end of this function.
        std::vector<int> local_clustersize_hist(max_clustersize, 0);
        std::vector<int> local_energy_hist(p_energy_histogram->size(), 0);
        int joint_rows = 0, joint_cols = 0;
        std::vector<int> local_joint_hist_flat; // flattened [row*joint_cols + col], only used if p_joint_histogram is set
        if (p_joint_histogram)
        {
            joint_rows = (int)p_joint_histogram->size();
            joint_cols = (int)(*p_joint_histogram)[0].size();
            local_joint_hist_flat.assign((size_t)joint_rows * joint_cols, 0);
        }

        for (int i = 0; i < _bs; ++i)
        {
            if (!used[i]){
                upper_lim = std::min(i + cluster_range, _bs);

                for (int j = i+1; j < upper_lim; ++j)
                {
                    if (!used[j]){
                        bool x_condition = ((*buffer[_buffer_id])[i].kx > (*buffer[_buffer_id])[j].kx) ?
                                        ((*buffer[_buffer_id])[i].kx - (*buffer[_buffer_id])[j].kx <= dspace) :
                                        ((*buffer[_buffer_id])[j].kx - (*buffer[_buffer_id])[i].kx <= dspace);

                        bool y_condition = ((*buffer[_buffer_id])[i].ky > (*buffer[_buffer_id])[j].ky) ?
                                        ((*buffer[_buffer_id])[i].ky - (*buffer[_buffer_id])[j].ky <= dspace) :
                                        ((*buffer[_buffer_id])[j].ky - (*buffer[_buffer_id])[i].ky <= dspace);

                        bool t_condition = ((*buffer[_buffer_id])[i].toa > (*buffer[_buffer_id])[j].toa) ?
                                        ((*buffer[_buffer_id])[i].toa - (*buffer[_buffer_id])[j].toa <= dtime) :
                                        ((*buffer[_buffer_id])[j].toa - (*buffer[_buffer_id])[i].toa <= dtime);

                        if (x_condition && y_condition && t_condition)
                        {
                            used[j] = true;
                            clustersize++;
                            energy_buffer[i] += (*buffer[_buffer_id])[j].tot;
                            wx_buffer[i] += (double)(*buffer[_buffer_id])[j].kx * (double)(*buffer[_buffer_id])[j].tot;
                            wy_buffer[i] += (double)(*buffer[_buffer_id])[j].ky * (double)(*buffer[_buffer_id])[j].tot;
                        }
                    }
                }
                if (clustersize < max_clustersize) local_clustersize_hist[clustersize]++;
                size_buffer[i] = clustersize;
                clustersize = 1;
                lcl_keep->push_back(i);
                energy_buffer[i] += (*buffer[_buffer_id])[i].tot;
                wx_buffer[i] += (double)(*buffer[_buffer_id])[i].kx * (double)(*buffer[_buffer_id])[i].tot;
                wy_buffer[i] += (double)(*buffer[_buffer_id])[i].ky * (double)(*buffer[_buffer_id])[i].tot;

                ClusterInfo info;
                info.tot_sum = (uint64_t)energy_buffer[i];
                info.size = size_buffer[i];
                if (energy_buffer[i] > 0)
                {
                    info.kx = wx_buffer[i] / energy_buffer[i];
                    info.ky = wy_buffer[i] / energy_buffer[i];
                }
                else
                {
                    // zero-ToT hit (shouldn't normally happen): fall back to the seed's own position
                    info.kx = (*buffer[_buffer_id])[i].kx;
                    info.ky = (*buffer[_buffer_id])[i].ky;
                }
                lcl_cluster_info->push_back(info);
            }
            used[i] = true;
        }

        for (int i = 0; i < _bs; ++i)
        {
            if (energy_buffer[i] < (int)local_energy_hist.size())
            {
                local_energy_hist[energy_buffer[i]]++;
            }
        }

        // Joint (cluster size, cluster total ToT) histogram -- the 2D "hits vs. summed
        // ToT" plot used to visually identify single- vs. multi-electron pile-up bands
        // (see e.g. Kuttruff et al. Fig. 4c). Only accumulated if set_joint_histogram
        // was called; left null this whole block is skipped, same as before.
        if (p_joint_histogram)
        {
            for (int i : *lcl_keep)
            {
                int size_for_i = size_buffer[i];
                uint64_t tot_for_i = (uint64_t)energy_buffer[i];
                if (size_for_i >= 0 && size_for_i < joint_rows && tot_for_i < (uint64_t)joint_cols)
                {
                    local_joint_hist_flat[(size_t)size_for_i * joint_cols + tot_for_i]++;
                }
            }
        }

        // Merge this task's local histogram deltas into the shared histograms -- the
        // only section of decluster() that touches state shared across concurrently
        // running buffer tasks, so it's the only section that needs the lock.
        {
            std::lock_guard<std::mutex> lock(histogram_mutex);
            for (size_t k = 0; k < local_clustersize_hist.size(); ++k)
                (*p_clustersize_histogram)[k] += local_clustersize_hist[k];
            for (size_t k = 0; k < local_energy_hist.size(); ++k)
                (*p_energy_histogram)[k] += local_energy_hist[k];
            if (p_joint_histogram)
                for (int a = 0; a < joint_rows; ++a)
                    for (int b = 0; b < joint_cols; ++b)
                        (*p_joint_histogram)[a][b] += local_joint_hist_flat[(size_t)a * joint_cols + b];
        }

        n_electrons_kept += lcl_keep->size();
        keep[_buffer_id] = lcl_keep;
        cluster_info[_buffer_id] = lcl_cluster_info;
        ++this->n_buffer_declustered;
        buffer_ready[_buffer_id].store(true, std::memory_order_release);
    };


    void write_to_file(int _buffer_id)
    {
        auto& lcl_keep = *keep[_buffer_id];
        auto& lcl_cluster_info = *cluster_info[_buffer_id];
        for (size_t local = 0; local < lcl_keep.size(); ++local)
        {
            int i = lcl_keep[local];

            // Optional hook for consumers other than the .electron writer below (e.g.
            // Roi's charge-weighted declustered extraction) -- gets every kept cluster's
            // seed hit (for its raw scan position) plus its resolved centroid/charge,
            // regardless of the x_crop/y_crop gate that only applies to the file write.
            // write_to_file() always runs on this single thread (schedule_writing),
            // never concurrently, so the callback itself never needs its own locking.
            if (p_cluster_callback) p_cluster_callback((*buffer[_buffer_id])[i], lcl_cluster_info[local]);

            if (disable_file_write) continue;

            electron.kx = (*buffer[_buffer_id])[i].kx/det_bin;
            electron.ky = (*buffer[_buffer_id])[i].ky/det_bin;
            electron.rx = (*buffer[_buffer_id])[i].rx/scan_bin;
            electron.ry = (*buffer[_buffer_id])[i].ry/scan_bin;
            electron.id_image = (*buffer[_buffer_id])[i].id_image;

            if (electron.rx < (x_crop/scan_bin) && electron.ry < (y_crop/scan_bin))
            {
                (*p_file).write((const char *)&electron, sizeof(electron));
            }
        }
    };

    void schedule_declustering()
    {
        while ((still_reading) || (n_buffer_submitted < n_buffer_filled))
        {
            if (n_buffer_submitted < n_buffer_filled)
            {
                int buf_id = n_buffer_submitted % n_buffer;
                buffer_ready[buf_id].store(false, std::memory_order_relaxed);
                #ifdef LOG
                    Logger::getInstance().log("Submitting buffer " + std::to_string(n_buffer_submitted) + " of " + std::to_string(n_buffer_filled));
                #endif
                pool->push_task([this, buf_id]{ this->decluster(buf_id); });
                ++n_buffer_submitted;
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
        // Wait for every submitted task to truly finish. BoundedThreadPool's own
        // wait_for_completion() only waits for its internal queue to drain, NOT for
        // tasks already dequeued and running on a worker to actually finish -- with
        // more than one worker thread that's not good enough here, so this waits on
        // n_buffer_declustered instead, which decluster() only increments at its true
        // end (after the histogram merge and buffer_ready flag are set).
        while (n_buffer_declustered < n_buffer_submitted)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        still_processing = false;
    };

    void schedule_writing()
    {
        int _buffer_id;
        while (still_processing || n_buffer_written < n_buffer_submitted)
        {
            _buffer_id = n_buffer_written % n_buffer;
            if (n_buffer_written < n_buffer_submitted && buffer_ready[_buffer_id].load(std::memory_order_acquire))
            {
                write_to_file(_buffer_id);
                buffer[_buffer_id]->clear();
                #ifdef LOG
                    Logger::getInstance().log("clearing buffer " + std::to_string(n_buffer_written) + " of " + std::to_string(n_buffer_filled));
                #endif
                ++n_buffer_written;
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
        still_writing = false;
    };


public:
    // Owned via unique_ptr -- previously raw `new`'d pointers with no matching
    // `delete` (same leak as `pool` above, just 128x over). RAII now handles cleanup
    // automatically when a Declusterer (owned by a fresh CHEETAH/TIMEPIX object
    // constructed on every Roi/Electron .run() call) is destroyed.
    std::unique_ptr<std::vector<cluster_event>> buffer[n_buffer];
    std::shared_ptr<std::vector<int>> keep[n_buffer];
    std::shared_ptr<std::vector<ClusterInfo>> cluster_info[n_buffer];

    bool still_reading = true;
    bool still_processing = true;
    bool still_writing = true;
    int n_buffer_filled = 0;
    int buffer_id_filling = 0;
    // Number of buffers handed to the thread pool so far (renamed/split from what used
    // to be the single "n_buffer_declustered" counter -- now that declustering can run
    // on multiple threads at once, "submitted" and "actually finished" are no longer
    // the same thing and need separate counters).
    int n_buffer_submitted = 0;
    std::atomic<int> n_buffer_declustered = 0;
    int n_buffer_written = 0;
    std::atomic<int> n_electrons_kept = 0;
    std::vector<int> *p_clustersize_histogram;
    std::vector<int> *p_energy_histogram;
    int max_clustersize = 0;

    // Optional hook consumed by callers other than the default .electron file writer
    // (e.g. Roi's charge-weighted declustered extraction, see Timepix.hpp
    // enable_roi_declustered). Left null by default so existing Electron-mode behavior
    // is completely unchanged unless a caller explicitly opts in via set_cluster_callback.
    std::function<void(const cluster_event&, const ClusterInfo&)> p_cluster_callback = nullptr;
    // When true, write_to_file() skips writing .electron records entirely -- used when
    // Declusterer is driving a callback-only consumer (like Roi) that has no dat file.
    bool disable_file_write = false;
    // Optional joint (cluster size x cluster total ToT) histogram, e.g. for
    // reproducing a "hits vs. summed ToT" 2D calibration plot. Null (default) is a
    // no-op -- existing callers that never set this see no behavior change.
    std::vector<std::vector<int>>* p_joint_histogram = nullptr;

    void set_cluster_callback(std::function<void(const cluster_event&, const ClusterInfo&)> callback)
    {
        p_cluster_callback = callback;
    }

    void set_disable_file_write(bool value)
    {
        disable_file_write = value;
    }

    void set_joint_histogram(std::vector<std::vector<int>>* p)
    {
        p_joint_histogram = p;
    }

    void init(uint64_t dtime, uint16_t dspace, int cluster_range,int x_crop,int y_crop,int scan_bin, int det_bin, std::ofstream& _p_file, int n_threads, std::vector<int> *_p_clustersize_histogram, std::vector<int> *_p_energy_histogram)
    {
        this->dtime = dtime;
        this->dspace = dspace;
        this->cluster_range = cluster_range;
        this->p_file = &_p_file;
        this->scan_bin = scan_bin;
        this->det_bin = det_bin;
        this->x_crop = x_crop;
        this->y_crop = y_crop;
        this->p_clustersize_histogram = _p_clustersize_histogram;
        this->p_energy_histogram = _p_energy_histogram;
        max_clustersize = p_clustersize_histogram->size();

        std::cout << "Declustering param: dtime = " << dtime << ", dspace = " << dspace << ", cluster_range = " << cluster_range << ", n_threads = " << n_threads << std::endl;

        // Ring-buffer slots (n_buffer of them) are fully independent of each other for
        // clustering purposes -- each slot's algorithm only ever looks within its own
        // buffer -- so declustering can safely run on multiple slots concurrently.
        // Queue limit = n_buffer so the producer (schedule_declustering) never has to
        // block on push_task under normal operation; set_buffer_read() already
        // provides backpressure against the ring filling up faster than it drains.
        pool = std::make_unique<BoundedThreadPool>(n_threads, n_buffer);
    };

    void set_buffer_read()
    {
        ++n_buffer_filled;
        buffer_id_filling = n_buffer_filled % n_buffer;

        while (n_buffer_filled - n_buffer_written >= n_buffer)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    };

    void run()
    {
        decluster_thread = std::thread(&Declusterer::schedule_declustering, this);
        write_thread = std::thread(&Declusterer::schedule_writing, this);
    };

    void terminate()
    {
        while (!write_thread.joinable() || !decluster_thread.joinable()) {std::this_thread::sleep_for(std::chrono::milliseconds(1));}
        write_thread.join();
        decluster_thread.join();
    };

    Declusterer(){
        for (int i = 0; i < n_buffer; ++i) {
            buffer[i] = std::make_unique<std::vector<cluster_event>>();
            keep[i] = std::make_shared<std::vector<int>>();
            cluster_info[i] = std::make_shared<std::vector<ClusterInfo>>();
            buffer_ready[i].store(false, std::memory_order_relaxed);
        }
    }

    // No custom destructor needed: unique_ptr members (pool, buffer[i]) now clean
    // themselves up automatically. This is what fixes the per-run native memory leak
    // -- previously pool and every buffer[i] were raw `new`'d pointers with no
    // matching `delete` anywhere in this class, so every Roi/Electron .run() call
    // with decluster=True leaked one BoundedThreadPool (with its worker threads) and
    // 128 vectors' worth of native memory, never released until the whole process
    // exited. That leak was the leading suspect for a reproducible segfault when
    // calling decluster=True twice in the same process.
};

#endif // DECLUSTERER_HPP
