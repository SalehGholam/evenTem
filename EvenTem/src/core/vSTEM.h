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

#ifndef VSTEM_H
#define VSTEM_H

#include "LiveProcessor.h"

class vSTEM : public LiveProcessor
{
private: 

public: 
    AnnularDetector detector = AnnularDetector(20, 40);

    std::vector<float> inner_radia = {20};
    std::vector<float> outer_radia = {40};
    std::vector<std::array<float, 2>> offsets = {{0, 0}};

    std::vector<size_t> vSTEM_image;
    std::vector<atomwrapper<int>> atomic_vSTEM_image;
    std::vector<std::vector<size_t>> vSTEM_stack;

    bool allow_torch = false;
    bool allow_cuda = false;

    bool auto_offset = true;

    // Charge-weighted, declustered dose counting (CHEETAH/.tpx3 only, single-annulus
    // path only -- see vSTEM::run()). Off by default: when false, behavior is
    // byte-for-byte identical to before this feature existed. Mirrors Roi's own
    // decluster fields exactly (same defaults, same histogram shapes) so a single
    // ToT-sum calibration (e.g. from Roi.ClustersizeTotHistogram) can be reused here.
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

    void run();
    void reset();
    void from_atomic();

    void line_processor(
        size_t &img_num,
        size_t &first_frame,
        size_t &end_frame,
        ProgressMonitor *p_prog_mon,
        size_t &fr_total_u,
        BoundedThreadPool *pool
    );

    void set_offsets(std::vector<std::array<float, 2>>);
    std::vector<int> compute_detector();
    std::vector<int> get_detector();
    void set_detector_mask(py::array_t<int> mask);
    std::vector<int> detector_mask;
    bool use_mask = false;

    // Multi-process file-splitting (CHEETAH/.tpx3 only -- see Cheetah.hpp's
    // find_line_checkpoints and Timepix.hpp's file_byte_offset/
    // line_number_offset/stop_at_line). Returns n_splits-1 interior
    // (byte_offset, start_line) checkpoints; the caller is responsible for
    // constructing n_splits separate vSTEM objects (one per process), setting
    // FileByteOffset/LineNumberOffset/StopAtLine on each from these checkpoints
    // (the first worker uses the defaults 0/0/-1 up to the first checkpoint, the
    // last worker uses its checkpoint's values with the default -1 stop), and
    // summing the resulting vSTEM_image arrays -- the line ranges are disjoint,
    // so summing is the same as placing each worker's own pixels.
    // dt/rise_t/rise_fall are the pre-pass's own already-calibrated/tracked
    // values at the checkpoint -- feed them back as
    // seed_dt/seed_rise_t/seed_rise_fall on the worker built from that
    // checkpoint (see their declaration in Timepix.hpp) so it skips the
    // fallback-guess warm-up window and, more importantly (seed_rise_fall),
    // doesn't silently drop the still-active chips' remaining hits for the
    // boundary row. seed_line_count fixes a second, independent bug (some
    // chips can be 2+ lines ahead of the slowest one, not just 1) found only
    // at higher split counts (N=8/16) -- see its own declaration in
    // Timepix.hpp.
    std::vector<std::tuple<uintmax_t, int, uint64_t, std::vector<uint64_t>, std::vector<int>, std::vector<int>, int>> find_checkpoints(int n_splits);
    uintmax_t file_byte_offset = 0;
    int line_number_offset = 0;
    int stop_at_line = -1;
    uint64_t seed_dt = 0;
    std::vector<uint64_t> seed_rise_t = {0, 0, 0, 0};
    std::vector<int> seed_rise_fall = {0, 0, 0, 0};
    std::vector<int> seed_line_count = {-1, -1, -1, -1};

    // Multi-process file-splitting, pixel-trigger (smart-scan) mode only --
    // see Cheetah_pixeltrig.hpp's find_trigger_checkpoints for the full
    // design rationale (splits by raw trigger count against pattern.size(),
    // not by scan line -- current_line isn't a usable proxy here since a
    // sparse pattern's real trigger total can be far below nx*ny). No dt/
    // rise_t seed needed at all for this camera, unlike the raster case --
    // pixel-trigger's position lookup has no dwell-time calibration to warm
    // up. line_number_offset/stop_at_line/seed_rise_fall are shared with the
    // raster path above but reinterpreted here as trigger-count values, not
    // line numbers.
    // chip_id (last element) matters because it's set only by header packets
    // in the raw byte stream, and a header can be arbitrarily far from any
    // given checkpoint -- without seeding it, a resumed worker silently
    // misattributes every packet before its own first header to whatever
    // chip_id defaults to (found and fixed only after it caused a
    // reproducible ~1-2% miscredit, and once even a crash, on real data).
    std::vector<std::tuple<uintmax_t, int, std::vector<int>, std::vector<int>, int>> find_checkpoints_pixeltrig(int n_splits);
    std::vector<int> seed_probe_count_chip = {-1, -1, -1, -1};
    int seed_chip_id = -1;

    // Constructor
    vSTEM(int repetitions) : LiveProcessor(repetitions)
    {
    };

    // Destructor
    ~vSTEM(){};
};
#endif // !vSTEM_H