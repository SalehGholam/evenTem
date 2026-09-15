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

#ifndef PACBED_H
#define PACBED_H

#include "LiveProcessor.h"

class Pacbed : public LiveProcessor
{
private: 
   
public: 
   
    std::vector<size_t> Pacbed_image;

    // Charge-weighted, declustered PACBED (CHEETAH/.tpx3 only -- see Pacbed::run()).
    // Off by default: when false, behavior is byte-for-byte identical to before this
    // feature existed. Mirrors Var/vSTEM's own decluster fields exactly (same
    // defaults, same histogram shapes) so a single ToT-sum calibration (e.g. from
    // Roi.ClustersizeTotHistogram) can be reused here.
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

    // Masked-scan-pattern PACBED: restrict accumulation to scan positions selected by
    // an arbitrary flat nx*ny 0/1 mask (e.g. a segmented particle/vacuum shape),
    // instead of the whole scan. Works with or without decluster=True -- when both are
    // set, a cluster's SEED hit position decides whether it counts (see
    // Timepix.hpp's enable_Pacbed_declustered). Off by default; set_scan_mask() turns
    // it on.
    bool use_mask = false;
    std::vector<int> scan_mask;
    void set_scan_mask(py::array_t<int> mask);

    void run();
    void reset();

    void line_processor(
        size_t &img_num,
        size_t &first_frame,
        size_t &end_frame,
        ProgressMonitor *p_prog_mon,
        size_t &fr_total_u,
        BoundedThreadPool *pool
    );

    // Constructor
    Pacbed(int repetitions) : LiveProcessor(repetitions)
    {
    };

    // Destructor
    ~Pacbed(){};
};
#endif // !PACBED_H