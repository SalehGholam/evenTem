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

#ifndef ROI_H
#define ROI_H

#include "LiveProcessor.h"
#include "Roi4D.hpp"

class Roi : public LiveProcessor
{
private: 
   
public: 

    bool b_ROI_4D;
    int det_bin = 1;
    int roi_bitdepth = 8;
    
    std::vector<uint64_t> Roi_diffraction_pattern;
    std::vector<uint64_t> Roi_scan_image;
    std::shared_ptr<Roi4D<uint8_t>> Roi_4D_8;
    std::shared_ptr<Roi4D<uint16_t>> Roi_4D_16;
    std::shared_ptr<Roi4D<uint32_t>> Roi_4D_32;


    std::vector<std::vector<uint64_t>> Roi_diffraction_pattern_stack; 
    std::vector<std::vector<uint64_t>> Roi_scan_image_stack;
    int lower_left[2] = {0, 0};
    int upper_right[2] = {1, 1};
    int L_0;
    int L_1;
    int finish_line;

    bool use_mask = false;
    std::vector<std::vector<int>> roi_mask;
    void set_roi_mask(std::vector<py::array_t<int>> arrays);
    void set_bitdepth(int bitdepth);

    // ToT-sum extraction (CHEETAH/.tpx3 only -- see Roi::run()): each raw hit
    // contributes its own ToT value to the scan image / diffraction pattern / 4D
    // cube instead of +1, so the result is "summed deposited charge per probe"
    // rather than "electron/hit count per probe". No clustering involved (each hit
    // counted individually, at its own raw position) -- deliberately independent
    // of, and mutually exclusive with, decluster. Off by default: when false,
    // behavior is unchanged from before this feature existed.
    bool tot_mode = false;

    // Charge-weighted, declustered extraction (CHEETAH/.tpx3 only -- see Roi::run()).
    // Off by default: when false, behavior is byte-for-byte identical to before this
    // feature existed. dtime/dspace/cluster_range mirror Electron's own declustering
    // parameters/defaults. tot_per_electron has no safe default -- it depends on beam
    // energy and detector threshold, and must be calibrated by the user from their own
    // cluster ToT-sum histogram (see docs); decluster=True with tot_per_electron<=0 throws.
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
    // Diagnostic histograms from the declustering step (same meaning as Electron's),
    // useful for comparing against a calibration plot like the one used to pick
    // tot_per_electron: clustersize_histogram bins cluster sizes, energy_histogram
    // bins each cluster's total summed ToT.
    std::vector<int> clustersize_histogram = std::vector<int>(50,0);
    std::vector<int> energy_histogram = std::vector<int>(4096,0);
    // Joint (cluster size, cluster total ToT) histogram -- reproduces a "hits vs.
    // summed ToT" 2D calibration plot. Row = cluster size (0..49), column = summed ToT
    // (0..4095), same axis ranges/meaning as clustersize_histogram/energy_histogram.
    std::vector<std::vector<int>> clustersize_tot_histogram = std::vector<std::vector<int>>(50, std::vector<int>(4096, 0));


    void run();
    void reset();
    void set_roi(int,int,int,int);
    std::array<int,4> get_roi();

    // Multi-process file-splitting (CHEETAH/.tpx3 only -- see Cheetah.hpp's
    // find_line_checkpoints and Timepix.hpp's file_byte_offset/
    // line_number_offset/stop_at_line). Returns n_splits-1 interior
    // (byte_offset, start_line) checkpoints; the caller is responsible for
    // constructing n_splits separate Roi objects (one per process), setting
    // file_byte_offset/line_number_offset/stop_at_line on each from these
    // checkpoints (the first worker uses the defaults 0/0/-1 up to the
    // first checkpoint, the last worker uses its checkpoint's values with
    // the default -1 stop), and summing the resulting Roi_scan_image/
    // Roi_diffraction_pattern/roi_4d arrays -- the line ranges are
    // disjoint, so summing is the same as placing each worker's own
    // pixels. Mirrors vSTEM::find_checkpoints (see its own docstring for
    // the dt/rise_t/rise_fall/line_count/chip_id seeding this hands back);
    // unlike vSTEM's own vSTEM_image, Roi_scan_image/Roi_diffraction_pattern
    // are written directly on every hit (see Timepix.hpp::roi()), so Roi
    // does NOT need vSTEM's own "redo the final copy at terminate()" fix.
    // It DID need a different, Roi-specific fix, though: Roi::reset() has
    // its own row-restricted early-exit optimization (finish_line, set from
    // the ROI's y+height in set_roi(), so a small ROI doesn't require
    // decoding the whole frame) that runs independently of stop_at_line.
    // Before that fix, a worker whose own slice ended before finish_line
    // would never satisfy either termination check once its reader stopped,
    // and LiveProcessor::process_data()'s "while (*processor_line != -1)"
    // loop spun forever -- a genuine deadlock, caught by literally hanging
    // in a differential test against a plain single-process Roi reference on
    // real data, not by assuming the port from vSTEM was a complete mirror.
    // Fixed by making fr_total/fr_count_seed stop_at_line/line_number_offset
    // -aware in Roi::reset(), exactly like vSTEM::reset() already did.
    std::vector<std::tuple<uintmax_t, int, uint64_t, std::vector<uint64_t>, std::vector<int>, std::vector<int>, int>> find_checkpoints(int n_splits, bool allow_sidecar = true);
    uintmax_t file_byte_offset = 0;
    int line_number_offset = 0;
    int stop_at_line = -1;
    uint64_t seed_dt = 0;
    std::vector<uint64_t> seed_rise_t = {0, 0, 0, 0};
    std::vector<int> seed_rise_fall = {0, 0, 0, 0};
    std::vector<int> seed_line_count = {-1, -1, -1, -1};
    int seed_chip_id = -1;

    void line_processor(
        size_t &img_num,
        size_t &first_frame,
        size_t &end_frame,
        ProgressMonitor *p_prog_mon,
        size_t &fr_total_u,
        BoundedThreadPool *pool
    );

    // Roi4D's backing store is one flat, contiguous vector<T> (see Roi4D.hpp) --
    // shape comes from the four dim*() accessors, and handing it to numpy is a
    // single copy instead of walking a 4-level nested vector element-by-element.
    template <typename T>
    py::array_t<T> create_py_array(Roi4D<T>& roi4d) {
        std::vector<size_t> shape = {roi4d.dim0(), roi4d.dim1(), roi4d.dim2(), roi4d.dim3()};
        return py::array_t<T>(shape, roi4d.data.data());
    }

    py::array get_4D() {
        if (roi_bitdepth == 8) {
            return create_py_array<uint8_t>(*Roi_4D_8);
        } else if (roi_bitdepth == 16) {
            return create_py_array<uint16_t>(*Roi_4D_16);
        } else if (roi_bitdepth == 32) {
            return create_py_array<uint32_t>(*Roi_4D_32);
        } else {
            throw std::runtime_error("Unsupported bitdepth");
        }
    }

    // Constructor
    Roi(int repetitions, bool ROI_4D) : LiveProcessor(repetitions),
    b_ROI_4D(ROI_4D)
    {
        Roi_4D_8 = std::make_shared<Roi4D<uint8_t>>();
        Roi_4D_16 = std::make_shared<Roi4D<uint16_t>>();
        Roi_4D_32 = std::make_shared<Roi4D<uint32_t>>();
    };

    // Destructor
    ~Roi(){};
};
#endif // !Roi_H