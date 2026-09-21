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

#include <iostream>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include "Ricom.h"
#include "LiveProcessor.h"
#include "vSTEM.h"
#include "Pacbed.h"
#include "Var.h"
#include "Roi.h"
#include "core/Electron.h"
#include "EELS.h"
#include "FourD.h"
#include "tcBF.h"

#ifdef GPRI_OPTION_ENABLED
        #include "GPRI.h"
#endif

namespace py = pybind11;

// Lets a build produce a distinctly-importable module (e.g. "eventem_new",
// for a second build of this same source coexisting in the same Python
// process as the main "eventem" module -- Python's import machinery keys an
// extension module on the PyInit_<name> symbol derived from its target/file
// name, so two separately-built .pyd's literally named "eventem" can't both
// be imported in one process; renaming just the file doesn't work either
// (confirmed directly) since the compiled-in init symbol name has to match
// too). Pass -DMODULE_NAME=eventem_new (CMake target_compile_definitions) to
// build under that name; every other target is unaffected by this default.
#ifndef MODULE_NAME
#define MODULE_NAME eventem
#endif

PYBIND11_MODULE(MODULE_NAME, m) {

        py::class_<LiveProcessor>(m, "LiveProcessor", py::module_local())
        .def_readwrite("nx", &LiveProcessor::nx)
        .def_readwrite("ny", &LiveProcessor::ny)
        .def_readwrite("dt", &LiveProcessor::dt)
        .def_readwrite("detector_size", &LiveProcessor::n_cam)
        .def("set_socket", &LiveProcessor::set_socket)
        .def("accept_socket", &LiveProcessor::accept_socket)
        .def("close_socket", &LiveProcessor::close_socket)
        .def_readwrite("n_threads", &LiveProcessor::n_threads)
        .def_readwrite("file_path", &LiveProcessor::file_path)
        .def_readwrite("repetitions", &LiveProcessor::rep)
        .def("set_dwell_time", &LiveProcessor::set_dwell_time)
        .def_readonly("progress", &LiveProcessor::progress_percent)
        .def_readwrite("b_cumulative", &LiveProcessor::b_cumulative)   
        .def_readwrite("b_continuous", &LiveProcessor::b_continuous)
        .def_readwrite("rc_quit", &LiveProcessor::rc_quit)
        .def_readonly("elapsed_seconds", &LiveProcessor::elapsed_seconds_vec)
        .def_readonly("reached_pp_id", &LiveProcessor::reached_pp_id)
        .def("set_pattern_file", &LiveProcessor::set_pattern_file)
        .def_readonly("processing_rate", &LiveProcessor::processing_rate)
        .def_readonly("processor_line", &LiveProcessor::processor_line)
        .def("set_file", &LiveProcessor::set_file);


        py::class_<Ricom,LiveProcessor>(m, "Ricom", py::module_local())
        .def(py::init<int>(),py::arg("repetitions"))
        .def("add_child", py::overload_cast<vSTEM*>(&Ricom::add_child))
        .def("set_kernel", &Ricom::set_kernel)
        .def("set_kernel_filtered", &Ricom::set_kernel_filtered)
        .def("run", &Ricom::run)
        .def_readonly("offset", &Ricom::offset)
        .def("set_offset",&Ricom::set_offset)
        .def("get_kernel", &Ricom::get_kernel)
        .def("set_mask", &Ricom::set_masked_com)
        .def_readonly("comx_image", &Ricom::comx_image)
        .def_readonly("comy_image", &Ricom::comy_image)
        .def_readonly("ricom_stack", &Ricom::ricom_image_stack)
        .def_readonly("ricom_image", &Ricom::ricom_image);


        py::class_<vSTEM,LiveProcessor>(m, "vSTEM", py::module_local())
        .def(py::init<int>(),py::arg("repetitions"))
        .def_readwrite("inner_radia", &vSTEM::inner_radia)
        .def_readwrite("outer_radia", &vSTEM::outer_radia)
        .def_readonly("offsets", &vSTEM::offsets)
        .def("set_offsets", &vSTEM::set_offsets)
        .def("get_detector", &vSTEM::get_detector,py::return_value_policy::copy)
        .def("set_detector_mask", &vSTEM::set_detector_mask)
        .def("run", &vSTEM::run)
        .def_readwrite("allow_torch", &vSTEM::allow_torch)
        .def_readwrite("allow_cuda", &vSTEM::allow_cuda)        
        .def_readonly("vSTEM_stack", &vSTEM::vSTEM_stack)
        .def_readwrite("decluster", &vSTEM::decluster)
        .def_readwrite("dtime", &vSTEM::dtime)
        .def_readwrite("dspace", &vSTEM::dspace)
        .def_readwrite("cluster_range", &vSTEM::cluster_range)
        .def_readwrite("tot_per_electron", &vSTEM::tot_per_electron)
        .def_readwrite("electron_count_lut", &vSTEM::electron_count_lut)
        .def_readwrite("electron_count_lut_file", &vSTEM::electron_count_lut_file)
        .def_readonly("clustersize_histogram", &vSTEM::clustersize_histogram)
        .def_readonly("energy_histogram", &vSTEM::energy_histogram)
        .def_readonly("clustersize_tot_histogram", &vSTEM::clustersize_tot_histogram)
        .def("find_checkpoints", &vSTEM::find_checkpoints, py::arg("n_splits"), py::arg("allow_sidecar") = true)
        .def_readwrite("file_byte_offset", &vSTEM::file_byte_offset)
        .def_readwrite("line_number_offset", &vSTEM::line_number_offset)
        .def_readwrite("stop_at_line", &vSTEM::stop_at_line)
        .def_readwrite("seed_dt", &vSTEM::seed_dt)
        .def_readwrite("seed_rise_t", &vSTEM::seed_rise_t)
        .def_readwrite("seed_rise_fall", &vSTEM::seed_rise_fall)
        .def_readwrite("seed_line_count", &vSTEM::seed_line_count)
        .def("find_checkpoints_pixeltrig", &vSTEM::find_checkpoints_pixeltrig, py::arg("n_splits"))
        .def_readwrite("seed_probe_count_chip", &vSTEM::seed_probe_count_chip)
        .def_readwrite("seed_chip_id", &vSTEM::seed_chip_id)
        .def("get_image",[](vSTEM& self) {
            return py::array_t<size_t>(
                {self.nx, self.ny},
                {sizeof(size_t) * self.ny, sizeof(size_t)},
                self.vSTEM_image.data(),
                py::cast(&self)
            );
        })
        .def_readonly("vSTEM_image", &vSTEM::vSTEM_image);

        py::class_<tcBF,LiveProcessor>(m, "tcBF", py::module_local())
        .def(py::init<int>(),py::arg("repetitions"))
        .def("set_detector_mask", &tcBF::set_detector_mask)
        .def("run", &tcBF::run)        
        .def_readonly("tcBF_stack", &tcBF::tcBF_stack)
        .def_readonly("BF_image", &tcBF::BF_image);

                
        py::class_<FourD<8>, LiveProcessor>(m, "FourD8", py::module_local())
        .def(py::init<const std::string&, int, int,int>(), py::arg("output_filename"), py::arg("repetitions"), py::arg("bitdepth"),py::arg("compression_factor"))
        .def("run", &FourD<8>::run)
        .def("allocate_chunk", &FourD<8>::allocate_chunk)
        .def("init_4D_file", &FourD<8>::init_4D_file)
        .def_readwrite("det_bin", &FourD<8>::det_bin)
        .def_readwrite("scan_bin", &FourD<8>::scan_bin)
        .def_readwrite("chunksize", &FourD<8>::chunksize)
        .def_readwrite("chunksize_x", &FourD<8>::chunksize_x)
        .def_readwrite("format", &FourD<8>::format)
        .def_readwrite("save_metadata", &FourD<8>::save_metadata)
        .def_readwrite("decluster", &FourD<8>::decluster)
        .def_readwrite("dtime", &FourD<8>::dtime)
        .def_readwrite("dspace", &FourD<8>::dspace)
        .def_readwrite("cluster_range", &FourD<8>::cluster_range)
        .def_readwrite("tot_per_electron", &FourD<8>::tot_per_electron)
        .def_readwrite("electron_count_lut", &FourD<8>::electron_count_lut)
        .def_readwrite("electron_count_lut_file", &FourD<8>::electron_count_lut_file)
        .def_readonly("clustersize_histogram", &FourD<8>::clustersize_histogram)
        .def_readonly("energy_histogram", &FourD<8>::energy_histogram)
        .def_readonly("clustersize_tot_histogram", &FourD<8>::clustersize_tot_histogram)
        .def_readonly("Dose_image", &FourD<8>::Dose_image);


        py::class_<FourD<16>, LiveProcessor>(m, "FourD16", py::module_local())
        .def(py::init<const std::string&, int, int,int>(), py::arg("output_filename"), py::arg("repetitions"), py::arg("bitdepth"),py::arg("compression_factor"))
        .def("run", &FourD<16>::run)
        .def("allocate_chunk", &FourD<16>::allocate_chunk)
        .def("init_4D_file", &FourD<16>::init_4D_file)
        .def_readwrite("det_bin", &FourD<16>::det_bin)
        .def_readwrite("scan_bin", &FourD<16>::scan_bin)
        .def_readwrite("chunksize", &FourD<16>::chunksize)
        .def_readwrite("chunksize_x", &FourD<16>::chunksize_x)
        .def_readwrite("format", &FourD<16>::format)
        .def_readwrite("save_metadata", &FourD<16>::save_metadata)
        .def_readwrite("decluster", &FourD<16>::decluster)
        .def_readwrite("dtime", &FourD<16>::dtime)
        .def_readwrite("dspace", &FourD<16>::dspace)
        .def_readwrite("cluster_range", &FourD<16>::cluster_range)
        .def_readwrite("tot_per_electron", &FourD<16>::tot_per_electron)
        .def_readwrite("electron_count_lut", &FourD<16>::electron_count_lut)
        .def_readwrite("electron_count_lut_file", &FourD<16>::electron_count_lut_file)
        .def_readonly("clustersize_histogram", &FourD<16>::clustersize_histogram)
        .def_readonly("energy_histogram", &FourD<16>::energy_histogram)
        .def_readonly("clustersize_tot_histogram", &FourD<16>::clustersize_tot_histogram)
        .def_readonly("Dose_image", &FourD<16>::Dose_image);


        py::class_<FourD<32>, LiveProcessor>(m, "FourD32", py::module_local())
        .def(py::init<const std::string&, int, int,int>(), py::arg("output_filename"), py::arg("repetitions"), py::arg("bitdepth"),py::arg("compression_factor"))
        .def("run", &FourD<32>::run)
        .def("allocate_chunk", &FourD<32>::allocate_chunk)
        .def("init_4D_file", &FourD<32>::init_4D_file)
        .def_readwrite("det_bin", &FourD<32>::det_bin)
        .def_readwrite("scan_bin", &FourD<32>::scan_bin)
        .def_readwrite("chunksize", &FourD<32>::chunksize)
        .def_readwrite("chunksize_x", &FourD<32>::chunksize_x)
        .def_readwrite("format", &FourD<32>::format)
        .def_readwrite("save_metadata", &FourD<32>::save_metadata)
        .def_readwrite("decluster", &FourD<32>::decluster)
        .def_readwrite("dtime", &FourD<32>::dtime)
        .def_readwrite("dspace", &FourD<32>::dspace)
        .def_readwrite("cluster_range", &FourD<32>::cluster_range)
        .def_readwrite("tot_per_electron", &FourD<32>::tot_per_electron)
        .def_readwrite("electron_count_lut", &FourD<32>::electron_count_lut)
        .def_readwrite("electron_count_lut_file", &FourD<32>::electron_count_lut_file)
        .def_readonly("clustersize_histogram", &FourD<32>::clustersize_histogram)
        .def_readonly("energy_histogram", &FourD<32>::energy_histogram)
        .def_readonly("clustersize_tot_histogram", &FourD<32>::clustersize_tot_histogram)
        .def_readonly("Dose_image", &FourD<32>::Dose_image);


        #ifdef GPRI_OPTION_ENABLED
                py::class_<GPRI,LiveProcessor>(m, "GPRI", py::module_local())
                .def(py::init<int,std::string&,bool>(),py::arg("repetitions"), py::arg("path_to_library"),py::arg("allow_cuda"))
                .def("add_child", py::overload_cast<vSTEM*>(&GPRI::add_child))
                .def("get_GPRI_result", &GPRI::get_GPRI_result)
                .def("get_GPRI_stack_result", &GPRI::get_GPRI_stack_result)
                .def("set_SparseFrame", &GPRI::set_SparseFrame)
                .def_readwrite("scan_index", &GPRI::scan_index)
                .def_readwrite("interval_R_ratio", &GPRI::interval_R_ratio)
                .def_readwrite("center", &GPRI::center)
                .def_readwrite("center_scan", &GPRI::center_scan) 
                .def_readwrite("N_pxl_radius", &GPRI::N_pxl_radius)
                .def_readwrite("detector_bin", &GPRI::detector_bin)
                .def_readwrite("scan_bin", &GPRI::scan_bin)
                .def_readonly("result_shape", &GPRI::result_shape)
                .def_readonly("N_electrons_map_scangrid", &GPRI::N_electrons_map_scangrid)
                .def_readwrite("normalize", &GPRI::normalize)   
                .def_readonly("N_image_modes", &GPRI::image_modes)
                .def_readonly("kernelsizes", &GPRI::kernelsizes)
                .def("run", &GPRI::run);
        #endif
        
        py::class_<Pacbed,LiveProcessor>(m, "Pacbed", py::module_local())
        .def(py::init<int>(),py::arg("repetitions"))
        .def("run", &Pacbed::run)
        .def("set_scan_mask", &Pacbed::set_scan_mask)
        .def_readwrite("decluster", &Pacbed::decluster)
        .def_readwrite("dtime", &Pacbed::dtime)
        .def_readwrite("dspace", &Pacbed::dspace)
        .def_readwrite("cluster_range", &Pacbed::cluster_range)
        .def_readwrite("tot_per_electron", &Pacbed::tot_per_electron)
        .def_readwrite("electron_count_lut", &Pacbed::electron_count_lut)
        .def_readwrite("electron_count_lut_file", &Pacbed::electron_count_lut_file)
        .def_readonly("clustersize_histogram", &Pacbed::clustersize_histogram)
        .def_readonly("energy_histogram", &Pacbed::energy_histogram)
        .def_readonly("clustersize_tot_histogram", &Pacbed::clustersize_tot_histogram)
        .def_readonly("Pacbed_image", &Pacbed::Pacbed_image);

        py::class_<Roi,LiveProcessor>(m, "Roi", py::module_local())
        .def(py::init<int,bool>(), py::arg("repetitions"), py::arg("extract_4D"))
        .def("run", &Roi::run)
        .def_readonly("Roi_scan_image", &Roi::Roi_scan_image)
        .def_readonly("Roi_diffraction_pattern", &Roi::Roi_diffraction_pattern)
        .def_readonly("Roi_scan_image_stack", &Roi::Roi_scan_image_stack)
        .def_readonly("Roi_diffraction_pattern_stack", &Roi::Roi_diffraction_pattern_stack)
        .def("get_4D", &Roi::get_4D)
        .def_readwrite("det_bin", &Roi::det_bin)
        .def("get_roi", &Roi::get_roi)
        .def("set_roi_mask", &Roi::set_roi_mask)
        .def("set_bitdepth", &Roi::set_bitdepth)
        .def("set_roi", &Roi::set_roi,py::arg("x"),py::arg("y"),py::arg("width"),py::arg("height"))
        .def_readwrite("decluster", &Roi::decluster)
        .def_readwrite("dtime", &Roi::dtime)
        .def_readwrite("dspace", &Roi::dspace)
        .def_readwrite("cluster_range", &Roi::cluster_range)
        .def_readwrite("tot_per_electron", &Roi::tot_per_electron)
        .def_readwrite("electron_count_lut", &Roi::electron_count_lut)
        .def_readwrite("electron_count_lut_file", &Roi::electron_count_lut_file)
        .def_readonly("clustersize_histogram", &Roi::clustersize_histogram)
        .def_readonly("energy_histogram", &Roi::energy_histogram)
        .def_readonly("clustersize_tot_histogram", &Roi::clustersize_tot_histogram)
        .def_readwrite("tot_mode", &Roi::tot_mode);

        py::class_<Var,LiveProcessor>(m, "Var", py::module_local())
        .def(py::init<int>(),py::arg("repetitions"))
        .def("run", &Var::run)
        .def("set_offset", &Var::set_offset)
        .def_readwrite("inner_radius", &Var::inner_radius)
        .def_readwrite("outer_radius", &Var::outer_radius)
        .def_readonly("offset", &Var::offset)
        .def_readwrite("decluster", &Var::decluster)
        .def_readwrite("dtime", &Var::dtime)
        .def_readwrite("dspace", &Var::dspace)
        .def_readwrite("cluster_range", &Var::cluster_range)
        .def_readwrite("tot_per_electron", &Var::tot_per_electron)
        .def_readwrite("electron_count_lut", &Var::electron_count_lut)
        .def_readwrite("electron_count_lut_file", &Var::electron_count_lut_file)
        .def_readonly("clustersize_histogram", &Var::clustersize_histogram)
        .def_readonly("energy_histogram", &Var::energy_histogram)
        .def_readonly("clustersize_tot_histogram", &Var::clustersize_tot_histogram)
        .def_readonly("Var_image", &Var::Var_image);

        py::class_<Electron,LiveProcessor>(m, "Electron", py::module_local())
        .def(py::init<int>(),py::arg("repetitions"))
        .def("run", &Electron::run)
        .def_readwrite("decluster", &Electron::decluster)
        .def_readwrite("dtime", &Electron::dtime)
        .def_readwrite("dspace", &Electron::dspace)
        .def_readwrite("cluster_range", &Electron::cluster_range)
        .def_readwrite("n_threads", &Electron::n_threads)
        .def_readwrite("x_crop", &Electron::x_crop)
        .def_readwrite("y_crop", &Electron::y_crop)
        .def_readwrite("scan_bin", &Electron::scan_bin)
        .def_readwrite("detector_bin", &Electron::detector_bin)
        .def_readwrite("clustersize_histogram", &Electron::clustersize_histogram)
        .def_readwrite("energy_histogram", &Electron::energy_histogram);

        py::class_<EELS,LiveProcessor>(m, "EELS", py::module_local())
        .def(py::init<int>(),py::arg("repetitions"))
        .def("run", &EELS::run)
        .def_readonly("EELS_data", &EELS::EELS_data);

}


PYBIND11_MODULE(pacbed, m) {

        py::class_<LiveProcessor>(m, "LiveProcessor", py::module_local())
        .def_readwrite("nx", &LiveProcessor::nx)
        .def_readwrite("ny", &LiveProcessor::ny)
        .def_readwrite("dt", &LiveProcessor::dt)
        .def_readwrite("detector_size", &LiveProcessor::n_cam)
        .def("set_socket", &LiveProcessor::set_socket)
        .def("accept_socket", &LiveProcessor::accept_socket)
        .def("close_socket", &LiveProcessor::close_socket)
        .def_readwrite("n_threads", &LiveProcessor::n_threads)
        .def_readwrite("file_path", &LiveProcessor::file_path)
        .def_readwrite("repetitions", &LiveProcessor::rep)
        .def("set_dwell_time", &LiveProcessor::set_dwell_time)
        .def_readonly("progress", &LiveProcessor::progress_percent)
        .def_readwrite("b_cumulative", &LiveProcessor::b_cumulative)   
        .def_readwrite("b_continuous", &LiveProcessor::b_continuous)
        .def_readwrite("rc_quit", &LiveProcessor::rc_quit)
        .def_readonly("elapsed_seconds", &LiveProcessor::elapsed_seconds_vec)
        .def_readonly("reached_pp_id", &LiveProcessor::reached_pp_id)
        .def("set_pattern_file", &LiveProcessor::set_pattern_file)
        .def_readonly("processing_rate", &LiveProcessor::processing_rate)
        .def_readonly("processor_line", &LiveProcessor::processor_line)
        .def("set_file", &LiveProcessor::set_file);

        py::class_<Pacbed,LiveProcessor>(m, "Pacbed", py::module_local())
        .def(py::init<int>(),py::arg("repetitions"))
        .def("run", &Pacbed::run)
        .def("set_scan_mask", &Pacbed::set_scan_mask)
        .def_readwrite("decluster", &Pacbed::decluster)
        .def_readwrite("dtime", &Pacbed::dtime)
        .def_readwrite("dspace", &Pacbed::dspace)
        .def_readwrite("cluster_range", &Pacbed::cluster_range)
        .def_readwrite("tot_per_electron", &Pacbed::tot_per_electron)
        .def_readwrite("electron_count_lut", &Pacbed::electron_count_lut)
        .def_readwrite("electron_count_lut_file", &Pacbed::electron_count_lut_file)
        .def_readonly("clustersize_histogram", &Pacbed::clustersize_histogram)
        .def_readonly("energy_histogram", &Pacbed::energy_histogram)
        .def_readonly("clustersize_tot_histogram", &Pacbed::clustersize_tot_histogram)
        .def_readonly("Pacbed_image", &Pacbed::Pacbed_image);
}