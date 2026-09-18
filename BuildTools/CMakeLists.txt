# /* Copyright (C) 2025 Thomas Friedrich, Chu-Ping Yu, Arno Annys
#  * University of Antwerp - All Rights Reserved. 
#  * You may use, distribute and modify
#  * this code under the terms of the GPL3 license.
#  * You should have received a copy of the GPL3 license with
#  * this file. If not, please visit: 
#  * https://www.gnu.org/licenses/gpl-3.0.en.html
#  * 
#  * Authors: 
#  *   Thomas Friedrich <>
#  *   Chu-Ping Yu <>
#  *   Arno Annys <arno.annys@uantwerpen.be>
#  */

cmake_minimum_required(VERSION 3.16.0)
if(USE_CUDA)
    project(eventem LANGUAGES C CXX CUDA)
else()
    project(eventem LANGUAGES C CXX)
endif()
set(CMAKE_CXX_STANDARD 17)

option(GPRI_OPTION "option for enabling or disabling GPRI" OFF)
option(TORCH "option for enabling or disabling Torch use for frame processing (besides GPRI)" OFF)
option(PIXET "option for enabling or disabling Pixet" OFF)
option(LOG "option for enabling or disabling debug logging" OFF)
set(CMAKE_BUILD_TYPE Release)
# This machine's HDF5 (and its bundled zlib) CMake config packages only export a
# RelWithDebInfo imported configuration, not a Release one -- without this mapping,
# imported targets/variables resolved from find_package(HDF5 ...) silently have no
# usable library path under a plain Release build, which shows up at link time as
# "unresolved external symbol" for every H5:: class and for zlib's compress2/
# compressBound, not as a missing-package configure error.
set(CMAKE_MAP_IMPORTED_CONFIG_RELEASE "Release;RelWithDebInfo")

set(SOURCES
    ../EvenTem/src/utils/Declusterer.hpp
    ../EvenTem/src/utils/SocketConnector.cpp
    ../EvenTem/src/utils/SocketConnector.h
    ../EvenTem/src/utils/FileConnector.cpp
    ../EvenTem/src/utils/FileConnector.h
    ../EvenTem/src/utils/FileConnectorMMAP.cpp
    ../EvenTem/src/utils/FileConnectorMMAP.h
    ../EvenTem/src/utils/FileHeaderParser.hpp
    ../EvenTem/src/utils/ProgressMonitor.cpp
    ../EvenTem/src/utils/ProgressMonitor.h
    ../EvenTem/src/utils/Logger.hpp
    ../EvenTem/src/utils/Roi4D.hpp
    ../EvenTem/src/utils/AtomicWrapper.hpp
    ../EvenTem/src/utils/AnnularDetector.hpp
    ../EvenTem/src/core/LiveProcessor.cpp
    ../EvenTem/src/core/LiveProcessor.h
    ../EvenTem/src/core/Ricom.cpp 
    ../EvenTem/src/core/Ricom.h
    ../EvenTem/src/core/vSTEM.h
    ../EvenTem/src/core/vSTEM.cpp
    ../EvenTem/src/core/tcBF.h
    ../EvenTem/src/core/tcBF.cpp
    ../EvenTem/src/core/Pacbed.h
    ../EvenTem/src/core/Pacbed.cpp
    ../EvenTem/src/core/Var.h
    ../EvenTem/src/core/Var.cpp
    ../EvenTem/src/core/Roi.h
    ../EvenTem/src/core/Roi.cpp
    ../EvenTem/src/core/Electron.h 
    ../EvenTem/src/core/Electron.cpp
    ../EvenTem/src/core/EELS.h
    ../EvenTem/src/core/EELS.cpp
    ../EvenTem/src/core/FourD.h
     ../EvenTem/src/core/FourD.cpp
    ../EvenTem/src/detectors/Cheetah.hpp
    ../EvenTem/src/detectors/Cheetah_pixeltrig.hpp
    ../EvenTem/src/detectors/Advapix.hpp
    ../EvenTem/src/detectors/Timepix.hpp
    ../EvenTem/src/detectors/ElectronFile.hpp
    ../EvenTem/src/detectors/Merlin.hpp
    ../EvenTem/src/detectors/HDF5_DS.hpp
    ../EvenTem/src/detectors/FrameBased.hpp
    ../EvenTem/src/detectors/Numpy.hpp
    ../EvenTem/src/pybind.cpp
)

#----------------------------------------------------------------------------------------------------------------
if (WIN32)
    add_definitions(-DEVENTEM)

    # set(pybind11_ROOT "C:/Users/Customer/.conda/envs/tpx_4DSTEM_env/Lib/site-packages/pybind11")
    find_package(pybind11 REQUIRED)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    # set(FFTW3_DIR ./FFTW)
    # link_directories(${FFTW3_DIR})
    
    # set(HDF5_ROOT "Z:/Arno/HDF5/1.14.3")
    # Per-machine HDF5 install location -- a plain set() here would silently override
    # any -DHDF5_ROOT=... passed on the cmake command line (a normal variable shadows
    # a cache variable of the same name), so only fall back to this default when the
    # caller hasn't already supplied one.
    if(NOT DEFINED HDF5_ROOT)
        set(HDF5_ROOT "C:/Program Files/HDF_Group/HDF5/2.2.0")
    endif()
    find_package(HDF5 1.14.3 REQUIRED COMPONENTS CXX)
    # Without this, H5Cpp.h's DLLVAR macros (see H5api_adpt.h) fall back to plain
    # `extern` instead of `__declspec(dllimport)` for the H5:: wrapper classes'
    # static members (PredType::NATIVE_UINT8, FileAccPropList::DEFAULT, etc.) --
    # compiles fine, but the linker then expects those symbols defined in OUR OWN
    # object files instead of imported from hdf5_cpp.dll, failing as "unresolved
    # external symbol" for basically every H5:: static despite hdf5_cpp.lib being
    # linked correctly.
    add_definitions(-DH5_BUILT_AS_DYNAMIC_LIB)
    # HDF5's own bundled zlib (HDF5_ROOT/lib/zs.lib, see cmake/ZLIB-static*.cmake)
    # was itself built with Intel's compiler and calls Intel-specific intrinsics
    # (_intel_fast_memcpy/_intel_fast_memset) with no fallback -- those are real
    # calls, not just speculative default-lib requests, so there's no way to link
    # against that particular zlib without an actual Intel toolchain. Use vcpkg's
    # plain, MSVC-built zlib instead for the compress2/compressBound symbols
    # FourD.cpp's Zarr writer needs -- functionally identical (same zlib API), and
    # this is the only zlib actually used by our own code; HDF5.dll has its own
    # zlib statically linked into it already, from whenever the HDF Group built it.
    set(HDF5_BUNDLED_ZLIB "C:/vcpkg/installed/x64-windows-static-md/lib/zs.lib")

    set(BUILD_DIR ${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
    message(STATUS "Build directory: ${BUILD_DIR}")
    add_definitions(-DNOMINMAX)
    

    if (LOG)
        add_definitions(-DDBG_LOG)
        message(STATUS "Logging enabled")
    endif()

    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        MESSAGE(STATUS "Using ${CMAKE_CXX_COMPILER_ID} Compiler!")
        set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread -std=c++17")
        set(CMAKE_CXX_FLAGS_DEBUG "-g")
        set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive")
    endif()

    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
	MESSAGE(STATUS "Using ${CMAKE_CXX_COMPILER_ID} Compiler!")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /nologo")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GL")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GA")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MT")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qpar")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /favor:INTEL64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /LTCG:PGInstrument")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /LTCG:PGUpdate")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /LTCG:PGOptimize")
    endif()

    if (PIXET)
        list(APPEND SOURCES ../EvenTem/src/detectors/Advapix.cpp)
        message(STATUS "Pixet enabled")
        add_definitions(-DPIXET_ENABLED)
    endif()

    if (GPRI_OPTION)
        add_definitions(-DGPRI_OPTION_ENABLED)
        message(STATUS "GPRI support enabled")
    endif()  

    if (TORCH)
        add_definitions(-DFRAMEBASED_TORCH_ENABLED)
        message(STATUS "Torch support enabled")
    endif()

    if (GPRI_OPTION OR TORCH)
        set(__LIBTORCH_VERSION "2.6.0") 
        set(__CUDA_VERSION "cu126") 

        message(STATUS "CUDA support enabled")
        set(CMAKE_PREFIX_PATH "c:/libtorch${__LIBTORCH_VERSION}+${__CUDA_VERSION}/libtorch") 
        set(Torch_DIR "c:/libtorch${__LIBTORCH_VERSION}+${__CUDA_VERSION}/libtorch/share/cmake/Torch")  

        find_package(CUDA REQUIRED)
        find_package(Torch REQUIRED)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}") 
    endif()

    if(GPRI_OPTION)
        enable_language(CUDA)
        list(INSERT SOURCES 0 ../EvenTem/src/core/GPRI.h ../EvenTem/src/core/GPRI.cpp ../EvenTem/src/core/GPRI.cu)
    endif()

    pybind11_add_module(eventem ${SOURCES})
    pybind11_add_module(pacbed ${SOURCES})
    # A second, full build of the exact same sources under a distinct importable
    # name (see pybind.cpp's MODULE_NAME docstring for why this is necessary --
    # renaming just the output file doesn't work). Lets a caller (e.g. EDyssey)
    # import both an old, separately-vendored "eventem" build and this repo's
    # current source as "eventem_new" in the same Python process, for a direct
    # side-by-side comparison. Opt-in: only built when -DBUILD_EVENTEM_NEW=ON is
    # passed; off by default so the normal eventem/pacbed build is unaffected.
    option(BUILD_EVENTEM_NEW "Also build a second copy of eventem importable as 'eventem_new'" OFF)
    if(BUILD_EVENTEM_NEW)
        pybind11_add_module(eventem_new ${SOURCES})
        target_compile_definitions(eventem_new PRIVATE MODULE_NAME=eventem_new)
    endif()

    # HDF5's official Windows binary (hdf5_cpp.lib, linked below via HDF5_LIBRARIES)
    # is itself built with Intel's compiler and embeds default-library directives
    # for several Intel runtime libs, which this machine has no Intel toolchain to
    # provide -- confirmed not caused by any compiler flag above (removed each in
    # turn and still hit the same error), and found one at a time as each newly
    # revealed the next ("cannot open file 'libircmt.lib'", then, once that was
    # excluded, 'svml_dispmd.lib'). The actual SVML/runtime implementation those
    # symbols would call lives inside hdf5_cpp.dll itself (already compiled and
    # linked by the HDF Group; libmmd.dll -- the one member of this family actually
    # needed at RUNTIME -- already ships alongside the built .pyd for exactly this
    # reason). Our own link step never calls into any of these directly, so
    # suppressing the demand for the files, rather than supplying them, is the
    # correct fix, not a workaround. Set via the LINK_FLAGS target property (not
    # CMAKE_*_LINKER_FLAGS or target_link_options) since pybind11_add_module's
    # generated targets don't pick up those.
    set(HDF5_INTEL_RUNTIME_EXCLUDE " /NODEFAULTLIB:libircmt.lib /NODEFAULTLIB:svml_dispmd.lib /NODEFAULTLIB:libmmd.lib /NODEFAULTLIB:libdecimal.lib /NODEFAULTLIB:libirc.lib")
    set_property(TARGET eventem APPEND_STRING PROPERTY LINK_FLAGS "${HDF5_INTEL_RUNTIME_EXCLUDE}")
    set_property(TARGET pacbed APPEND_STRING PROPERTY LINK_FLAGS "${HDF5_INTEL_RUNTIME_EXCLUDE}")
    if(BUILD_EVENTEM_NEW)
        set_property(TARGET eventem_new APPEND_STRING PROPERTY LINK_FLAGS "${HDF5_INTEL_RUNTIME_EXCLUDE}")
    endif()


    if(GPRI_OPTION OR TORCH)
        target_include_directories(eventem PRIVATE ${TORCH_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
        target_link_libraries(eventem PRIVATE ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})
    endif()

    if (PIXET)
        target_link_directories(eventem PRIVATE include)
        target_link_libraries(eventem PRIVATE pxcore.lib)

        target_link_directories(pacbed PRIVATE include)
        target_link_libraries(pacbed PRIVATE pxcore.lib)
    endif()

    # target_include_directories(eventem PRIVATE include ../EvenTem/src/core ../EvenTem/src/utils ../EvenTem/src/detectors)
    target_include_directories(eventem PRIVATE include ../EvenTem/src/core ../EvenTem/src/utils ../EvenTem/src/detectors ${Python3_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS}) #${FFTW3_DIR}
    # target_link_libraries(eventem PRIVATE pybind11::module ${CMAKE_DL_LIBS} ws2_32)
    target_link_libraries(eventem PRIVATE ${Python3_LIBRARIES} ${CMAKE_DL_LIBS} ${HDF5_LIBRARIES} ${HDF5_BUNDLED_ZLIB} ws2_32) #libfftw3f-3

    target_include_directories(pacbed PRIVATE include ../EvenTem/src/core ../EvenTem/src/utils ../EvenTem/src/detectors ${Python3_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
    target_link_libraries(pacbed PRIVATE ${Python3_LIBRARIES} ${CMAKE_DL_LIBS} ${HDF5_LIBRARIES} ${HDF5_BUNDLED_ZLIB} ws2_32)

    if(BUILD_EVENTEM_NEW)
        target_include_directories(eventem_new PRIVATE include ../EvenTem/src/core ../EvenTem/src/utils ../EvenTem/src/detectors ${Python3_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
        target_link_libraries(eventem_new PRIVATE ${Python3_LIBRARIES} ${CMAKE_DL_LIBS} ${HDF5_LIBRARIES} ${HDF5_BUNDLED_ZLIB} ws2_32)
        if (PIXET)
            target_link_directories(eventem_new PRIVATE include)
            target_link_libraries(eventem_new PRIVATE pxcore.lib)
        endif()
        if(GPRI_OPTION OR TORCH)
            target_include_directories(eventem_new PRIVATE ${TORCH_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
            target_link_libraries(eventem_new PRIVATE ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})
        endif()
    endif()

endif (WIN32)
#----------------------------------------------------------------------------------------------------------------
if (APPLE)
    add_definitions(-DEVENTEM)

    find_package(PkgConfig)
    find_package(pybind11 REQUIRED)
    find_package(HDF5 REQUIRED COMPONENTS CXX)
    
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

    if (GPRI_OPTION)
        pybind11_add_module(eventem ${SOURCES})
        target_link_libraries(eventem PRIVATE pybind11::module ${CMAKE_DL_LIBS})
        target_include_directories(eventem  PRIVATE include ../EvenTem/src/core ../EvenTem/src/utils ../EvenTem/src/detectors)
    else()
        pybind11_add_module(eventem ${SOURCES})
        target_link_libraries(eventem PRIVATE pybind11::module ${CMAKE_DL_LIBS})
        target_include_directories(eventem  PRIVATE include ../EvenTem/src/core ../EvenTem/src/utils ../EvenTem/src/detectors)
    endif()

    if (GPRI_OPTION)
        add_definitions(-DGPRI_OPTION_ENABLED)
        message(STATUS "GPRI support enabled")
        set(CMAKE_PREFIX_PATH "/Users/arnoannys/Programming/libtorch")
        set(Torch_DIR "/Users/arnoannys/Programming/libtorch/share/cmake/Torch")
        find_package(Torch REQUIRED)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

        list(APPEND SOURCES ../EvenTem/src/core/GPRI.h ../EvenTem/src/core/GPRI.cpp)
        target_link_libraries(eventem PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(eventem  PRIVATE ${TORCH_INCLUDE_DIRS})
    endif()

    target_link_libraries(eventem PRIVATE ${HDF5_LIBRARIES} pybind11::module ${CMAKE_DL_LIBS})
    target_include_directories(eventem  PRIVATE include ../EvenTem/src/core ../EvenTem/src/utils ../EvenTem/src/detectors ${HDF5_INCLUDE_DIRS})
    

endif (APPLE) 
#----------------------------------------------------------------------------------------------------------------
if (UNIX AND NOT APPLE)
    add_definitions(-DEVENTEM)

    set(pybind11_ROOT "/usr")
    find_package(PkgConfig)
    find_package(pybind11 REQUIRED)
    add_definitions(-DNOMINMAX)

    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")

    message(STATUS "HDF5 support enabled")
    set(HDF5_ROOT "/usr")
    find_package(HDF5 REQUIRED COMPONENTS CXX)

    if (TORCH)
        add_definitions(-DTORCH_ENABLED)
        message(STATUS "Torch support enabled")
    endif()

    set(CMAKE_CXX_FLAGS "-w -Wall -Wextra -pthread")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

    if (GPRI_OPTION)
        add_definitions(-DTORCH_ENABLED)
        add_definitions(-DGPRI_OPTION_ENABLED)
        message(STATUS "GPRI support enabled")

        if (USE_CUDA)
            add_definitions(-DUSE_CUDA)
            set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++")
            message(STATUS "CUDA support enabled")
            set(CMAKE_PREFIX_PATH "/home/arno/libtorch$ENV{__LIBTORCH_VERSION}+$ENV{__CUDA_VERSION}/libtorch") 
            set(Torch_DIR "/home/arno/libtorch$ENV{__LIBTORCH_VERSION}+$ENV{__CUDA_VERSION}/libtorch/share/cmake/Torch") 
        else()
            message(STATUS "CUDA support disabled")
            set(CMAKE_PREFIX_PATH "/home/arno/libtorch$ENV{__LIBTORCH_VERSION}+cpu/libtorch") 
            set(Torch_DIR "/home/arno/libtorch$ENV{__LIBTORCH_VERSION}+cpu/libtorch/share/cmake/Torch") 
        endif()
        
        find_package(Torch REQUIRED)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")  
    endif()  

    pybind11_add_module(eventem ${SOURCES})
    target_include_directories(eventem PRIVATE include ../EvenTem/src/core ../EvenTem/src/utils ../EvenTem/src/detectors ${HDF5_INCLUDE_DIRS})
    target_link_libraries(eventem PRIVATE pybind11::module ${HDF5_LIBRARIES})


    if (GPRI_OPTION AND USE_CUDA)
        enable_language(CUDA)
        list(INSERT SOURCES 0 ../EvenTem/src/core/GPRI.h ../EvenTem/src/core/GPRI.cpp ../EvenTem/src/core/GPRI.cu)
    endif()
        
    if(GPRI_OPTION)
        target_include_directories(eventem PRIVATE ${TORCH_INCLUDE_DIRS})
        target_link_libraries(eventem PRIVATE ${TORCH_LIBRARIES})

        message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
        message(STATUS "Torch include directories: ${TORCH_INCLUDE_DIRS}")
    endif()

endif (UNIX AND NOT APPLE)
