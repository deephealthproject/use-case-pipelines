# This file is part of ECVL an European Computer Vision Library.
# Developed within the DeepHealth project.
# Boosting AI in Europe.
#
# The MIT License (MIT)
#
# Copyright (c) 2019
#           Laura Canalini, <laura.canalini@unimore.it>
#           Michele Cancilla, <michele.cancilla@unimore.it>
#           Federico Bolelli, <federico.bolelli@unimore.it>
#           Costantino Grana, <costantino.grana@unimore.it>
#           Stefano Allegretti, <stefano.allegretti@unimore.it>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required (VERSION 3.13)

project (pipeline_ecvl_eddl CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")
set(DOWNLOAD_MNIST ON CACHE BOOL "If flagged the mnist dataset will be automatically downloaded")
#--------------------------------------------------------------------------------------------------------
# BEGIN: Check for c++17 support
#--------------------------------------------------------------------------------------------------------
# This should be done as follows but it doesn't work!?
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "C flags: ${CMAKE_C_FLAGS}")
#--------------------------------------------------------------------------------------------------------
# END: Check for c++17 support
#--------------------------------------------------------------------------------------------------------

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

#--------------------------------------------------------------------------------------------------------
# BEGIN: static/dinamic runtime
#--------------------------------------------------------------------------------------------------------
set(pipeline_ecvl_eddl_STATIC ON CACHE BOOL "Whether to build static or dinamic libraries")

if(MSVC)
    if(pipeline_ecvl_eddl_STATIC)
        message(STATUS "Setting static runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
    else()
        message(STATUS "Setting dinamic runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
    endif()
endif()
#--------------------------------------------------------------------------------------------------------
# END: static/dinamic runtime
#--------------------------------------------------------------------------------------------------------

add_library(METRICS STATIC "")
add_library(MODELS STATIC "")
add_library(UTILS STATIC "")
add_executable(MNIST_BATCH)
add_executable(SKIN_LESION_CLASSIFICATION)
add_executable(SKIN_LESION_SEGMENTATION)
add_subdirectory(src)
add_subdirectory("src/metrics")
add_subdirectory("src/models")
add_subdirectory("src/utils")

# Set SKIN_LESION_CLASSIFICATION as default
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT SKIN_LESION_CLASSIFICATION)

set(ECVL_BUILD ON CACHE BOOL "Whether to build ECVL or not (requires ECVL)")

find_package(ecvl)
if(ecvl_FOUND)
    target_link_libraries(METRICS ECVL_CORE ECVL_EDDL DATASET_PARSER)
    target_link_libraries(UTILS METRICS)
    target_link_libraries(MODELS UTILS)
    target_link_libraries(MNIST_BATCH MODELS)
    target_link_libraries(SKIN_LESION_CLASSIFICATION MODELS)
    target_link_libraries(SKIN_LESION_SEGMENTATION MODELS)
    include_directories("${ecvl_DIR}/include")
    message(STATUS "ECVL found")
endif()

# Download and unpack MNIST dataset
if(DOWNLOAD_MNIST)
    if (IS_DIRECTORY "${CMAKE_BINARY_DIR}/mnist")
        message(STATUS "${CMAKE_BINARY_DIR}/mnist dataset folder already exists, download skipped!")
    else()
        message(STATUS "Downloading MNIST...")
        file(DOWNLOAD https://www.dropbox.com/s/8rslhjpects95k0/mnist.zip "${CMAKE_BINARY_DIR}/mnist.zip"
            STATUS status
            SHOW_PROGRESS
        )
        list(GET status 0 error_code)
        if(error_code)
            message(WARNING "CMake is not able to download the mnist dataset.\n
            Please, retry or download it manually (The dataset is also available at http://imagelab.ing.unimore.it/files/DeepHealth/mnist.zip).")
        else()
            message(STATUS "Downloading done.")
            if(EXISTS "${CMAKE_BINARY_DIR}/mnist.zip")
                message(STATUS "Unpacking (it may take a while)...")
                execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvf "${CMAKE_BINARY_DIR}/mnist.zip" WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" OUTPUT_QUIET)
                message(STATUS "Unpacking done.")
            endif()
        endif()
    endif()
endif()