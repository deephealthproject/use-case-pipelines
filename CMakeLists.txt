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
add_executable(MNIST_BATCH)
add_executable(SKIN_LESION_CLASSIFICATION_TRAINING)
add_executable(SKIN_LESION_SEGMENTATION_TRAINING)
add_executable(SKIN_LESION_CLASSIFICATION_INFERENCE)
add_executable(SKIN_LESION_SEGMENTATION_INFERENCE)
add_executable(PNEUMOTHORAX_SEGMENTATION_TRAINING)
add_executable(PNEUMOTHORAX_SEGMENTATION_INFERENCE)
add_executable(COPY_GROUND_TRUTH_PNEUMOTHORAX)
add_subdirectory(src)
add_subdirectory("src/metrics")
add_subdirectory("src/models")

# Set SKIN_LESION_CLASSIFICATION_TRAINING as default
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT SKIN_LESION_CLASSIFICATION_TRAINING)

set(ECVL_BUILD ON CACHE BOOL "Whether to build ECVL or not (requires ECVL)")

find_package(ecvl)
if(ecvl_FOUND)
    target_link_libraries(METRICS ECVL_CORE ECVL_EDDL DATASET)
    target_link_libraries(MODELS METRICS)
    target_link_libraries(MNIST_BATCH MODELS)
    target_link_libraries(SKIN_LESION_CLASSIFICATION_TRAINING MODELS)
    target_link_libraries(SKIN_LESION_SEGMENTATION_TRAINING MODELS)
    target_link_libraries(SKIN_LESION_CLASSIFICATION_INFERENCE MODELS)
    target_link_libraries(SKIN_LESION_SEGMENTATION_INFERENCE MODELS)
    target_link_libraries(PNEUMOTHORAX_SEGMENTATION_TRAINING MODELS)
    target_link_libraries(PNEUMOTHORAX_SEGMENTATION_INFERENCE MODELS)
    target_link_libraries(COPY_GROUND_TRUTH_PNEUMOTHORAX stdc++fs)
    set_target_properties(METRICS PROPERTIES FOLDER "Utils")
    set_target_properties(MODELS PROPERTIES FOLDER "Utils")
    set_target_properties(COPY_GROUND_TRUTH_PNEUMOTHORAX PROPERTIES FOLDER "Utils")

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