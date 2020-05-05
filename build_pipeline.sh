#!/bin/bash

UCP_PATH=`pwd`
DEVICE="GPU"
BUILD_TYPE="Release"
# BUILD_TYPE="Debug"
DEP_DIR="deephealth_lin"

mkdir -p $DEP_DIR && cd $DEP_DIR

############ EDDL
git clone --recurse-submodule https://github.com/deephealthproject/eddl.git 
cd eddl
git checkout tags/0.4.3
mkdir -p build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DBUILD_TARGET=$DEVICE -DBUILD_SHARED_LIB=OFF -DBUILD_PROTOBUF=OFF -DCMAKE_INSTALL_PREFIX=install ..
make -j$(nproc) && make install
EDDL_INSTALL_DIR=$UCP_PATH/$DEP_DIR/eddl/build/cmake

############ OPENCV
cd $UCP_PATH/$DEP_DIR
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout tags/4.3.0
mkdir -p build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=install -DBUILD_LIST=core,imgproc,imgcodecs,photo -DBUILD_opencv_apps=OFF -DBUILD_opencv_java_bindings_generator=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python_tests=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF -DBUILD_JPEG=ON -DBUILD_IPP_IW=OFF -DBUILD_ITT=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_PNG=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_TIFF=ON -DBUILD_WEBP=OFF -DBUILD_ZLIB=OFF -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DWITH_FFMPEG=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_JPEG=ON -DWITH_LAPACK=OFF -DWITH_MATLAB=OFF -DWITH_OPENCL=OFF -DWITH_OPENEXR=OFF -DWITH_OPENGL=OFF -DWITH_PNG=ON -DWITH_PROTOBUF=OFF -DWITH_QT=OFF -DWITH_TBB=OFF -DWITH_TIFF=ON -DWITH_V4L=OFF -DWITH_WEBP=OFF ..
make -j$(nproc) && make install
OPENCV_INSTALL_DIR=$UCP_PATH/$DEP_DIR/opencv/build

############ ECVL
cd $UCP_PATH/$DEP_DIR
git clone https://github.com/deephealthproject/ecvl.git
cd ecvl
git checkout tags/v0.2.1 # Release 0.2.1
mkdir -p build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DOpenCV_DIR=$OPENCV_INSTALL_DIR -Deddl_DIR=$EDDL_INSTALL_DIR -DECVL_BUILD_EDDL=ON -DECVL_DATASET=ON -DECVL_BUILD_GUI=OFF -DCMAKE_INSTALL_PREFIX=install ..
make -j$(nproc) && make install
ECVL_INSTALL_DIR=$UCP_PATH/$DEP_DIR/ecvl/build/install

############ PIPELINE
cd $UCP_PATH
mkdir -p bin_lin && cd bin_lin
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -Decvl_DIR=$ECVL_INSTALL_DIR ..
make -j$(nproc) && ./MNIST_BATCH
