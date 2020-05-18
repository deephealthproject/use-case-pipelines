#!/bin/bash

UCP_PATH=$(pwd)
DEVICE="GPU"
BUILD_TYPE="Release"
# BUILD_TYPE="Debug"
DEPENDENCIES_DIR="deephealth_lin"
OPENCV_VERSION=4.3.0
PROC=$(($(nproc)-1))

mkdir -p $DEPENDENCIES_DIR && cd $DEPENDENCIES_DIR

############ EDDL
git clone --recurse-submodule https://github.com/deephealthproject/eddl.git 
cd eddl
git checkout 02e37c0dfb674468495f4d0ae3c159de3b2d3cc0 # Latest master, waiting for the release
mkdir -p build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DBUILD_TARGET=$DEVICE -DBUILD_SUPERBUILD=ON -DCMAKE_INSTALL_PREFIX=install ..
make -j$PROC && make install
EDDL_INSTALL_DIR=$UCP_PATH/$DEPENDENCIES_DIR/eddl/build/cmake

############ OPENCV
cd $UCP_PATH/$DEPENDENCIES_DIR
wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz -O $OPENCV_VERSION.tar.gz
tar -xf $OPENCV_VERSION.tar.gz
rm $OPENCV_VERSION.tar.gz
cd opencv-$OPENCV_VERSION
mkdir -p build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=install -DBUILD_LIST=core,imgproc,imgcodecs,photo -DBUILD_opencv_apps=OFF -DBUILD_opencv_java_bindings_generator=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python_tests=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF -DBUILD_JPEG=ON -DBUILD_IPP_IW=OFF -DBUILD_ITT=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_PNG=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_TIFF=ON -DBUILD_WEBP=OFF -DBUILD_ZLIB=OFF -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DWITH_EIGEN=OFF -DWITH_FFMPEG=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_JPEG=ON -DWITH_LAPACK=OFF -DWITH_MATLAB=OFF -DWITH_OPENCL=OFF -DWITH_OPENEXR=OFF -DWITH_OPENGL=OFF -DWITH_PNG=ON -DWITH_PROTOBUF=OFF -DWITH_QT=OFF -DWITH_TBB=OFF -DWITH_TIFF=ON -DWITH_V4L=OFF -DWITH_WEBP=OFF ..
make -j$PROC && make install
OPENCV_INSTALL_DIR=$UCP_PATH/$DEPENDENCIES_DIR/opencv/build

############ ECVL
cd $UCP_PATH/$DEPENDENCIES_DIR
git clone https://github.com/deephealthproject/ecvl.git
cd ecvl
git checkout c5326665e93ab4f34d143bd94de527f4fe643053 # Latest master, waiting for the release
mkdir -p build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DOpenCV_DIR=$OPENCV_INSTALL_DIR -Deddl_DIR=$EDDL_INSTALL_DIR -DECVL_BUILD_EDDL=ON -DECVL_DATASET=ON -DECVL_BUILD_GUI=OFF -DECVL_WITH_DICOM=ON -DCMAKE_INSTALL_PREFIX=install ..
make -j$PROC && make install
ECVL_INSTALL_DIR=$UCP_PATH/$DEPENDENCIES_DIR/ecvl/build/install

############ PIPELINE
cd $UCP_PATH
mkdir -p bin_lin && cd bin_lin
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -Decvl_DIR=$ECVL_INSTALL_DIR ..
make -j$PROC && ./MNIST_BATCH
