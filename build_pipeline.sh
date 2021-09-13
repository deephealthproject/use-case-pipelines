#!/bin/bash

UCP_PATH=$(pwd)
DEVICE="GPU"
#DEVICE="CUDNN"
BUILD_TYPE=${1:-Release}
# BUILD_TYPE=${1:-Debug}
DEPENDENCIES_DIR="${2:-deephealth_lin}"
CMAKE_GENERATOR=${3:-"Unix Makefiles"}

EDDL_VERSION="${4:-v1.0.2a}"
ECVL_VERSION="${5:-v0.4.1}"
OPENCV_VERSION=3.4.13

PROC=$(nproc)
IS_CI=${6:-false}

set -e
set -o pipefail

#export CC=gcc-9
#export CXX=g++-9
#export CUDACXX=/usr/local/cuda-10.2/bin/nvcc

mkdir -p $DEPENDENCIES_DIR && cd $DEPENDENCIES_DIR

############ EDDL
if [ ! -d "eddl" ]; then
  git clone https://github.com/deephealthproject/eddl.git
  cd eddl
  git checkout tags/${EDDL_VERSION}
else
  cd eddl
fi
mkdir -p build && cd build
cmake -G"${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DBUILD_TARGET=$DEVICE -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_SUPERBUILD=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_HPC=OFF -DCMAKE_INSTALL_PREFIX=install ..
make -j$PROC && make install
EDDL_INSTALL_DIR=$(pwd)/install

############ OPENCV
cd $UCP_PATH && cd $DEPENDENCIES_DIR
if [ ! -d "opencv-$OPENCV_VERSION" ]; then
  wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz -O $OPENCV_VERSION.tar.gz
  tar -xzf $OPENCV_VERSION.tar.gz
  rm $OPENCV_VERSION.tar.gz
fi
cd opencv-$OPENCV_VERSION
mkdir -p build && cd build
cmake -G"${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=install -DBUILD_LIST=core,imgproc,imgcodecs,photo -DBUILD_opencv_apps=OFF -DBUILD_opencv_java_bindings_generator=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python_tests=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF -DBUILD_JPEG=ON -DBUILD_IPP_IW=OFF -DBUILD_ITT=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_PNG=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_TIFF=ON -DBUILD_WEBP=OFF -DBUILD_ZLIB=OFF -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DWITH_EIGEN=OFF -DWITH_FFMPEG=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_JPEG=ON -DWITH_LAPACK=OFF -DWITH_MATLAB=OFF -DWITH_OPENCL=OFF -DWITH_OPENEXR=OFF -DWITH_OPENGL=OFF -DWITH_PNG=ON -DWITH_PROTOBUF=OFF -DWITH_QT=OFF -DWITH_TBB=OFF -DWITH_TIFF=ON -DWITH_V4L=OFF -DWITH_WEBP=OFF ..
make -j$PROC && make install
OPENCV_BUILD_DIR=$(pwd)

############ ECVL
cd $UCP_PATH && cd $DEPENDENCIES_DIR
if [ ! -d "ecvl" ]; then
  git clone https://github.com/deephealthproject/ecvl.git
fi
cd ecvl
# Latest release
git checkout tags/${ECVL_VERSION}
mkdir -p build && cd build
cmake -G"${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DECVL_GPU=OFF -DOpenCV_DIR=$OPENCV_BUILD_DIR -Deddl_DIR=$EDDL_INSTALL_DIR/lib/cmake/eddl -DECVL_BUILD_EDDL=ON -DECVL_DATASET=ON -DECVL_BUILD_GUI=OFF -DECVL_WITH_DICOM=ON -DECVL_TESTS=OFF -DCMAKE_INSTALL_PREFIX=install ..
make -j$PROC && make install
ECVL_INSTALL_DIR=$(pwd)/install

if [ "$IS_CI" = false ] ; then
    ############ PIPELINE
    cd $UCP_PATH
    mkdir -p bin_lin && cd bin_lin
    cmake -G"${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -Decvl_DIR=$ECVL_INSTALL_DIR ..
    make -j$PROC
    echo "Pipeline built"
fi
cd $UCP_PATH