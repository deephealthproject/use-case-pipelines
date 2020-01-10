#!/bin/bash

CUR_PATH=`pwd`
mkdir -p deephealth && cd deephealth

############ EDDL
git clone git@github.com:deephealthproject/eddl.git --recursive
cd eddl
git checkout 2406875be5c8fcf209138f57f90fc946110dc24c
mkdir -p bin && cd bin
cmake -G "Unix Makefiles" -DBUILD_TARGET=GPU -DEDDL_WITH_CUDA=ON -DCMAKE_INSTALL_PREFIX=install ..
make -j$(nproc) && make install

############ OPENCV
cd $CUR_PATH/deephealth
git clone git@github.com:opencv/opencv.git
cd opencv
git checkout tags/4.1.1
mkdir -p build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=install -DBUILD_LIST=core,imgproc,imgcodecs -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_ZLIB=OFF -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DWITH_FFMPEG=OFF -DWITH_OPENCL=OFF -DWITH_QT=OFF -DWITH_IPP=OFF -DWITH_MATLAB=OFF -DWITH_OPENGL=OFF -DWITH_QT=OFF -DWITH_TIFF=OFF -DWITH_TBB=OFF -DWITH_V4L=OFF -DWITH_LAPACK=OFF -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF WITH_OPENEXR=OFF ..
make -j$(nproc) && make install

############ ECVL
cd $CUR_PATH/deephealth
git clone git@github.com:deephealthproject/ecvl.git
cd ecvl
git checkout a3d8a31075ff529623df9c7b78f06c924b03ac4c  # Development branch
mkdir -p bin && cd bin
cmake -G "Unix Makefiles" -DOpenCV_DIR=$CUR_PATH/deephealth/opencv/build -Deddl_DIR=$CUR_PATH/deephealth/eddl/bin/install -DECVL_BUILD_EDDL=ON -DECVL_DATASET_PARSER=ON -DECVL_BUILD_GUI=OFF -DCMAKE_INSTALL_PREFIX=install ..
make -j$(nproc) && make install

############ PIPELINE
cd $CUR_PATH
mkdir -p bin && cd bin
cmake -G "Unix Makefiles" -Decvl_DIR=$CUR_PATH/deephealth/ecvl/bin/install ..
make -j$(nproc) && ./MNIST_BATCH
