#!/bin/bash

CUR_PATH=`pwd`
mkdir -p deephealth && cd deephealth

############ EDDL
git clone --recurse-submodule git@github.com:deephealthproject/eddl.git 
cd eddl
git checkout 6fdef431af870caba07cf3276c78e08828341f48
mkdir -p build && cd build
cmake -G "Unix Makefiles" -DBUILD_TARGET=GPU -DEDDL_WITH_CUDA=ON -DCMAKE_INSTALL_PREFIX=install ..
make -j$(nproc) && make install

############ OPENCV
cd $CUR_PATH/deephealth
git clone git@github.com:opencv/opencv.git
cd opencv
git checkout tags/4.1.1
mkdir -p build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=install -DBUILD_LIST=core,imgproc,imgcodecs -DBUILD_opencv_apps=OFF -DBUILD_opencv_java_bindings_generator=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python_tests=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_ZLIB=OFF -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DWITH_FFMPEG=OFF -DWITH_OPENCL=OFF -DWITH_QT=OFF -DWITH_IPP=OFF -DWITH_MATLAB=OFF -DWITH_OPENGL=OFF -DWITH_QT=OFF -DWITH_TIFF=ON -DWITH_TBB=OFF -DWITH_V4L=OFF -DWITH_LAPACK=OFF -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF WITH_OPENEXR=OFF ..
make -j$(nproc) && make install

############ ECVL
cd $CUR_PATH/deephealth
git clone git@github.com:deephealthproject/ecvl.git
cd ecvl
git checkout 983119668e8c0a2f45c29b4d586c475dc901aba3  # Master branch
mkdir -p build && cd build
cmake -G "Unix Makefiles" -DOpenCV_DIR=$CUR_PATH/deephealth/opencv/build -Deddl_DIR=$CUR_PATH/deephealth/eddl/build/cmake -DECVL_BUILD_EDDL=ON -DECVL_DATASET_PARSER=ON -DECVL_BUILD_GUI=OFF -DCMAKE_INSTALL_PREFIX=install ..
make -j$(nproc) && make install

############ PIPELINE
cd $CUR_PATH
mkdir -p bin && cd bin
cmake -G "Unix Makefiles" -Decvl_DIR=$CUR_PATH/deephealth/ecvl/build/install ..
make -j$(nproc) && ./MNIST_BATCH
