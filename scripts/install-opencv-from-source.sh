#!/bin/bash

cd ${HOME}/temp

env_vars_definition_script="${HOME}/temp/use_case_pipeline/scripts/env-vars.sh"

if [ ! -f ${env_vars_definition_script} ]
then
	echo "File not found!  ${env_vars_definition_script}"
	exit 1
fi

. ${env_vars_definition_script}


############ OPENCV
if [ ! -d opencv-${OPENCV_VERSION} ]
then
    wget -c https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz -O ${OPENCV_VERSION}.tar.gz
    tar -xf ${OPENCV_VERSION}.tar.gz
fi

cd opencv-${OPENCV_VERSION}
mkdir -p build && cd build

cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DBUILD_LIST=core,imgproc,imgcodecs,photo,calib3d \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_java_bindings_generator=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_python_bindings_generator=OFF \
        -DBUILD_opencv_python_tests=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_JPEG=ON \
        -DBUILD_IPP_IW=OFF \
        -DBUILD_ITT=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_PNG=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_TIFF=ON \
        -DBUILD_WEBP=OFF \
        -DBUILD_ZLIB=OFF \
        -DINSTALL_C_EXAMPLES=OFF \
        -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DWITH_EIGEN=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_IPP=OFF \
        -DWITH_ITT=OFF \
        -DWITH_JPEG=ON \
        -DWITH_LAPACK=OFF \
        -DWITH_MATLAB=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENEXR=OFF \
        -DWITH_OPENGL=OFF \
        -DWITH_PNG=ON \
        -DWITH_PROTOBUF=OFF \
        -DWITH_QT=OFF \
        -DWITH_TBB=OFF \
        -DWITH_TIFF=ON \
        -DWITH_V4L=OFF \
        -DWITH_WEBP=OFF \
        ..

make -j$(nproc) && make install

