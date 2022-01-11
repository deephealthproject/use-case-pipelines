#!/bin/bash

CUR_PATH="$(pwd)"

if [ "${CACHE}" != 'true' ]; then
    mkdir -p ${DEPS_PATH}
    mkdir -p ${DEPS_INSTALL_PATH}

    ############ EDDL DEPS
    cd ${DEPS_PATH}
    PB_VERSION=3.14.0
    wget -nv --no-check-certificate https://github.com/protocolbuffers/protobuf/releases/download/v${PB_VERSION}/protobuf-cpp-${PB_VERSION}.tar.gz
    tar -xzf protobuf-cpp-${PB_VERSION}.tar.gz
    rm protobuf-cpp-${PB_VERSION}.tar.gz
    mv protobuf-${PB_VERSION} protobuf
    cd protobuf
    mkdir -p build_dir && cd build_dir
    cmake -G"${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_PATH}/protobuf ../cmake
    cmake --build . --config $BUILD_TYPE --parallel $PROC
    cmake --build . --config $BUILD_TYPE --target install

    # Copy protobuf in /usr
    sudo cp -R ${DEPS_INSTALL_PATH}/protobuf/* /usr
    sudo ldconfig

    ############ EDDL
    cd ${DEPS_PATH}
    wget -nv --no-check-certificate https://github.com/deephealthproject/eddl/archive/$EDDL_VERSION.tar.gz
    mkdir -p eddl-$EDDL_VERSION && tar -xzf $EDDL_VERSION.tar.gz -C eddl-$EDDL_VERSION --strip-components 1
    rm $EDDL_VERSION.tar.gz
    cd eddl-${EDDL_VERSION}
    mkdir -p build && cd build
    cmake -G"${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DBUILD_TARGET=CPU -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_SUPERBUILD=OFF -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_PATH}/eddl-${EDDL_VERSION} ..
    cmake --build . --config $BUILD_TYPE --parallel $PROC
    cmake --build . --config $BUILD_TYPE --target install

    ############ OPENCV
    # cd ${DEPS_PATH}
    # if [ ! -d "opencv-$OPENCV_VERSION" ]; then
    #     wget -nv --no-check-certificate https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz -O $OPENCV_VERSION.tar.gz
    #     tar -xf $OPENCV_VERSION.tar.gz
    #     rm $OPENCV_VERSION.tar.gz
    # fi
    # cd opencv-$OPENCV_VERSION
    # mkdir -p build && cd build
    # cmake -G"${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_PATH}/opencv -DBUILD_LIST=core,imgproc,imgcodecs,photo -DBUILD_opencv_apps=OFF -DBUILD_opencv_java_bindings_generator=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python_tests=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF -DBUILD_JPEG=ON -DBUILD_IPP_IW=OFF -DBUILD_ITT=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_PNG=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_TIFF=ON -DBUILD_WEBP=OFF -DBUILD_ZLIB=OFF -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DWITH_EIGEN=OFF -DWITH_FFMPEG=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_JPEG=ON -DWITH_LAPACK=OFF -DWITH_MATLAB=OFF -DWITH_OPENCL=OFF -DWITH_OPENEXR=OFF -DWITH_OPENGL=OFF -DWITH_PNG=ON -DWITH_PROTOBUF=OFF -DWITH_QT=OFF -DWITH_TBB=OFF -DWITH_TIFF=ON -DWITH_V4L=OFF -DWITH_WEBP=OFF ..
    # cmake --build . --config $BUILD_TYPE --parallel $PROC
    # cmake --build . --config $BUILD_TYPE --target install

    ############ ECVL
    cd ${DEPS_PATH}
    git clone --recurse-submodules https://github.com/deephealthproject/ecvl.git
    cd ecvl
    git checkout tags/${ECVL_VERSION}
    mkdir -p build && cd build
    cmake -G"${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_PATH}/ecvl -DOpenCV_DIR=${DEPS_INSTALL_PATH}/opencv -Deddl_DIR=${DEPS_INSTALL_PATH}/eddl-${EDDL_VERSION}/lib/cmake/eddl -DECVL_BUILD_EDDL=ON -DECVL_DATASET=ON -DECVL_BUILD_GUI=OFF -DECVL_WITH_DICOM=ON -DECVL_TESTS=OFF ..
    cmake --build . --config $BUILD_TYPE --parallel $PROC
    cmake --build . --config $BUILD_TYPE --target install
fi

# Copy protobuf in /usr
sudo cp -R ${DEPS_INSTALL_PATH}/protobuf/* /usr
sudo ldconfig

cd ${CUR_PATH}