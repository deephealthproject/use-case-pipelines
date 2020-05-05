@echo off
set UCP_PATH=%cd:\=/%
set DEVICE=CPU
set BUILD_TYPE=Release
REM set BUILD_TYPE=Debug
set DEP_DIR=deephealth_win

mkdir %DEP_DIR% & cd %DEP_DIR%

REM EDDL
git clone --recurse-submodule https://github.com/deephealthproject/eddl.git
cd eddl
git checkout tags/0.4.3
mkdir build & cd build
cmake -A x64 -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DBUILD_TARGET=%DEVICE% -DBUILD_SHARED_LIB=OFF -DBUILD_PROTOBUF=OFF -DCMAKE_INSTALL_PREFIX=install ..
cmake --build . --config %BUILD_TYPE% --parallel 4
cmake --build . --config %BUILD_TYPE% --target INSTALL
set EDDL_INSTALL_DIR=%UCP_PATH%/%DEP_DIR%/eddl/build/cmake

REM OPENCV
cd %UCP_PATH%\%DEP_DIR%
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout tags/4.3.0
mkdir build & cd build
cmake -A x64 -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX=install -DBUILD_LIST=core,imgproc,imgcodecs,photo -DBUILD_opencv_apps=OFF -DBUILD_opencv_java_bindings_generator=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python_tests=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF -DBUILD_JPEG=ON -DBUILD_IPP_IW=OFF -DBUILD_ITT=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_PNG=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_TIFF=ON -DBUILD_WEBP=OFF -DBUILD_ZLIB=OFF -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DWITH_FFMPEG=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_JPEG=ON -DWITH_LAPACK=OFF -DWITH_MATLAB=OFF -DWITH_OPENCL=OFF -DWITH_OPENEXR=OFF -DWITH_OPENGL=OFF -DWITH_PNG=ON -DWITH_PROTOBUF=OFF -DWITH_QT=OFF -DWITH_TBB=OFF -DWITH_TIFF=ON -DWITH_V4L=OFF -DWITH_WEBP=OFF ..
cmake --build . --config %BUILD_TYPE% --parallel 4
cmake --build . --config %BUILD_TYPE% --target INSTALL
set OPENCV_INSTALL_DIR=%UCP_PATH%/%DEP_DIR%/opencv/build

REM ECVL
cd %UCP_PATH%\%DEP_DIR%
git clone https://github.com/deephealthproject/ecvl.git
cd ecvl
git checkout tags/v0.2.1
mkdir build & cd build
cmake -A x64 -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DOpenCV_DIR=%OPENCV_INSTALL_DIR% -Deddl_DIR=%EDDL_INSTALL_DIR% -DECVL_BUILD_EDDL=ON -DECVL_DATASET=ON -DECVL_BUILD_GUI=OFF -DCMAKE_INSTALL_PREFIX=install ..
cmake --build . --config %BUILD_TYPE% --parallel 4
cmake --build . --config %BUILD_TYPE% --target INSTALL
set ECVL_INSTALL_DIR=%UCP_PATH%/%DEP_DIR%/ecvl/build/install

REM PIPELINE
cd %UCP_PATH%
mkdir bin_win & cd bin_win
cmake -A x64 -Decvl_DIR=%ECVL_INSTALL_DIR% ..
cmake --build . --config %BUILD_TYPE% --target ALL_BUILD
%BUILD_TYPE%\\MNIST_BATCH.exe
