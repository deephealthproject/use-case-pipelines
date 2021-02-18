@echo off
set UCP_PATH=%cd:\=/%
set GENERATOR=Visual Studio 16 2019
set DEVICE=GPU
set BUILD_TYPE=Release
REM set BUILD_TYPE=Debug
set DEPENDENCIES_DIR=deephealth_win
set OPENCV_VERSION=4.5.1
set /a PROC=%NUMBER_OF_PROCESSORS%-1

echo The default CMAKE_GENERATOR is "%GENERATOR%".
set changeGen=n
set /p changeGen=Do you want to change it [y/n] (default: %changeGen%)?:

if "%changeGen%"=="y" (
    set /p GENERATOR=Type name of new generator (example: Visual Studio 15 2017)
)
echo Building Pipeline for CMAKE_GENERATOR: "%GENERATOR%".

mkdir %DEPENDENCIES_DIR% & cd %DEPENDENCIES_DIR%

REM EDDL
git clone https://github.com/deephealthproject/eddl.git
cd eddl
REM Latest release
git checkout tags/v0.9a
mkdir build & cd build
cmake -G "%GENERATOR%" -A x64 -DBUILD_TARGET=%DEVICE% -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_SUPERBUILD=ON -DBUILD_HPC=OFF -DCMAKE_INSTALL_PREFIX=install ..
cmake --build . --config %BUILD_TYPE% --parallel %PROC% --target INSTALL
set EDDL_INSTALL_DIR=%UCP_PATH%/%DEPENDENCIES_DIR%/eddl/build/install

REM OPENCV
cd %UCP_PATH%\%DEPENDENCIES_DIR%
IF NOT EXIST opencv-%OPENCV_VERSION%\ (
    curl https://github.com/opencv/opencv/archive/%OPENCV_VERSION%.tar.gz -L -o %OPENCV_VERSION%.tar.gz
    tar -xf %OPENCV_VERSION%.tar.gz
    del %OPENCV_VERSION%.tar.gz
)
cd opencv-%OPENCV_VERSION%
mkdir build & cd build
cmake -G "%GENERATOR%" -A x64 -DCMAKE_INSTALL_PREFIX=install -DBUILD_LIST=core,imgproc,imgcodecs,photo,calib3d -DBUILD_opencv_apps=OFF -DBUILD_opencv_java_bindings_generator=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python_tests=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF -DBUILD_JPEG=ON -DBUILD_IPP_IW=OFF -DBUILD_ITT=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_PNG=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_TIFF=ON -DBUILD_WEBP=OFF -DBUILD_ZLIB=ON -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DWITH_EIGEN=OFF -DWITH_FFMPEG=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_JPEG=ON -DWITH_LAPACK=OFF -DWITH_MATLAB=OFF -DWITH_OPENCL=OFF -DWITH_OPENEXR=OFF -DWITH_OPENGL=OFF -DWITH_PNG=ON -DWITH_PROTOBUF=OFF -DWITH_QT=OFF -DWITH_TBB=OFF -DWITH_TIFF=ON -DWITH_V4L=OFF -DWITH_WEBP=OFF ..
cmake --build . --config %BUILD_TYPE% --parallel %PROC% --target INSTALL
set OPENCV_INSTALL_DIR=%UCP_PATH%/%DEPENDENCIES_DIR%/opencv-%OPENCV_VERSION%/build

REM ECVL
cd %UCP_PATH%\%DEPENDENCIES_DIR%
git clone https://github.com/deephealthproject/ecvl.git
cd ecvl
git checkout tags/v0.3.1 REM Latest release
mkdir build & cd build
cmake -G "%GENERATOR%" -A x64 -DOpenCV_DIR=%OPENCV_INSTALL_DIR% -Deddl_DIR=%EDDL_INSTALL_DIR%/lib/cmake/eddl -DECVL_BUILD_EDDL=ON -DECVL_DATASET=ON -DECVL_BUILD_GUI=OFF -DECVL_WITH_DICOM=ON -DECVL_GPU=OFF -DECVL_TESTS=OFF -DCMAKE_INSTALL_PREFIX=install ..
cmake --build . --config %BUILD_TYPE% --parallel %PROC% --target INSTALL
set ECVL_INSTALL_DIR=%UCP_PATH%/%DEPENDENCIES_DIR%/ecvl/build/install

REM PIPELINE
cd %UCP_PATH%
mkdir bin_win & cd bin_win
cmake -G "%GENERATOR%" -A x64 -Decvl_DIR=%ECVL_INSTALL_DIR% ..
cmake --build . --config %BUILD_TYPE% --parallel %PROC% --target ALL_BUILD
echo Pipeline built
