set CUR_PATH=%cd:\=/%
mkdir deephealth
cd deephealth

:: EDDL
git clone https://github.com/deephealthproject/eddl.git --recursive
cd eddl
git checkout tags/v0.4.2
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" -DBUILD_TARGET=GPU -DBUILD_SHARED_LIB=OFF -DUSE_CUDA=ON -DCMAKE_INSTALL_PREFIX=install ..
cmake --build . --config Release --target ALL_BUILD
cmake --build . --config Release --target INSTALL

:: OPENCV
cd %CUR_PATH%/deephealth/
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout tags/4.1.1
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=install -DBUILD_LIST=core,imgproc,imgcodecs,photo -DBUILD_opencv_apps=OFF -DBUILD_opencv_java_bindings_generator=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python_tests=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_ZLIB=OFF -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DWITH_FFMPEG=OFF -DWITH_OPENCL=OFF -DWITH_QT=OFF -DWITH_IPP=OFF -DWITH_MATLAB=OFF -DWITH_OPENGL=OFF -DWITH_QT=OFF -DWITH_TIFF=ON -DWITH_TBB=OFF -DWITH_V4L=OFF -DWITH_LAPACK=OFF -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF WITH_OPENEXR=OFF ..
cmake --build . --config Release --target ALL_BUILD
cmake --build . --config Release --target INSTALL

:: ECVL
cd %CUR_PATH%/deephealth
git clone https://github.com/deephealthproject/ecvl.git
cd ecvl
git checkout ed0bc3dc90a5d93217f0d8d72e2d21b3aecc4925 
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" -DOpenCV_DIR=%CUR_PATH%/deephealth/opencv/build -Deddl_DIR=%CUR_PATH%/deephealth/eddl/build/install -DECVL_BUILD_EDDL=ON -DECVL_DATASET_PARSER=ON -DECVL_BUILD_GUI=OFF -DCMAKE_INSTALL_PREFIX=install ..
cmake --build . --config Release --target ALL_BUILD
cmake --build . --config Release --target INSTALL

:: PIPELINE
cd %CUR_PATH%
mkdir bin
cd bin
cmake -G "Visual Studio 15 2017 Win64" -Decvl_DIR=deephealth/ecvl/build/install ..
cmake --build . --config Release --target ALL_BUILD
Release\\MNIST_BATCH.exe
