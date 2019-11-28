set CUR_PATH=%cd:\=/%
mkdir deephealth
cd deephealth

:: EDDL
git clone git@github.com:deephealthproject/eddl.git --recursive
cd eddl
git checkout 9e11e447efb3184eaf8f747336eba9ce7fb86674
mkdir bin
cd bin
cmake -G "Visual Studio 15 2017 Win64" -DBUILD_TARGET=GPU -DEDDL_WITH_CUDA=ON -DCMAKE_INSTALL_PREFIX=install ..
cmake --build . --config Release --target ALL_BUILD
cmake --build . --config Release --target INSTALL

:: OPENCV
cd %CUR_PATH%/deephealth/
git clone git@github.com:opencv/opencv.git
cd opencv
git checkout tags/4.1.1
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=install -DBUILD_LIST=core,imgproc,imgcodecs -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_ZLIB=OFF -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DWITH_FFMPEG=OFF -DWITH_OPENCL=OFF -DWITH_QT=OFF -DWITH_IPP=OFF -DWITH_MATLAB=OFF -DWITH_OPENGL=OFF -DWITH_QT=OFF -DWITH_TIFF=OFF -DWITH_TBB=OFF -DWITH_V4L=OFF -DWITH_LAPACK=OFF -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF WITH_OPENEXR=OFF ..
cmake --build . --config Release --target ALL_BUILD
cmake --build . --config Release --target INSTALL

:: ECVL
cd %CUR_PATH%/deephealth
git clone git@github.com:deephealthproject/ecvl.git
cd ecvl
git checkout a88c883b94bcf3a5ea8126616aba2178514ffac2
mkdir bin
cd bin
cmake -G "Visual Studio 15 2017 Win64" -DOpenCV_DIR=%CUR_PATH%/deephealth/opencv/build -DEDDL_DIR=%CUR_PATH%/deephealth/eddl/bin/install -DECVL_BUILD_EDDL=ON -DECVL_DATASET_PARSER=ON -DECVL_BUILD_GUI=OFF -DCMAKE_INSTALL_PREFIX=install ..
cmake --build . --config Release --target ALL_BUILD
cmake --build . --config Release --target INSTALL

:: PIPELINE
cd %CUR_PATH%
mkdir bin
cd bin
cmake -G "Visual Studio 15 2017 Win64" -DECVL_DIR=deephealth/ecvl/bin/install ..
cmake --build . --config Release --target ALL_BUILD
Release\\MNIST_BATCH.exe
