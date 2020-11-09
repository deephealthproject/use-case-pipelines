#!/bin/bash

UCP_PATH=$(pwd)
DEPENDENCIES_DIR="eddl_dependencies"

mkdir -p $DEPENDENCIES_DIR && cd $DEPENDENCIES_DIR

############ EDDL dependencies

git clone https://gitlab.com/libeigen/eigen.git
cd eigen
git checkout tags/3.3.7
mkdir -p build && cd build
cmake ..
make install

cd $UCP_PATH/$DEPENDENCIES_DIR
wget -q https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protobuf-cpp-3.11.4.tar.gz
tar xf protobuf-cpp-3.11.4.tar.gz
cd protobuf-3.11.4
./configure
make -j$(nproc)
make install
ldconfig

cd $UCP_PATH
rm -rf $DEPENDENCIES_DIR
