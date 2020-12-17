#!/bin/bash

cd ${HOME}/temp/use_case_pipeline/
mkdir -p build && cd build

#cmake -Decvl_DIR=/home/lxd/temp/deephealth/lib/cmake/ecvl ..
cmake   -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -Decvl_DIR=${HOME}/temp/deephealth/lib/cmake/ecvl ..


echo ""
echo ""
echo "   run   make -j"
echo ""
