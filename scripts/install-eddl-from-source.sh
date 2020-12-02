#!/bin/bash

cd ${HOME}/temp

env_vars_definition_script="${HOME}/temp/use_case_pipeline/scripts/env-vars.sh"

if [ ! -f ${env_vars_definition_script} ]
then
	echo "File not found!  ${env_vars_definition_script}"
	exit 1
fi

. ${env_vars_definition_script}

if [ ! -d eddl ]
then
    git clone --recurse-submodule https://github.com/deephealthproject/eddl.git 
    cd eddl
    #git checkout 02e37c0dfb674468495f4d0ae3c159de3b2d3cc0 # Latest master, waiting for the release
    #git checkout tags/0.7.1
else
    cd eddl
fi

mkdir -p build && cd build

#cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DBUILD_TARGET=$DEVICE -DBUILD_SUPERBUILD=ON -DCMAKE_INSTALL_PREFIX=install ..
cmake -G "Unix Makefiles" \
	-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
	-DBUILD_TARGET=${DEVICE} \
	-DBUILD_SUPERBUILD=ON \
	-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} \
	-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
	-DBUILD_TESTS=ON \
	..

# BUILD_SUPERBUILD must be set to OFF in this case because we are working on conda, see previous script.

make -j${PROC} && make install
