#!/bin/bash

cd ${HOME}/temp

env_vars_definition_script="${HOME}/temp/use_case_pipeline/scripts/env-vars.sh"

if [ ! -f ${env_vars_definition_script} ]
then
	echo "File not found!  ${env_vars_definition_script}"
	exit 1
fi

. ${env_vars_definition_script}


############ ECVL
if [ ! -d ecvl ]
then
	git clone https://github.com/deephealthproject/ecvl.git
	cd ecvl
	#git checkout c5326665e93ab4f34d143bd94de527f4fe643053 # Latest master, waiting for the release
	#git checkout tags/v0.2.3
else
	cd ecvl
fi


mkdir -p build && cd build

cmake -G "Unix Makefiles" \
	-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
	-DOpenCV_DIR=${OPENCV_INSTALL_DIR} \
	-Deddl_DIR=${EDDL_INSTALL_DIR} \
	-DECVL_BUILD_EDDL=ON \
	-DECVL_DATASET=ON \
	-DECVL_BUILD_GUI=OFF \
	-DECVL_WITH_DICOM=ON \
	-DCMAKE_INSTALL_PREFIX=${ECVL_INSTALL_DIR} \
	..

#	-DCMAKE_INSTALL_PREFIX=${ECVL_INSTALL_DIR} \
#	-DCMAKE_INSTALL_PREFIX=${HOME}/temp/install ..

make -j${PROC} && make install

