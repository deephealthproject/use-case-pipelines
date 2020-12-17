#
# commented because currently the conda environment is not used
#
#if [ "${CONDA_DEFAULT_ENV:-XX}" != "eddl" ]
#then
#    echo ""
#    echo "FATAL ERROR: you are not in the 'eddl' conda environment!"
#    echo ""
#    exit 1
#fi


#export CC=/usr/bin/gcc-8
#export CXX=/usr/bin/g++-8

export DEVICE="GPU"
#export DEVICE="CPU"
#export BUILD_TYPE="Debug"
export BUILD_TYPE="Release"


export WORKING_PREFIX="${HOME}/temp"
export INSTALL_PREFIX="${WORKING_PREFIX}/deephealth"
export CONDA_PREFIX="${WORKING_PREFIX}/deephealth" # because the scripts using these definitions use it
export EDDL_INSTALL_DIR="${INSTALL_PREFIX}/lib/cmake/eddl"
export EDDL_DIR="${INSTALL_PREFIX}/lib/cmake/eddl/"

export OPENCV_VERSION=4.5.0
#export OPENCV_INSTALL_DIR=${WORKING_PREFIX}/opencv-${OPENCV_VERSION}/build
export OPENCV_INSTALL_DIR="${INSTALL_PREFIX}/lib/cmake/opencv4"

export ECVL_INSTALL_DIR="${INSTALL_PREFIX}/lib/cmake/ecvl"

export PROC=$(($(nproc)-1))
