# Scripts for installing and compiling all the components for completing the pipeline

The following described steps asume the pipeline repository was already downloaded
in  `${HOME}/temp/use_case_pipeline`


## Installation example 

The following scripts assume the ***cudatoolkit*** with the corresponding libraries is already installed in your system.

### Shell variable definitions

Please, update the script [env-vars.sh](env-vars.sh) according to your configuration preferences.

### Sequence of scripts to complete the installation in a Unix-like system

1. [system_dependencies.sh](system_dependencies.sh)
   Run this script if your Unix-like system does not have installed what is required.

2. [install-eddl-from-source.sh](install-eddl-from-source.sh)
   Run this script from `${HOME}/temp` as `use_case_pipeline/scripts/install-eddl-from-source.sh`

3. [install-opencv-from-source.sh](install-opencv-from-source.sh)
   Run this script from `${HOME}/temp` as `use_case_pipeline/scripts/install-opencv-from-source.sh`

4. [install-ecvl-from-source.sh](install-ecvl-from-source.sh)
   Run this script from `${HOME}/temp` as `use_case_pipeline/scripts/install-ecvl-from-source.sh`

5. [cmake-pipeline.sh](cmake-pipeline.sh)
   This script changes the working directory to `${HOME}/temp/use_case_pipeline`


## Example of launching scripts

Please, modify these scripts to adapt them to your configuration:
   
1. [launch_training.sh](launch_training.sh)

2. [launch_testing.sh](launch_testing.sh)
