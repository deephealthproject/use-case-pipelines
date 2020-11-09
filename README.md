# 3rd DeepHealth Hackathon

The 3rd DeepHealth Hackathon concern segmentation of multiple sclerosis lesion volumes. For this task a baseline Pipeline that uses EDDL and ECVL has been prepared.

## Requirements
- CMake 3.13 or later
- C++ Compiler with C++17 support (e.g. GCC 6 or later, Clang 5.0 or later)
- MSSEG Dataset


## Multiple Sclerosis Lesion Segmentation Dataset
Dataset of the MSSEG challenge which took place during MICCAI 2016 (https://portal.fli-iam.irisa.fr/msseg-challenge).
1. Subscribe the challenge in order to dowload data [here](https://portal.fli-iam.irisa.fr/msseg-challenge/overview?p_p_id=registration_WAR_fliiamportlet&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&p_p_col_id=column-1&p_p_col_pos=1&p_p_col_count=3&_registration_WAR_fliiamportlet_mvcPath=%2Fhtml%2Fregistration%2Fregistration.jsp).
    1. Create the MSSEG directory (`mkdir MSSEG`)
    1. Download the `Unprocessed training dataset` from [here](https://portal.fli-iam.irisa.fr/documents/20182/22089/Unprocessed+training+dataset/d487c807-8fd2-4fa0-892e-53d4578f343a?version=1.2) and place the zip inside `MSSEG`.
    1. Download the `Pre-processed training dataset` from [here](https://portal.fli-iam.irisa.fr/documents/20182/22089/Pre-processed+training+dataset/6f1a997b-8d32-4791-b124-a35ee4fc4655?version=1.2) and place the zip inside `MSSEG`.
1. Download the script at [https://github.com/deephealthproject/use_case_pipeline/blob/3rd_hackathon/dataset/extract_data.sh](https://github.com/deephealthproject/use_case_pipeline/blob/3rd_hackathon/dataset/extract_data.sh), save it in `MSSEG` folder, and run it.
    ```shell
    cd ~
    mkdir MSSEG && cd MSSEG
    wget https://github.com/deephealthproject/use_case_pipeline/blob/3rd_hackathon/dataset/extract_data.sh
    chmod +x extract_data.sh
    ./extract_data.sh
    ```
1. Place the `ms_segmentation.yaml` and put it inside `MSSEG` directory.
    ```shell
    wget https://github.com/deephealthproject/use_case_pipeline/blob/3rd_hackathon/dataset/ms_segmentation.yml
    ```


<!-- #### PNEUMOTHORAX
Dataset taken from a kaggle challenge (more details [here](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)).
  1. Download training and test images [here](https://www.kaggle.com/seesee/siim-train-test/download).
  1. Download from [here](https://drive.google.com/uc?id=1e9f0LzPB8euHRJLA5URknUFZHD-8AtE9&export=download) ground truth masks and the YAML dataset file.
  1. In order to copy the ground truth masks in the directory of the corresponding images, edit the `cpp/copy_ground_truth_pneumothorax.cpp` file with the path to the downloaded dataset and ground truth directory and run it. Move the YAML file in the `siim` dataset folder.
  
  Short [video](https://drive.google.com/uc?id=17qlmm9Jf_D3K4iB3Y9pfrpDssFxk2q69&export=download) in which these steps are shown.
  
From the 2669 distinct training images with mask, 200 are randomly sampled as validation set.
- Training set: 3086 total images - 80% with mask and 20% without mask.
- Validation set: 250 total images - 80% with mask and 20% without mask.

To perform only inference on test set, change the dataset path into the `pneumothorax_segmentation_inference.cpp` source file and download checkpoint [here](https://drive.google.com/uc?id=13-bSsMHxKp7WO_HrdWcy5y9n9hbOXNyT&export=download) for EDDL versions >= 0.4.3 or [here](https://drive.google.com/uc?id=1kLhNpzBi5OYm9y4YNlK_XuUf52WItUVT&export=download) for EDDL versions <= 0.4.2 (best Dice Coefficient on validation in 50 epochs). -->


### CUDA
On Linux systems, starting from CUDA 10.1, cuBLAS libraries are installed in the `/usr/lib/<arch>-linux-gnu/` or `/usr/lib64/`. Create a symlink to resolve the issue:
```bash
sudo ln -s /usr/lib/<arch>-linux-gnu/libcublas.so /usr/local/cuda-10.1/lib64/libcublas.so
```

## Building

- **\*nix**
    - Building from scratch, assuming CUDA driver already installed if you want to use GPUs ([video](https://drive.google.com/uc?id=1xGPHIEXK-vzxEF0y8N148EhFud1Ackm4&export=download) in which these steps are performed in a clean nvidia docker image):
        ```bash
        sudo apt update
        sudo apt install wget git make gcc g++

        # cmake version >= 3.13 is required for ECVL
        wget https://cmake.org/files/v3.13/cmake-3.13.5-Linux-x86_64.tar.gz
        tar -xf cmake-3.13.5-Linux-x86_64.tar.gz

        # symbolic link for cmake
        sudo ln -s /<path/to>/cmake-3.13.5-Linux-x86_64/bin/cmake /usr/bin/cmake
        # symbolic link for cublas if we have cuda >= 10.1
        sudo ln -s /usr/lib/<arch>-linux-gnu/libcublas.so /usr/local/cuda-10.1/lib64/libcublas.so

        git clone https://github.com/deephealthproject/use_case_pipeline.git
        cd use_case_pipeline

        # install dependencies as sudo so that they will be installed in "standard" system directories
        chmod +x install_dependencies.sh
        sudo ./install_dependencies.sh

        # install EDDL, OpenCV, ECVL and build the pipeline
        chmod +x build_pipeline.sh
        ./build_pipeline.sh
        ```

    - Building with all the dependencies already installed:
        ```bash
        git clone https://github.com/deephealthproject/use_case_pipeline.git
        cd use_case_pipeline
        mkdir build && cd build

        # if ECVL is not installed in a "standard" system directory (like /usr/local/) you have to provide the installation directory
        cmake -Decvl_DIR=/<path/to>/ecvl/build/install ..
        make
        ```

    
**N.B.** EDDL is built for GPU by default.
    
## Training and inference

- The project creates different executables: MNIST_BATCH, SKIN_LESION_CLASSIFICATION_TRAINING, SKIN_LESION_SEGMENTATION_TRAINING, SKIN_LESION_CLASSIFICATION_INFERENCE, SKIN_LESION_SEGMENTATION_INFERENCE, PNEUMOTHORAX_SEGMENTATION_TRAINING and PNEUMOTHORAX_SEGMENTATION_INFERENCE, which are better described in [master branch](https://github.com/deephealthproject/use_case_pipeline).

- MS_SEGMENTATION_TRAINING trains the neural network loading the dataset (volumes and their ground truth masks) in batches with a custom function for this specific segmentation task. Each volume is loaded in memory and then some slices (specified by `n_channels_` variable) are extracted and used as input for the neural network.
