# DH USE CASE Pipeline 

Pipeline that uses EDDL and ECVL to train a CNN on three different datasets (_MNIST_, _ISIC_ and _PNEUMOTHORAX_), applying different image augmentations, for both the classification and the segmentation task.

## Requirements
- _C++-17_ compiler (gcc-8 or Visual Studio 2017);
- CMake;
- (Optional) ISIC dataset.
- (Optional) Pneumothorax dataset.

### Datasets
The YAML datasets format is described [here](https://github.com/deephealthproject/ecvl/wiki/DeepHealth-Toolkit-Dataset-Format). Each dataset listed below contains both the data and the YAML description format, but they can also be downloaded separately: [ISIC classification](https://drive.google.com/uc?id=1pZotvwM5rltg5OhYGr9oSLVW8yxjs3U_&export=download), [ISIC segmentation](https://drive.google.com/uc?id=1HHmBNiyQ1dH398E3ECl8WqVoZuV23fjM&export=download) and [Pneumothorax segmentation](https://drive.google.com/uc?id=1D1IM9Gw2wWzvnWeX7ac7ZsjetxU8kCit&export=download).


#### MNIST
Automatically downloaded and extracted by CMake.

#### ISIC - [isic-archive.com](https://www.isic-archive.com/#!/topWithHeader/tightContentTop/challenges)

_Classification_: Download it from [here](https://drive.google.com/uc?id=1TCE-uswZ41nlqMe5SWHoCGF7Mtq6r15A&export=download) and extract it. Change the dataset path into the `skin_lesion_classification_training.cpp` source file accordingly. To perform only inference, change the dataset path into the `skin_lesion_classification_inference.cpp` source file and download checkpoints [here](https://drive.google.com/file/d/1HzEtAni3WNmpwBBBT6fZ5hkW9wJAgqF2/view?usp=sharing) (best accuracy on validation in 50 epochs).

 _Segmentation_: Download it from [here](https://drive.google.com/uc?id=1RyYa32x9aqwd2kkQpCZ4Xa2h_VcgH3wI&export=download) and extract it. Change the dataset path into the `skin_lesion_segmentation_training.cpp` source file accordingly. To perform only inference, change the dataset path into the `skin_lesion_segmentation_inference.cpp` source file and download checkpoints [here](https://drive.google.com/file/d/13lbpkjrHNZygbdkdux8yr7GlpTW0MgxM/view?usp=sharing) (best Mean Intersection over Union on validation in 50 epochs).

#### PNEUMOTHORAX
Dataset taken from a kaggle challenge (more details [here](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)).
  1. Download training and test images [here](https://www.kaggle.com/seesee/siim-train-test/download).
  1. Download from [here](https://drive.google.com/uc?id=1e9f0LzPB8euHRJLA5URknUFZHD-8AtE9&export=download) ground truth masks and the YAML dataset file.
  1. In order to copy the ground truth masks in the directory of the corresponding images, edit the `src/copy_ground_truth_pneumothorax.cpp` file with the path to the downloaded dataset and ground truth directory and run it. Move the YAML file in the `siim` dataset folder.
  
From the 2669 distinct training images with mask, 200 are randomly sampled as validation set.
- Training set: 3086 total images - 80% with mask and 20% without mask.
- Validation set: 250 total images - 80% with mask and 20% without mask.

To perform only inference on test set, change the dataset path into the `pneumothorax_segmentation_inference.cpp` source file and download checkpoint [here](https://drive.google.com/uc?id=13-bSsMHxKp7WO_HrdWcy5y9n9hbOXNyT&export=download) for EDDL versions >= 0.4.3 or [here](https://drive.google.com/uc?id=1kLhNpzBi5OYm9y4YNlK_XuUf52WItUVT&export=download) for EDDL versions <= 0.4.2 (best Dice Coefficient on validation in 50 epochs).


### CUDA
On Linux systems, starting from CUDA 10.1, cuBLAS libraries are installed in the `/usr/lib/<arch>-linux-gnu/` or `/usr/lib64/`. Create a symlink to resolve the issue:
```bash
sudo ln -s /usr/lib/<arch>-linux-gnu/libcublas.so /usr/local/cuda-10.1/lib64/libcublas.so
```

## Building

- **\*nix**
    - Building from scratch, assuming CUDA driver already installed if you want to use GPUs:
        ```bash
        sudo apt update
        sudo apt install wget git make gcc-8 g++-8

        # cmake version >= 3.13 is required for ECVL
        wget https://cmake.org/files/v3.13/cmake-3.13.5-Linux-x86_64.tar.gz
        tar -xf cmake-3.13.5-Linux-x86_64.tar.gz

        # symbolic link for cmake
        sudo ln -s /<path/to>/cmake-3.13.5-Linux-x86_64/bin/cmake /usr/bin/cmake
        # symbolic link for cublas if we have cuda >= 10.1
        sudo ln -s /usr/lib/<arch>-linux-gnu/libcublas.so /usr/local/cuda-10.1/lib64/libcublas.so

        # if other versions of gcc (e.g., gcc-7) are present, set a higher priority to gcc-8 so that it is chosen as the default
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 --slave /usr/bin/g++ g++ /usr/bin/g++-8
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 --slave /usr/bin/g++ g++ /usr/bin/g++-7

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
    
- **Windows**
    - Building assuming `cmake >= 3.13`, `wget`, `git`, Visual Studio 2017 or 2019, CUDA driver (if you want to use GPUs) already installed 
        ```bash
        # install EDDL and all its dependencies, OpenCV, ECVL and build the pipeline
        git clone https://github.com/deephealthproject/use_case_pipeline.git
        cd use_case_pipeline
        build_pipeline.bat
        ```
    
**N.B.** EDDL is built for GPU by default.
    
## Training and inference

- The project creates different executables: MNIST_BATCH, SKIN_LESION_CLASSIFICATION_TRAINING, SKIN_LESION_SEGMENTATION_TRAINING, SKIN_LESION_CLASSIFICATION_INFERENCE, SKIN_LESION_SEGMENTATION_INFERENCE, PNEUMOTHORAX_SEGMENTATION_TRAINING and PNEUMOTHORAX_SEGMENTATION_INFERENCE.
    1. MNIST_BATCH and SKIN_LESION_CLASSIFICATION_TRAINING train the neural network loading the dataset in batches (needed when the dataset is too large to fit in memory).
    1. SKIN_LESION_SEGMENTATION_TRAINING trains the neural network loading the dataset (images and their ground truth masks) in batches for the segmentation task.
    1. PNEUMOTHORAX_SEGMENTATION_TRAINING trains the neural network loading the dataset (images and their ground truth masks) in batches with a custom function for this specific segmentation task.
    1. SKIN_LESION_CLASSIFICATION_INFERENCE, SKIN_LESION_SEGMENTATION_INFERENCE and PNEUMOTHORAX_SEGMENTATION_INFERENCE perform only inference on classification or segmentation task loading weights from a previous training process.

- Examples of output for the pre-trained models provided:
    1. *ISIC segmentation test set*:

       The red line represents the prediction processed by ECVL to obtain contours that are overlaid on the original image.

        ![](/imgs/isic_1.png)  |  ![](/imgs/isic_2.png)  |  ![](/imgs/isic_3.png) 
        :----------------------|-------------------------|----------------------:
    1. *Pneumothorax segmentation validation set*:

       The red area represents the prediction, the green area the ground truth. The yellow area therefore represents the correctly predicted pixels.

       ![](/imgs/pneumothorax_1.png) | ![](/imgs/pneumothorax_2.png) | ![](/imgs/pneumothorax_3.png)
       :----------------------------:|:-----------------------------:|:----------------------------:
