# DH USE CASE Pipeline 

Pipeline that uses EDDL and ECVL to train a CNN on two different datasets (_MNIST_ and _ISIC_) for both the classification and the segmentation task.


## Requirements
- _C++-17_ compiler (gcc-8 or Visual Studio 2017);
- CMake;
- (Optional) ISIC dataset.

### Datasets
The YAML datasets format is described [here](https://github.com/deephealthproject/ecvl/wiki/DeepHealth-Toolkit-Dataset-Format). Each dataset listed below contains both the data and the YAML description format.

- _MNIST_: Automatically downloaded and extracted by CMake.
- _ISIC_:
  - Classification: Download it from [here](https://drive.google.com/uc?id=1TCE-uswZ41nlqMe5SWHoCGF7Mtq6r15A&export=download) and extract it. Change the dataset path into the `skin_lesion_classification.cpp` source file accordingly.
  - Segmentation: Download it from [here](https://drive.google.com/uc?id=1RyYa32x9aqwd2kkQpCZ4Xa2h_VcgH3wI&export=download) and extract it. Change the dataset path into the `skin_lesion_segmentation.cpp` source file accordingly.

The YAML dataset files can also be downloaded separately: [classification](https://drive.google.com/uc?id=1pZotvwM5rltg5OhYGr9oSLVW8yxjs3U_&export=download) and [segmentation](https://drive.google.com/uc?id=1HHmBNiyQ1dH398E3ECl8WqVoZuV23fjM&export=download).

### CUDA
On Linux systems, starting from CUDA 10.1, cuBLAS libraries are installed in the `/usr/lib/<arch>-linux-gnu/` or `/usr/lib64/`. Create a symlink to resolve the issue:
```bash
sudo ln -s /usr/lib/<arch>-linux-gnu/libcublas.so /usr/local/cuda-10.1/lib64/libcublas.so
```

## Building
1.  Downloads and builds the dependencies of the project running:
    - `build_pipeline.sh` (_\*nix_)
    - `build_pipeline.bat` (_Windows_)

    **N.B.** EDDL is built for GPU by default.
2. The project creates different executables: MNIST_FIT, MNIST_BATCH, SKIN_LESION_CLASSIFICATION, and SKIN_LESION_SEGMENTATION.
    1. MNIST_FIT loads the entire dataset in memory, after that trains the neural network with batches of the dataset.
    1. MNIST_BATCH and SKIN_LESION_CLASSIFICATION train the neural network loading the dataset in batches (needed when the dataset is too large to fit in memory).
    1. SKIN_LESION_SEGMENTATION trains the neural network loading the dataset (images and their ground truth masks) in batches for the segmentation task.

## TODO
- Apply image augmentations when a batch is loaded
