# DH USE CASE Pipeline 

Pipeline that uses EDDL and ECVL to train a CNN on two different datasets (_MNIST_ and _ISIC_) for the classification task.


## Requirements
- Download, build and install EDDL. Set the `BUILD_TARGET` option to `GPU` and enable `EDDL_WITH_CUDA` flag if you have a GPU.
- Download, build and install ECVL enabling `ECVL_BUILD_EDDL` and providing the EDDL and OpenCV install directories in `EDDL_DIR` and `OpenCV_DIR`.
- (Optional) Download the ISIC dataset.

### Datasets

- _MNIST_: Automatically downloaded and extracted by CMake.
- _ISIC_: Download it from [here](https://drive.google.com/uc?id=1wo3Ai0gBTwy42s89aa3Jl20B7EGm7nKa&export=download) and extract it. Put the dataset path into the `skin_lesion_classification.cpp` source file.

### CUDA
- On Linux systems, starting from CUDA 10.1, cuBLAS libraries are installed in the `/usr/lib/<arch>-linux-gnu/` or `/usr/lib64/`. Create a symlink to resolve the issue:
```bash
sudo ln -s /usr/lib/<arch>-linux-gnu/libcublas.so /usr/local/cuda-10.1/lib64/libcublas.so
```

## Building
1. Build the project filling the `ECVL_DIR` CMake variable with the installation path of ECVL.
1. The project creates three executables: MNIST_FIT, MNIST_BATCH and SKIN_LESION_CLASSIFICATION.
	1. MNIST_FIT loads the entire dataset in memory, after that trains the neural network with batches of the dataset.
	1. MNIST_BATCH and SKIN_LESION_CLASSIFICATION train the neural network loading the dataset in batches (needed when the dataset is too large to fit in memory).