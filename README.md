# DH USE CASE Pipeline 

Pipeline that uses EDDL and ECVL to train a CNN on two different datasets (_MNIST_ and _ISIC_), applying different image augmentations, for both the classification and the segmentation task.

## Requirements
- _C++-17_ compiler (gcc-8 or Visual Studio 2017);
- CMake;
- (Optional) ISIC dataset.

### Datasets
The YAML datasets format is described [here](https://github.com/deephealthproject/ecvl/wiki/DeepHealth-Toolkit-Dataset-Format). Each dataset listed below contains both the data and the YAML description format.

- **MNIST**: Automatically downloaded and extracted by CMake.
- **ISIC**:
  - _Classification_: Download it from [here](https://drive.google.com/uc?id=1TCE-uswZ41nlqMe5SWHoCGF7Mtq6r15A&export=download) and extract it. Change the dataset path into the `skin_lesion_classification_training.cpp` source file accordingly. To perform only inference, change the dataset path into the `skin_lesion_classification_inference.cpp` source file and download checkpoints [here](https://drive.google.com/file/d/1HzEtAni3WNmpwBBBT6fZ5hkW9wJAgqF2/view?usp=sharing) (best accuracy on validation in 50 epochs).
  - _Segmentation_: Download it from [here](https://drive.google.com/uc?id=1RyYa32x9aqwd2kkQpCZ4Xa2h_VcgH3wI&export=download) and extract it. Change the dataset path into the `skin_lesion_segmentation_training.cpp` source file accordingly. To perform only inference, change the dataset path into the `skin_lesion_segmentation_inference.cpp` source file and download checkpoints [here](https://drive.google.com/file/d/13lbpkjrHNZygbdkdux8yr7GlpTW0MgxM/view?usp=sharing) (best Mean Intersection over Union on validation in 50 epochs).
  
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
2. The project creates different executables: MNIST_BATCH, SKIN_LESION_CLASSIFICATION_TRAINING, SKIN_LESION_SEGMENTATION_TRAINING, SKIN_LESION_CLASSIFICATION_INFERENCE, SKIN_LESION_SEGMENTATION_INFERENCE.
    1. MNIST_BATCH and SKIN_LESION_CLASSIFICATION_TRAINING train the neural network loading the dataset in batches (needed when the dataset is too large to fit in memory).
    1. SKIN_LESION_SEGMENTATION_TRAINING trains the neural network loading the dataset (images and their ground truth masks) in batches for the segmentation task.
    1. SKIN_LESION_CLASSIFICATION_INFERENCE and SKIN_LESION_SEGMENTATION_INFERENCE perform only inference on classification or segmentation task loading weights from a previous training process.
