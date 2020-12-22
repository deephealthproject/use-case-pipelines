# Tensorflow pipeline implementation

This folder contains a classification example, using the [DeepHealth Toolkit Dataset Format](https://github.com/deephealthproject/ecvl/wiki/DeepHealth-Toolkit-Dataset-Format).


Currently, only `skin_lesion_classification` script has been ported.

The repository has the following structure:
```text
|
|- dataset.py  # Loads a `tf.data.Dataset` and applies augmentations
|- models.py  # Lists some neural network models
|- skin_lesion_classification_training.py  # Main file that performs training using custom training loop
|- skin_lesion_classification_training.py  # Main file that uses fit
|- keras_sequence  # Contains scripts which use `keras.utils.Sequence`, instead of `tf.data.Dataset` 
```

### Building
The ISIC dataset must be downloaded following instructions in the main [README](../README.md).  

Build and run:
```bash
cd tensorflow
virtualenv tf_pipeline
source tf_pipeline/bin/activate
# Change tensorflow version in requirements.txt 
# according to your installed cuda/cudnn versions
pip install -r requirements.txt
python skin_lesion_classification.py path/to/isic_classification.yml -e 100 -b 24 -l 1e-4 --name isic_class_resnet --do-test
```