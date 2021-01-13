"""\
Tensorflow skin lesion classification training example.
"""
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

import models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from dataset import ISICClassification
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import imgaug.augmenters as iaa
import random

# Set seed value
seed_value = 50
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def main(args):
    # ee = tf.data.Dataset()
    num_classes = 8
    size = (224, 224, 3)  # size of images
    learning_rate = args.learning_rate

    # Rruntime initialization will not allocate all memory on GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metrics = [keras.metrics.SparseCategoricalAccuracy()]

    model = models.vgg16(input_shape=size, num_classes=num_classes, classifier_activation='softmax')
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    # if os.path.exists('isic_class.h5'):
    #     # load 3rd epoch weights
    #     model.load_weights('isic_class.h5')

    train_aug = iaa.Sequential([
        iaa.Resize(size=size[:-1], interpolation='cubic'),
        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5),
        iaa.Rotate(rotate=(-180, 180)),
        iaa.AdditivePoissonNoise(lam=(0, 10)),
        iaa.GammaContrast(gamma=(.8, 1.5)),
        iaa.GaussianBlur(sigma=(.0, .8)),
        iaa.CoarseDropout(p=(.02, .1), size_px=(0.02, 0.05), size_percent=0.5),
    ])

    val_aug = iaa.Sequential([iaa.Resize(size=size[:-1], interpolation='cubic')])

    training_dataset = ISICClassification(args.dataset, 'training', args.batch_size, train_aug)
    validation_dataset = ISICClassification(args.dataset, 'validation', args.batch_size, val_aug, shuffle=False)

    print('Loading validation images...', end='')
    val_samples = validation_dataset.load_samples()
    print('done')
    # Save checkpoints
    checkpoint = ModelCheckpoint("isic_class_vgg.h5", monitor='val_sparse_categorical_accuracy', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

    # stop training after 20 epochs of no improvement
    early = EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta=0, patience=args.epochs // 4, verbose=1,
                          mode='auto')

    # Train the model
    history = model.fit(
        x=training_dataset,
        epochs=args.epochs,
        verbose=1,
        callbacks=[checkpoint, early],
        validation_data=val_samples,
        steps_per_epoch=len(training_dataset),
        validation_steps=len(validation_dataset),
        use_multiprocessing=True,
        workers=8,
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', metavar='INPUT_DATASET')
    parser.add_argument('-e', '--epochs', type=int, metavar='INT', default=50)
    parser.add_argument('-b', '--batch-size', type=int, metavar='INT', default=32)
    parser.add_argument('-l', '--learning-rate', type=float, metavar='FLOAT', default=1e-3)
    # parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--out-dir', metavar='DIR', help='if set, save images in this directory')
    main(parser.parse_args())
