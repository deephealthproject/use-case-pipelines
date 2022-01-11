"""\
Tensorflow skin lesion classification example
"""
import argparse
import os
import random
import sys
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import models
from dataset import ISICClassification

# Set seed value
seed_value = 50
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def main(args):
    # Print settings
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    num_classes = 8
    size = (224, 224, 3)  # size of images

    # Runtime initialization will not allocate all memory on GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    # Create checkpoints dir
    os.makedirs('saved_models', exist_ok=True)

    optimizer = optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metrics = [keras.metrics.SparseCategoricalAccuracy()]

    # model = models.vgg16(input_shape=size, num_classes=num_classes, classifier_activation='softmax')
    model = models.resnet50(input_shape=size, num_classes=num_classes, classifier_activation='softmax')
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    if args.checkpoints:
        if os.path.exists(args.checkpoints):
            print(f'Loading checkpoints: {args.checkpoints}')
            model.load_weights(args.checkpoints)
        else:
            print(f'Checkpoints `{args.checkpoints}` not found', file=sys.stderr)

    os.makedirs("logs/scalars/", exist_ok=True)
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Log loss/metrics for training and validation
    tensorboard = keras.callbacks.TensorBoard(log_dir=logdir)

    if args.train:
        # Same augs as C++
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
        training_tfdata = training_dataset.map_samples(args.epochs)

        validation_dataset = ISICClassification(args.dataset, 'validation', args.batch_size, val_aug, shuffle=False)
        validation_tfdata = validation_dataset.map_samples(args.epochs)

        # Save checkpoints
        checkpoint = ModelCheckpoint(f'saved_models/{args.name}.h5', monitor='val_sparse_categorical_accuracy',
                                     verbose=1,
                                     save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

        # Stop training after 20 epochs of no improvement
        early = EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta=0, patience=args.epochs // 4,
                              verbose=1,
                              mode='auto')

        # Train the model
        model.fit(
            x=training_tfdata,
            epochs=args.epochs,
            verbose=1,
            callbacks=[checkpoint, early, tensorboard],
            validation_data=validation_tfdata,
            steps_per_epoch=len(training_dataset),
            validation_steps=len(validation_dataset),
        )

    if args.test:
        # Test model on test set
        test_aug = iaa.Sequential([iaa.Resize(size=size[:-1], interpolation='cubic')])
        test_dataset = ISICClassification(args.dataset, 'test', args.batch_size, test_aug)
        test_tfdata = test_dataset.map_samples(1)

        results = model.evaluate(test_tfdata, verbose=1, callbacks=[tensorboard])
        print("Test set loss and accuracy:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', metavar='DATASET_YAML_PATH')
    parser.add_argument('-e', '--epochs', type=int, metavar='INT', default=50)
    parser.add_argument('-b', '--batch-size', type=int, metavar='INT', default=32)
    parser.add_argument('-l', '--learning-rate', type=float, metavar='FLOAT', default=1e-3)
    parser.add_argument('-n', '--name', metavar='DIR', help='Experiment name', default='isic_class')
    parser.add_argument('-c', '--checkpoints', metavar='CHECKPOINTS_PATH', help='Continue training from a checkpoint')
    parser.add_argument('--train', dest='train', action='store_true', help='Do train')
    parser.add_argument('--no-train', dest='train', action='store_false', help='Do not train')
    parser.add_argument('--test', dest='test', action='store_true', help='Do inference')
    parser.add_argument('--no-test', dest='test', action='store_false', help='Do not inference')
    parser.set_defaults(train=True, test=False)

    main(parser.parse_args())
