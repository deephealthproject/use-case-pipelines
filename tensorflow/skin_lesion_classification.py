"""\
Tensorflow skin lesion classification example
"""
import argparse
import os
import random
import sys
from datetime import datetime

import imgaug.augmenters as iaa
import numpy as np
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

import models
import utils
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

    display_step = 5
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
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # model = models.vgg16(input_shape=size, num_classes=num_classes, classifier_activation='softmax')
    model = models.resnet50(input_shape=size, num_classes=num_classes, classifier_activation='softmax')
    model.build(input_shape=size)
    model.summary()

    if args.checkpoints:
        if os.path.exists(args.checkpoints):
            print(f'Loading checkpoints: {args.checkpoints}')
            model.load_weights(args.checkpoints)
        else:
            print(f'Checkpoints `{args.checkpoints}` not found', file=sys.stderr)

    os.makedirs("logs/scalars/", exist_ok=True)
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{args.name}"
    summary_writer = tf.summary.create_file_writer(logdir)

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
            iaa.CoarseDropout(p=(.02, .1), size_percent=(0.02, 0.05), per_channel=0.5),
        ])

        val_aug = iaa.Sequential([iaa.Resize(size=size[:-1], interpolation='cubic')])

        training_dataset = ISICClassification(args.dataset, 'training', args.batch_size, train_aug)
        training_tfdata = training_dataset.map_samples(args.epochs)
        training_iter = iter(training_tfdata)

        validation_dataset = ISICClassification(args.dataset, 'validation', args.batch_size, val_aug, shuffle=False)
        validation_tfdata = validation_dataset.map_samples(args.epochs)
        validation_iter = iter(validation_tfdata)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        val_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        best_accuracy = 0.
        for e in range(1, args.epochs + 1):
            train_loss.reset_states()
            train_metric.reset_states()
            val_metric.reset_states()

            total_preds = []
            total_labels = []
            for step in range(1, len(training_dataset)):
                images, labels = next(training_iter)

                # Run the optimization to update W and b values
                with tf.GradientTape() as tape:
                    pred = model(images)
                    loss = loss_fn(labels, pred)
                total_preds.append(pred)
                total_labels.append(labels)

                gradients = tape.gradient(loss, model.trainable_variables)

                # Update W and b following gradients
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                # Log loss and metric
                train_loss.update_state(loss)
                train_metric.update_state(labels, pred)
                if step % display_step == 0:
                    print("\rTraining {:d}/{:d} (batch {:d}/{:d}) - Loss: {:.4f} - Accuracy: {:.4f}".format(
                        e, args.epochs, step, len(training_dataset), train_loss.result(), train_metric.result()
                    ), end="", flush=True)

            cm = utils.calculate_confusion_matrix(tf.concat(total_labels, axis=0), tf.concat(total_preds, axis=0))
            with summary_writer.as_default():
                tf.summary.scalar('loss/' + train_loss.name, train_loss.result(), step=e - 1)
                tf.summary.scalar('accuracy/' + train_metric.name, train_metric.result(), step=e - 1)
                tf.summary.image("cm/training_cm", cm, step=e)

            total_preds = []
            total_labels = []

            # Do validation
            print("\nValidation {:d}/{:d}".format(e, args.epochs), end="", flush=True)
            for step in range(1, len(validation_dataset)):
                images, labels = next(validation_iter)
                pred = model(images)
                val_metric.update_state(labels, pred)
                total_preds.append(pred)
                total_labels.append(labels)

            cm = utils.calculate_confusion_matrix(tf.concat(total_labels, axis=0), tf.concat(total_preds, axis=0))
            with summary_writer.as_default():
                tf.summary.scalar('accuracy/' + val_metric.name, val_metric.result(), step=e - 1)
                tf.summary.image("cm/validation_cm", cm, step=e)

            # Compute accuracy and save checkpoints
            accuracy = val_metric.result()
            print(" - Accuracy: {:.4f}".format(accuracy), flush=True)

            if accuracy > best_accuracy:
                print(f"Saving checkpoints (accuracy: {accuracy:.4f} > {best_accuracy:.4f})", flush=True)
                best_accuracy = accuracy
                model.save_weights(f'saved_models/{args.name}.h5')

    if args.test:
        # Test model on test set
        test_aug = iaa.Sequential([iaa.Resize(size=size[:-1], interpolation='cubic')])
        test_dataset = ISICClassification(args.dataset, 'test', args.batch_size, test_aug)
        test_tfdata = test_dataset.map_samples(1)
        tensorboard = keras.callbacks.TensorBoard(log_dir=logdir)
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
