"""\
Tensorflow skin lesion classification training example.
"""

import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import models
from dataset import ISICClassification


def main(args):
    num_classes = 8
    size = (224, 224, 3)  # size of images
    learning_rate = args.learning_rate
    display_step = 3

    # Runtime initialization will not allocate all memory on GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    model = models.VGG16(input_shape=size, num_classes=num_classes)
    model.build(input_shape=size)
    model.summary()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.Adam(learning_rate)

    training_dataset = ISICClassification(args.dataset, 'training', args.batch_size, size[:-1])
    validation_dataset = ISICClassification(args.dataset, 'validation', args.batch_size, size[:-1], shuffle=False)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    best_accuracy = 0.
    for e in range(1, args.epochs + 1):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_accuracy.reset_states()

        for step, (images, labels) in enumerate(training_dataset, 1):
            # Run the optimization to update W and b values
            with tf.GradientTape() as tape:
                pred = model(images, is_training=True)
                loss = loss_fn(labels, pred)

            train_loss.update_state(loss)
            train_accuracy.update_state(labels, pred)
            gradients = tape.gradient(loss, model.trainable_variables)

            # Update W and b following gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if step % display_step == 0:
                print("\rTraining {:d}/{:d} (batch {:d}/{:d}) - Loss: {:.4f} - Accuracy: {:.4f}".format(
                    e, args.epochs, step, len(training_dataset), train_loss.result(), train_accuracy.result()
                ), end="", flush=True)

        print('\n')
        # Do validation
        print("Validation {:d}/{:d}".format(e, args.epochs), end="", flush=True)
        for step, (images, labels) in enumerate(validation_dataset, 1):
            pred = model(images, is_training=False)
            val_accuracy.update_state(labels, pred)

        # Compute accuracy and save checkpoints
        accuracy = val_accuracy.result()
        print(" - Accuracy: {:.4f}".format(accuracy), flush=True)

        if accuracy > best_accuracy:
            print("Saving checkpoints")
            best_accuracy = accuracy
            model.save_weights("checkpoint.tf", save_format='tf')

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
