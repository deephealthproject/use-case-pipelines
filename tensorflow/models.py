import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def resnet50(input_shape, num_classes, classifier_activation=None):
    features = keras.applications.ResNet50V2(include_top=False, input_shape=input_shape, pooling='avg')
    features.trainable = False  # Freeze features extractor training
    model = keras.Sequential(features)
    # Classification blocks
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation=classifier_activation))
    return model


def vgg16(input_shape, num_classes, classifier_activation=None):
    features = keras.applications.VGG16(include_top=False, input_shape=input_shape)
    features.trainable = False
    model = keras.Sequential(features)
    # Classification block
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(num_classes, activation=classifier_activation))
    return model


class VGG16(Model):
    def __init__(self, input_shape, num_classes, classifier_activation=None):
        super().__init__()
        self.input_shape_ = input_shape
        self.features = keras.applications.VGG16(include_top=False, input_shape=input_shape)

        self.classifier = keras.Sequential(layers=[
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dense(4096, activation='relu'),
            layers.Dense(num_classes, activation=classifier_activation),
        ])

    # Set forward pass
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, *self.input_shape_])
        x = self.features(x)
        x = self.classifier(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only apply softmax when not training.
            x = tf.nn.softmax(x)
        return x
