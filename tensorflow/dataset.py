import os.path

import yaml
import cv2
import tensorflow as tf
import numpy as np

try:
    from yaml import CLoader as Loader
except ImportError:
    pass


class ISICClassification:
    def __init__(self, dataset, split, batch_size, transform=None, shuffle=True):
        """Initializes a pytorch Dataset object

        :param dataset: A filename (string), to identify the yaml file containing the dataset.
        :param split: A string which identifies a dataset split to be loaded by the Dataset object.
        :param batch_size: integer representing the size of the mini-batch.
        :param size: tuple which identifies the size of the input images.
        :param shuffle: Whether to shuffle sample data or not.

        """
        self.dataset = dataset
        self.data_root = os.path.dirname(dataset)
        self.split = split
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.files = []
        self.labels = []

        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        for i in d['split'][split]:
            self.files.append(os.path.join(self.data_root, d['images'][i]['location']))
            self.labels.append(d['images'][i]['label'])

    def __len__(self, *args, **kwargs):
        return len(self.labels)

    def __call__(self, *args, **kwargs):
        """
        Return all dataset samples
        """
        return self.files, self.labels

    def read_image(self, file):
        img = cv2.imread(file)[..., ::-1]
        if self.transform is not None:
            img = self.transform(image=img)
        img = img.astype(np.float32) / 255.
        return img

    def read_samples(self, file, label):
        if not isinstance(file, bytes):
            images = [self.read_image(f.decode("utf-8")) for f in file]
        else:
            images = self.read_image(file.decode("utf-8"))
        return images, label.astype(np.uint8)

    def map_samples(self, epochs):
        """
        Create a tf.Dataset from filenames and labels
        :param epochs: Total number of epochs
        :return:
        """
        dataset = tf.data.Dataset.from_tensor_slices(self.__call__())
        if self.shuffle:
            dataset = dataset.shuffle(len(self), reshuffle_each_iteration=True)
        parse_samples = lambda file, label: tf.numpy_function(self.read_samples,
                                                              inp=(file, label),
                                                              Tout=(tf.float32, tf.uint8))
        dataset = dataset.map(parse_samples, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(epochs)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
