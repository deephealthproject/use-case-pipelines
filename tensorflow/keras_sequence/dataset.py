import os.path
import random

import numpy as np
import yaml
from tensorflow import keras
import cv2

try:
    from yaml import CLoader as Loader
except ImportError:
    pass


class ISICClassification(keras.utils.Sequence):
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
        self.imgs = []
        self.lbls = []

        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        for i in d['split'][split]:
            self.imgs.append(os.path.join(self.data_root, d['images'][i]['location']))
            self.lbls.append(d['images'][i]['label'])

        self.samples = list(zip(self.imgs, self.lbls))
        self.on_epoch_end()

    def __getitem__(self, index):
        # Generate indexes of the batch
        samples = self.samples[index * self.batch_size:(index + 1) * self.batch_size]

        images = []
        labels = []
        # filenames = []
        for i, l in samples:
            img = cv2.imread(i)[..., ::-1]

            if self.transform is not None:
                img = self.transform(image=img)
            images.append(img)
            labels.append(l)
            # filenames.append(os.path.basename(i))

        images = np.array(images, np.float32) / 255.

        return images, np.array(labels)[:, np.newaxis]

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        random.shuffle(self.samples)

    def load_samples(self):
        """
        Return all dataset samples
        """
        images = [cv2.imread(i) for i in self.imgs]
        if self.transform is not None:
            images = [self.transform(image=i) for i in images]
        images = np.array(images, np.float32) / 255.
        return images, np.array(self.lbls)[:, np.newaxis]
