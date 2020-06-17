import torch.utils.data as data
from PIL import Image
import yaml
from yaml import CLoader as Loader
from random import choice
import pydicom
import numpy as np
from torchvision import transforms
import albumentations as albu
import os


class PneumothoraxDataset(data.Dataset):
    def __init__(self, partition, args):
        self.is_training = partition == 'train'
        self.images, self.train_split, self.val_split, self.test_split = self.read(args.dataset_filepath)
        self.total_indices = list(range(0, len(self.images)))
        self.black = list(set(self.total_indices) - set(self.train_split + self.val_split + self.test_split))

        self.replicate = 'AlbuNet' in args.model

        if self.is_training:
            self.num_samples = int(len(self.train_split) * 1.25)
            self.black = self.black[:len(self.black) - int(len(self.val_split) * 0.25)]
            self.split = self.train_split
            self.transforms = albu.Compose([
                                albu.HorizontalFlip(),
                                albu.OneOf([
                                    albu.RandomContrast(),
                                    albu.RandomGamma(),
                                    albu.RandomBrightness(),
                                ], p=0.3),
                                albu.OneOf([
                                    albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                                    albu.GridDistortion(),
                                    albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                                ], p=0.3),
                                albu.ShiftScaleRotate(),
                                albu.Resize(args.resize_dims, args.resize_dims, always_apply=True),
                              ])
        else:
            self.num_samples = int(len(self.val_split) * 1.25)
            self.black = self.black[len(self.black) - (self.num_samples - len(self.val_split)):]
            self.split = self.val_split
            self.transforms = albu.Resize(args.resize_dims, args.resize_dims, always_apply=True)

        self.r_images = []
        self.r_gt = []
        self.r_black = []
        for idx in self.split:
            self.r_images.append(pydicom.dcmread(os.path.join(os.path.dirname(args.dataset_filepath), self.images[idx]['location'])))
            self.r_gt.append(np.asarray(
                Image.open(os.path.join(os.path.dirname(args.dataset_filepath), self.images[idx]['label'])),
                np.uint8
            ))

        for idx in self.black:
            self.r_black.append(pydicom.dcmread(os.path.join(os.path.dirname(args.dataset_filepath), self.images[idx]['location'])))
        self.black_gt = np.asarray(
            Image.open(os.path.join(os.path.dirname(args.dataset_filepath), self.images[self.black[0]]['label'])),
            np.uint8
        )

    def read(self, filename):
        print('Reading dataset file...')
        with open(filename) as file:
            try:
                yaml_dict = yaml.load(file, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)
        return yaml_dict['images'], yaml_dict['split']['training'], yaml_dict['split']['validation'], yaml_dict['split']['test']

    def __len__(self):
        return self.num_samples

    def __getitem__(self, p_index):
        if p_index < len(self.r_images):
            image_orig = self.r_images[p_index]
            ground_orig = self.r_gt[p_index]
        else:
            if self.is_training:
                image_orig = choice(self.r_black)
            else:
                image_orig = self.r_black[p_index - len(self.r_images)]
            ground_orig = self.black_gt

        aug_dict = {"image": image_orig.pixel_array, "mask": ground_orig}
        augmented = self.transforms(**aug_dict)
        image, ground = augmented["image"], augmented["mask"]

        image = transforms.ToTensor()(image)
        ground = transforms.ToTensor()(ground)

        if self.replicate:
            image = image.repeat(3, 1, 1)

        return image, ground
