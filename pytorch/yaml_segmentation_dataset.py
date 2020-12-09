from pathlib import Path
import os.path

import torch.utils.data as data
import yaml
from PIL import Image
from yaml import CLoader as Loader
import nibabel as nib
import numpy as np


class YAMLSegmentationDataset(data.Dataset):

    def __init__(self, dataset=None, transform=None, input_image_transform=None,
                 mask_transform=None, split=['training']):
        """Initializes a pytorch Dataset object

        :param dataset: A filename (string), to identify the yaml file
          containing the dataset.
        :param transform: Transformation function to be applied to the input
          images AND the segmentation masks.
          This parameters requires functions that accept two images as input and
          return two images as output (input image AND segmentation mask).
        :param input_image_transform: Transformation function to be applied to the
          input images ONLY (e.g. created with torchvision.transforms.Compose()).
        :param mask_transform: Transformation function to be applied to the
          segmentation masks ONLY (e.g. created with torchvision.transforms.Compose()).
        :param split: A list of strings, one for each dataset split to be
          loaded by the Dataset object.
        """
        self.dataset = dataset
        self.transform = transform
        self.input_image_transform = input_image_transform
        self.mask_transform = mask_transform
        self.imgs = []
        self.masks = []

        data_root = os.path.dirname(dataset)

        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        for s in split:
            for i in d['split'][s]:
                self.imgs.append(
                    os.path.join(data_root, d['images'][i]['location']))
                self.masks.append(
                    os.path.join(data_root, d['images'][i]['label']))

    def __getitem__(self, index):
        image = Image.open(self.imgs[index])
        mask = Image.open(self.masks[index])

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if self.input_image_transform is not None:
            image = self.input_image_transform(image)

        if self.mask_transform is not None:
            mask = self.transform(mask)

        return image, mask, os.path.basename(self.imgs[index])

    def __len__(self):
        return len(self.imgs)


class MSSegmentationDataset(YAMLSegmentationDataset):
    def __init__(self, dataset=None, transform=None, input_image_transform=None, mask_transform=None,
                 split=['training']):
        """Initializes a pytorch Dataset object

        :param dataset: A filename (string), to identify the yaml file
          containing the dataset.
        :param transform: Transformation function to be applied to the input
          images AND the segmentation masks.
          This parameters requires functions that accept two images as input and
          return two images as output (input image AND segmentation mask).
        :param input_image_transform: Transformation function to be applied to the
          input images ONLY (e.g. created with torchvision.transforms.Compose()).
        :param mask_transform: Transformation function to be applied to the
          segmentation masks ONLY (e.g. created with torchvision.transforms.Compose()).
        :param split: A list of strings, one for each dataset split to be
          loaded by the Dataset object.
        """
        self.dataset = dataset
        self.transform = transform
        self.input_image_transform = input_image_transform
        self.mask_transform = mask_transform
        self.vols = []
        self.masks = []
        self.n_slices = []
        self.n_slices_cum = []

        data_root = Path(dataset).parent

        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        for s in split:
            for i in d['split'][s]:
                vol_path = Path(data_root, d['images'][i]['location'])
                gt_path = Path(data_root, d['images'][i]['label'])

                # Open volume and counts its slides
                vol = nib.load(vol_path)
                vol = np.array(vol.dataobj, np.float32)
                gt = nib.load(gt_path)
                gt = np.array(gt.dataobj, np.float32)

                self.n_slices.append(vol.shape[0])
                self.vols.append(vol)
                self.masks.append(gt)
        self.n_slices_cum = np.cumsum(self.n_slices)

    def __getitem__(self, index):
        greater_than_threshold = self.n_slices_cum > index
        vol_index = greater_than_threshold.searchsorted(True)

        image = self.vols[vol_index][index - self.n_slices_cum[vol_index], ...].copy()
        mask = self.masks[vol_index][index - self.n_slices_cum[vol_index], ...].copy()

        # Convert to PIL
        # image = Image.fromarray(image)
        # mask = Image.fromarray(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask.unsqueeze(0)  # , os.path.basename(self.vols[index])

    def __len__(self):
        return self.n_slices_cum[-1]
