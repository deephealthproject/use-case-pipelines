import os.path

import torch.utils.data as data
import yaml
from PIL import Image
from yaml import CLoader as Loader


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
