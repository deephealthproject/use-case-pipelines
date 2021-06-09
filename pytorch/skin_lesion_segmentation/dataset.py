import os.path

import albumentations as A
import numpy as np
import torch
import torch.utils.data as data
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import functional
from yaml import CLoader as Loader


class YAMLSegmentationDataset(data.Dataset):

    def __init__(self, dataset=None, transform=None, split=['training']):
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

    def CenterCrop(self, image, mask):
        resize_val = min(*image.shape[:2])
        image = A.center_crop(image, resize_val, resize_val)
        mask = A.center_crop(mask, resize_val, resize_val)
        return image, mask

    def Pad(self, image, mask):
        max_shape = max(*image.shape[:2])
        new_image = np.zeros((max_shape, max_shape, image.shape[-1]), dtype=np.uint8)
        new_mask = np.zeros((max_shape, max_shape), dtype=np.uint8)
        # Getting the centering position
        ax, ay = (max_shape - image.shape[1]) // 2, (max_shape - image.shape[0]) // 2
        new_image[ay:image.shape[0] + ay, ax:ax + image.shape[1]] = image
        new_mask[ay:image.shape[0] + ay, ax:ax + image.shape[1]] = mask
        return new_image, new_mask

    def __getitem__(self, index):
        image = np.array(Image.open(self.imgs[index]), dtype=np.uint8)
        mask = np.array(Image.open(self.masks[index]), dtype=np.uint8)

        original_shape = torch.tensor(mask.shape)
        image, mask = self.Pad(image, mask)
        # image, mask = self.CenterCrop(image, mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image, mask = functional.to_tensor(image), functional.to_tensor(mask)
        mask = torch.unsqueeze(mask, dim=0) / 255

        return image, mask, original_shape  # os.path.basename(self.imgs[index])

    def __len__(self):
        return len(self.imgs)

    def PrintTensor(self, image):
        image = image.detach().permute(1, 2, 0)
        image = image.numpy()
        plt.imshow(image)
        plt.show()
