import os.path

import cv2
import numpy as np
import torch.utils.data as data
import torchvision
import yaml

try:
    from yaml import CLoader as Loader
except:
    from yaml import Loader


class YAMLClassificationDataset(data.Dataset):

    def __init__(self, dataset=None, transform=None, split=['training'], normalization=None):
        """Initializes a pytorch Dataset object
        :param dataset: A filename (string), to identify the yaml file
          containing the dataset.
        :param transform: Transformation function to be applied to the input
          images (e.g. created with torchvision.transforms.Compose()).
        :param split: A list of strings, one for each dataset split to be
          loaded by the Dataset object.
        :param normalization: Tuple (mean,std) for normalization: subtract the mean from each pixel and
          then dividing the result by the standard deviation.
        """

        self.dataset = dataset
        self.transform = transform
        self.imgs = []
        self.lbls = []
        if normalization is None:
            # Normalize using imagenet statistics
            self.normalization = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            self.normalization = normalization

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
                self.lbls.append(d['images'][i]['label'])

    def __getitem__(self, index):
        image = cv2.imread(self.imgs[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)

        if self.transform is not None:
            image = self.transform(image=image)

        image = torchvision.transforms.ToTensor()(image)
        image = torchvision.transforms.Normalize(*self.normalization)(image)

        return image, self.lbls[index], os.path.basename(self.imgs[index])

    def __len__(self):
        return len(self.lbls)
