import os.path

import torch.utils.data as data
import yaml
from PIL import Image
from yaml import CLoader as Loader


class YAMLClassificationDataset(data.Dataset):

    def __init__(self, dataset=None, transform=None, split=['training']):
        """Initializes a pytorch Dataset object

        :param dataset: A filename (string), to identify the yaml file
          containing the dataset.
        :param transform: Transformation function to be applied to the input
          images (e.g. created with torchvision.transforms.Compose()).
        :param split: A list of strings, one for each dataset split to be
          loaded by the Dataset object.
        """

        self.dataset = dataset
        self.transform = transform
        self.imgs = []
        self.lbls = []

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
        image = Image.open(self.imgs[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, self.lbls[index], os.path.basename(self.imgs[index])

    def __len__(self):
        return len(self.lbls)
