# PyTorch Pipelines Adaptation

```
pytorch
  └── skin_lesion_classification
        └── ...
  └── yaml_segmentation_dataset.py
  └── yaml_classification_dataset.py
```

## PyTorch Dataloaders and DeepHealth Dataset Format

This folder contains two [PyTorch dataloaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for classification and segmentation, which are compatible with [DeepHealth Toolkit Dataset Format](https://github.com/deephealthproject/ecvl/wiki/DeepHealth-Toolkit-Dataset-Format).

The following lines provide an usage example for classification task (using torchvision for augmentations):
```python
from torchvision import transforms
from torch.utils.data import DataLoader
if __name__ == '__main__':
    dname = 'isic_classification.yml' # ECVL dataset path
    custom_training_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
    ])
    custom_evaluation_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
    ])
    dataset = YAMLClassificationDataset(dataset=dname, transform=custom_training_transforms, split=['training'])
    test_dataset = YAMLClassificationDataset(dataset=dname, transform=custom_evaluation_transforms, split=['test'])
    valid_dataset = YAMLClassificationDataset(dataset=dname, transform=custom_evaluation_transforms, split=['validation'])
    data_loader = DataLoader(dataset,
                             batch_size=8,
                             shuffle=True,
                             num_workers=4,
                             drop_last=True,
                             pin_memory=True)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=8,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False,
                                  pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset,
                                   batch_size=8,
                                   shuffle=False,
                                   num_workers=4,
                                   drop_last=False,
                                   pin_memory=True)
```