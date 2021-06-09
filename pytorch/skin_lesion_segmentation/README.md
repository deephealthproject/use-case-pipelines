# PyTorch Skin Lesion Segmentation

This folder contains the skin lesion segmentation pipeline written in PyTorch.

## Install and run:
```shell
virtualenv myenv
source myenv/bin/activate
cd pytorch/skin_lesion_segmentation
pip install -r requirements.txt
main.py /path_to/isic_segmentation.yml--model deeplabv3plus --gpu 1 --batch_size 10 --workers 4 --learning_rate 5e-3 --size 512 --onnx-export
```
