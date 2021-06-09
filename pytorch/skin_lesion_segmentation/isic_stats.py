import os.path

import numpy as np
import yaml
from PIL import Image
from yaml import CLoader as Loader

if __name__ == '__main__':
    # dataset_p = '/nas/softechict-nas-3/mcancilla/data/isic_segmentation/isic_segmentation.yml'
    dataset_p = 'D:/dataset/isic_segmentation/isic_segmentation.yml'
    # split = ['training', 'validation', 'test']
    split = ['test']
    images = {}
    images_stats = {}
    with open(dataset_p, 'r') as stream:
        try:
            d = yaml.load(stream, Loader=Loader)
        except yaml.YAMLError as exc:
            print(exc)

    data_root = os.path.dirname(dataset_p)
    for s in split:
        images[s] = []
        for img in d['split'][s]:
            images[s].append(os.path.join(data_root, d['images'][img]['location']))
    import time

    for s in split:
        images_stats[s] = [np.zeros(3), np.zeros(3)]
        for i, img in enumerate(images[s]):
            image = np.array(Image.open(img), dtype=np.float32) / 255
            start = time.time()
            # image = np.reshape(image, (-1, 3))
            images_stats[s][0] += np.mean(image, axis=tuple(np.arange(len(image.shape) - 1)))
            images_stats[s][1] += np.std(image, axis=tuple(np.arange(len(image.shape) - 1)))
            stop = time.time()
            if i % 100 == 0:
                print(f'{i}/{len(images[s])}', stop - start, images_stats[s][0], images_stats[s][1])

        mean = images_stats[s][0] / len(images[s])
        std = images_stats[s][1] / len(images[s])

        # output
        print('mean: ' + str(mean))
        print('std:  ' + str(std))
