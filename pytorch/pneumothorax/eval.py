from statistics import mean

class Eval():
    def __init__(self):
        self.metric_list = []
        self.eps = 1e-06

    def reset_eval(self):
        self.metric_list.clear()

    def mean_metric(self):
        return mean(self.metric_list)

    def binary_iou(self, image, gt, thresh=0.5):
        image[image < thresh] = 0.
        image[image >= thresh] = 1.

        gt[gt < thresh] = 0.
        gt[gt >= thresh] = 1.

        intersection = (image.astype(int) & gt.astype(int)).sum()
        unions = (image.astype(int) | gt.astype(int)).sum()

        iou = (intersection + self.eps) / (unions + self.eps)
        self.metric_list.append(iou)

        return iou

    def dice_coefficient(self, image, gt, thresh=0.5):
        image[image < thresh] = 0.
        image[image >= thresh] = 1.

        gt[gt < thresh] = 0.
        gt[gt >= thresh] = 1.

        intersection = (image.astype(int) & gt.astype(int)).sum(axis=(1,2))
        unions = (image.astype(int) + gt.astype(int)).sum(axis=(1,2))

        dice = (2 * intersection + self.eps) / (unions + self.eps)
        self.metric_list += list(dice)

        return dice
