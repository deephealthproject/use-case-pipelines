#include "metrics.h"

using namespace ecvl;

void Eval::ResetEval()
{
    metric_list_.clear();
}

float Eval::MeanMetric()
{
    return std::accumulate(metric_list_.begin(), metric_list_.end(), 0.0) / metric_list_.size();
}

float Eval::BinaryIoU(Image& img, Image& gt, float thresh)
{
    float intersection = 0;
    float unions = 0;

    auto i_img = img.ContiguousBegin<float>(), e_img = img.ContiguousEnd<float>();
    auto i_gt = gt.ContiguousBegin<float>();

    for (; i_img != e_img; ++i_img, ++i_gt) {
        *i_img = ((*i_img) < thresh) ? 0 : 1;
        *i_gt = ((*i_gt) < thresh) ? 0 : 1;

        intersection += ((*i_gt == 1) && (*i_img == 1));
        unions += ((*i_gt == 1) || (*i_img == 1));
    }

    float iou = (intersection + eps_) / (unions + eps_);
    metric_list_.push_back(iou);

    return iou;
}

float Eval::DiceCoefficient(Image& img, Image& gt, float thresh)
{
    float intersection = 0;
    float unions = 0;

    auto i_img = img.ContiguousBegin<float>(), e_img = img.ContiguousEnd<float>();
    auto i_gt = gt.ContiguousBegin<float>();

    for (; i_img != e_img; ++i_img, ++i_gt) {
        *i_img = ((*i_img) < thresh) ? 0 : 1;
        *i_gt = ((*i_gt) < thresh) ? 0 : 1;

        intersection += ((*i_gt == 1) && (*i_img == 1));
        unions += ((*i_img == 1) + (*i_gt == 1));
    }

    float dice = (2 * intersection + eps_) / (unions + eps_);
    metric_list_.push_back(dice);

    return dice;
}
