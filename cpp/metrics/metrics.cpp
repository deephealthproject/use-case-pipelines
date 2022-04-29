#include "metrics.h"

using namespace ecvl;

void Eval::ResetEval()
{
    metric_list_.clear();
}

double Eval::MeanMetric()
{
    return std::accumulate(metric_list_.begin(), metric_list_.end(), 0.0) / metric_list_.size();
}

double Eval::BinaryIoU(Image& img, Image& gt, float thresh)
{
    double intersection = 0;
    double unions = 0;

    auto i_img = img.ContiguousBegin<float>(), e_img = img.ContiguousEnd<float>();
    auto i_gt = gt.ContiguousBegin<float>();

    for (; i_img != e_img; ++i_img, ++i_gt) {
        *i_img = ((*i_img) < thresh) ? 0.f : 1.f;
        *i_gt = ((*i_gt) < thresh) ? 0.f : 1.f;

        intersection += ((*i_gt == 1) && (*i_img == 1));
        unions += ((*i_gt == 1) || (*i_img == 1));
    }

    double iou = (intersection + eps_) / (unions + eps_);
    metric_list_.push_back(iou);

    return iou;
}

double Eval::DiceCoefficient(Image& img, Image& gt, float thresh)
{
    double intersection = 0;
    double unions = 0;

    auto i_img = img.Begin<float>(), e_img = img.End<float>();
    auto i_gt = gt.Begin<float>();

    for (; i_img != e_img; ++i_img, ++i_gt) {
        *i_img = ((*i_img) < thresh) ? 0.f : 1.f;
        *i_gt = ((*i_gt) < thresh) ? 0.f : 1.f;

        intersection += ((*i_gt == 1) && (*i_img == 1));
        unions += ((*i_img == 1) + (*i_gt == 1));
    }

    double dice = (2 * intersection + eps_) / (unions + eps_);
    metric_list_.push_back(dice);

    return dice;
}