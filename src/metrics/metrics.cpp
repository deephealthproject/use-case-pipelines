#include "metrics.h"

using namespace ecvl;

void Eval::ResetEval()
{
    iou_list_.clear();
    eps_ = 1e-06;
}

float Eval::MIoU()
{
    return std::accumulate(iou_list_.begin(), iou_list_.end(), 0.0) / iou_list_.size();
}

float Eval::BinaryIoU(Image& img, Image& gt)
{
    float intersection = 0;
    float unions = 0;

    auto i_img = img.ContiguousBegin<float>(), e_img = img.ContiguousEnd<float>();
    auto i_gt = gt.ContiguousBegin<float>();

    for (; i_img != e_img; ++i_img, ++i_gt) {
        *i_img = ((*i_img) < 0.5) ? 0 : 1;

        intersection += ((*i_gt == 1) && (*i_img == *i_gt));
        unions += ((*i_gt == 1) || (*i_img == 1));
    }

    float iou = (intersection + eps_) / (unions + eps_);
    iou_list_.push_back(iou);

    return iou;
}