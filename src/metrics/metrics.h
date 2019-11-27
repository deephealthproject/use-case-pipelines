#ifndef METRICS_H_
#define METRICS_H_

#include "ecvl/core.h"

class Eval
{
public:
    std::vector<float> iou_list_;
    float eps_;
    void ResetEval();
    float MIoU();
    float BinaryIoU(ecvl::Image &img, ecvl::Image &gt);

    Eval() : eps_(1e-06) {}
};

#endif // METRICS_H_