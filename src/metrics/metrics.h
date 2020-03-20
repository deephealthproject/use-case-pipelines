#ifndef METRICS_H_
#define METRICS_H_

#include "ecvl/core.h"

class Eval
{
public:
    std::vector<float> metric_list_;
    const float eps_ = 1e-06;
    void ResetEval();
    float MeanMetric();
    float BinaryIoU(ecvl::Image &img, ecvl::Image &gt, float thresh = 0.5);
    float DiceCoefficient(ecvl::Image &img, ecvl::Image &gt, float thresh = 0.5);

    Eval() {}
};

#endif // METRICS_H_