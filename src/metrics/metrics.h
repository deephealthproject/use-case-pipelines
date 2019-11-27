#include "ecvl/core.h"

class Eval {
public:
    void ResetEval();
    std::vector<float> iou_list_;
    float eps_;
    float MIoU();
    float Eval::BinaryIoU(ecvl::Image& img, ecvl::Image& gt);

    Eval() 
    {
        eps_ = 1e-06;
    };
};