#include "utils.h"

using namespace ecvl;

void NormalizeToUint8(const Image& src, Image& dst)
{
    Table1D<NormalizeToUint8Struct> table;
    table(src.elemtype_)(src, dst);
}

void TensorSqueeze(tensor& t)
{
    for (int i = 0; i < t->ndim; i++) {
        if (t->shape[i] == 1) {
            t->shape.erase(t->shape.begin() + i);
            t->ndim--;
            break;
        }
    }
}

void ImageSqueeze(Image& img)
{
    for (int i = 0; i < img.dims_.size(); i++) {
        if (img.dims_[i] == 1) {
            img.dims_.erase(img.dims_.begin() + i);
            img.strides_.erase(img.strides_.begin() + i);
            img.channels_.erase(std::find(img.channels_.begin(), img.channels_.end(), 'z'));
            break;
        }
    }
}