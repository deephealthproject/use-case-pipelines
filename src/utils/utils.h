#include "ecvl/core.h"
#include "ecvl/eddl.h"

template <ecvl::DataType DT> // src type
struct NormalizeToUint8Struct
{
    static void _(const ecvl::Image& src, ecvl::Image& dst)
    {
        dst.Create(src.dims_, DataType::uint8, src.channels_, src.colortype_, src.spacings_);

        ConstView<DT> src_v(src);
        View<DataType::uint8> dst_v(dst);

        // find max and min
        TypeInfo_t<DT> max = *std::max_element(src_v.Begin(), src_v.End());
        TypeInfo_t<DT> min = *std::min_element(src_v.Begin(), src_v.End());

        auto dst_it = dst_v.Begin();
        auto src_it = src_v.Begin();
        auto src_end = src_v.End();
        for (; src_it != src_end; ++src_it, ++dst_it) {
            (*dst_it) = (((*src_it) - min) * 255) / (max - min);
        }
    }
};

void NormalizeToUint8(const ecvl::Image& src, ecvl::Image& dst);
void TensorSqueeze(tensor& t);
void ImageSqueeze(ecvl::Image& img);