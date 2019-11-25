#include "ecvl/core.h"
#include "ecvl/eddl.h"
#include "ecvl/dataset_parser.h"
#include "models/models.h"

#include <algorithm>
#include <iostream>
#include <random>

using namespace ecvl;
using namespace eddl;
using namespace std;

template <DataType DT> // src type
struct NormalizeToUint8Struct {
    static void _(const Image& src, Image& dst)
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

void NormalizeToUint8(const Image& src, Image& dst)
{
    Table1D<NormalizeToUint8Struct> table;
    table(src.elemtype_)(src, dst);
}

void Squeeze(tensor& t)
{
    for (int i = 0; i < t->ndim; i++) {
        if (t->shape[i] == 1) {
            t->shape.erase(t->shape.begin() + i);
            t->ndim--;
            break;
        }
    }
}

int main()
{
    // Settings
    int epochs = 10;
    int batch_size = 1;
    int num_classes = 1;
    std::vector<int> size{ 512,512 }; // Size of images

    std::random_device rd;
    std::mt19937 g(rd());

    // Define network
    layer in = Input({ 3, size[0],  size[1] });
    //layer out = UNetWithPadding(in, num_classes);
    layer out = SegNet(in, num_classes);
    layer out_sigm = Sigmoid(out);
    model net = Model({ in }, { out_sigm });

    // Build model
    build(net,
        //sgd(0.001, 0.9), // Optimizer
        adam(0.0001), //Optimizer
        //{ "cross_entropy" }, // Losses
        { "cross_entropy" }, // Losses
        //{ "categorical_accuracy" }, // Metrics
        { "mean_squared_error" } // Metrics
    );

    toGPU(net);

    // View model
    summary(net);
    plot(net, "model.pdf");

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d("D:/dataset/isic_2017/isic_segmentation.yml", batch_size, "training");

    // Prepare tensors which store batch
    tensor x_train = eddlT::create({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y_train = eddlT::create({ batch_size, 1, size[0], size[1] });

    // Set batch size
    resize_model(net, batch_size);
    // Set training mode
    set_mode(net, TRMODE);

    int num_samples = d.GetSplit().size();
    int num_batches = num_samples / batch_size;

    vector<int> indices(batch_size);
    iota(indices.begin(), indices.end(), 0);

    cout << "Starting training" << endl;
    for (int i = 0; i < epochs; ++i) {
        // Reset errors
        reset_loss(net);

        // Shuffle training list
        shuffle(std::begin(d.GetSplit()), std::end(d.GetSplit()), g);
        d.current_batch_ = 0;

        // Feed batches to the model
        for (int j = 0; j < num_batches; ++j, ++d.current_batch_) {
            cout << "Epoch " << i + 1 << "/" << epochs << " (batch " << j + 1 << "/" << num_batches << ") - ";

            // Load a batch
            LoadBatch(d, size, x_train, y_train);

            // Preprocessing
            x_train->div_(255.);
            y_train->div_(255.);

            // Prepare data
            vtensor tx{ x_train };
            vtensor ty{ y_train };

            // Train batch
            //train_batch(net, tx, ty, indices);
            zeroGrads(net);
            forward(net, tx);
            net->compute_loss();

            backward(net, ty);
            update(net);

            print_loss(net, j);
            printf("\n");
            //forward(net, batch_size);

            if (j % 100 == 0) {
                tensor output = getTensor(out_sigm);
                tensor img = eddlT::select(output, 0);
                eddlT::save(img, "out_sigm.png", "png");
                tensor input = getTensor(in);
                img = eddlT::select(input, 0);
                //eddlT::reshape_(img, { 1,1,size[0], size[1] });
                eddlT::save(img, "out_sigm_in.png", "png");
            }
        }
    }

    save(net, "isic_segmentation_checkpoint.bin");
    delete x_train;
    delete y_train;

    return EXIT_SUCCESS;
}