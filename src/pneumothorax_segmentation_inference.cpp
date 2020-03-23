#include "metrics/metrics.h"
#include "models/models.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

using namespace ecvl;
using namespace eddl;
using namespace std;
using namespace std::filesystem;

int main()
{
    // Settings
    int batch_size = 2;
    int num_classes = 1;
    std::vector<int> size{ 512, 512 }; // Size of images

    path output_path("../output_images_pneumothorax_inference");
    create_directory(output_path);

    // Define network
    layer in = Input({ 1, size[0], size[1] });
    layer out = SegNetBN(in, num_classes);
    layer out_sigm = Sigmoid(out);
    model net = Model({ in }, { out_sigm });

    // Build model
    build(net,
        adam(0.0001), //Optimizer
        { "cross_entropy" }, // Losses
        { "mean_squared_error" } // Metrics
    );

    toGPU(net, "low_mem");

    // View model
    summary(net);
    plot(net, "model.pdf");
    setlogfile(net, "pneumothorax_segmentation");

    // Set augmentations for training and test
    auto training_augs = make_unique<SequentialAugmentationContainer>(AugResizeDim(size, InterpolationType::nearest));
    auto test_augs = make_unique<SequentialAugmentationContainer>(AugResizeDim(size, InterpolationType::nearest));

    DatasetAugmentations dataset_augmentations{ {move(training_augs), nullptr, move(test_augs) } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d("/path/to/siim/pneumothorax.yml", batch_size, move(dataset_augmentations), ColorType::GRAY);

    // Prepare tensors which store batch
    tensor x = eddlT::create({ batch_size, d.n_channels_, size[0], size[1] });
    tensor output;

    // Get number of test samples.
    d.SetSplit(SplitType::test);
    int num_samples_test = d.GetSplit().size();
    int num_batches_test = num_samples_test / batch_size;

    View<DataType::float32> img_t;
    load(net, "pneumothorax_segnetBN_adam_lr_0.0001_loss_ce_size_512_epoch_44.bin");
    cout << "Starting test:" << endl;

    // Test for each batch
    for (int i = 0, n = 0; i < num_batches_test; ++i) {
        cout << "Test - (batch " << i << "/" << num_batches_test << ") ";

        // Load a batch
        d.LoadBatch(x);

        // Preprocessing
        x->div_(255.);

        forward(net, { x });
        output = getTensor(out_sigm);

        // Save the output images
        for (int k = 0; k < batch_size; ++k, ++n) {
            tensor img = eddlT::select(output, k);
            TensorToView(img, img_t);
            img_t.colortype_ = ColorType::GRAY;
            img_t.channels_ = "xyc";

            auto i_img = img_t.ContiguousBegin<float>(), e_img = img_t.ContiguousEnd<float>();

            for (; i_img != e_img; ++i_img) {
                *i_img = ((*i_img) < 0.50) ? 0 : 1;
            }

            path filename = d.samples_[d.GetSplit()[n]].location_[0].filename();

            img->mult_(255.);
            ImWrite(output_path / filename.replace_extension(".png"), img_t);
        }
        cout << endl;
    }

    delete x;
    delete output;

    return EXIT_SUCCESS;
}