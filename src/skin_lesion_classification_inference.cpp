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
    int batch_size = 12;
    int num_classes = 8;
    std::vector<int> size{ 224, 224 }; // Size of images

    bool save_images = true;
    path output_path;

    // Define network
    layer in = Input({ 3, size[0],  size[1] });
    layer out = VGG16(in, num_classes);
    model net = Model({ in }, { out });

    // Build model
    build(net,
        sgd(0.001f, 0.9f), // Optimizer
        { "soft_cross_entropy" }, // Losses
        { "categorical_accuracy" } // Metrics
    );

    toGPU(net);

    // View model
    summary(net);
    plot(net, "model.pdf");
    setlogfile(net, "skin_lesion_classification_inference");

    auto training_augs = make_unique<SequentialAugmentationContainer>(AugResizeDim(size));
    auto test_augs = make_unique<SequentialAugmentationContainer>(AugResizeDim(size));
    DatasetAugmentations dataset_augmentations{ {move(training_augs), nullptr, move(test_augs)} };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d("D:/dataset/isic_classification/isic_classification.yml", batch_size, move(dataset_augmentations));

    if (save_images) {
        output_path = "../output_images_classification_inference";
        create_directory(output_path);
        for (int c = 0; c < d.classes_.size(); ++c) {
            create_directories(output_path / path(d.classes_[c]));
        }
    }

    // Prepare tensors which store batch
    tensor x = eddlT::create({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y = eddlT::create({ batch_size, static_cast<int>(d.classes_.size()) });
    tensor output, target, result, single_image;

    d.SetSplit(SplitType::test);
    int num_samples_test = vsize(d.GetSplit());
    int num_batches_test = num_samples_test / batch_size;
    float sum = 0., ca = 0.;

    View<DataType::float32> img_t;

    vector<float> total_metric;
    Metric* m = getMetric("categorical_accuracy");

    load(net, "isic_class_VGG16_sgd_lr_0.001_momentum_0.9_loss_sce_size_224_epoch_48.bin");

    cout << "Starting test:" << endl;
    for (int i = 0, n = 0; i < num_batches_test; ++i) {
        cout << "Test: - (batch " << i << "/" << num_batches_test << ") - ";

        // Load a batch
        d.LoadBatch(x, y);

        // Preprocessing
        x->div_(255.0);

        forward(net, { x });
        output = getTensor(out);

        // Compute accuracy and optionally save the output images
        sum = 0.;
        for (int j = 0; j < batch_size; ++j, ++n) {
            result = eddlT::select(output, j);
            target = eddlT::select(y, j);

            ca = m->value(target, result);

            total_metric.push_back(ca);
            sum += ca;

            if (save_images) {
                float max = std::numeric_limits<float>::min();
                int classe = -1;
                int gt_class = -1;
                for (int i = 0; i < result->size; ++i) {
                    if (result->ptr[i] > max) {
                        max = result->ptr[i];
                        classe = i;
                    }

                    if (target->ptr[i] == 1.) {
                        gt_class = i;
                    }
                }

                single_image = eddlT::select(x, j);
                TensorToView(single_image, img_t);
                img_t.colortype_ = ColorType::BGR;
                single_image->mult_(255.);

                path filename = d.samples_[d.GetSplit()[n]].location_[0].filename();

                path cur_path = output_path / d.classes_[classe] / filename.replace_extension("_gt_class_" + to_string(gt_class) + ".png");
                ImWrite(cur_path, img_t);
            }

            delete result;
            delete target;
            delete single_image;
        }
        cout << "categorical_accuracy: " << static_cast<float>(sum) / batch_size << endl;
    }

    float total_avg = accumulate(total_metric.begin(), total_metric.end(), 0.0f) / total_metric.size();
    cout << "Total categorical accuracy: " << total_avg << endl;

    ofstream of("output_classification_inference.txt");
    of << "Total categorical accuracy: " << total_avg << endl;
    of.close();

    delete x;
    delete y;
    delete output;

    return EXIT_SUCCESS;
}