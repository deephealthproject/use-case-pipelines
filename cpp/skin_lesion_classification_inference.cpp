#include "data_generator/data_generator.h"
#include "models/models.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

#include "ecvl/core/filesystem.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace ecvl;
using namespace ecvl::filesystem;
using namespace eddl;
using namespace std;

int main()
{
    // Settings
    int batch_size = 50;
    bool save_images = true;
    path result_dir;

    // Define network
    model net = import_net_from_onnx_file("isic_classification_checkpoint_epoch_46.onnx");

    // Build model
    build(net,
        sgd(0.001f, 0.9f),          // Optimizer
        { "soft_cross_entropy" },   // Losses
        { "categorical_accuracy" }, // Metrics
        CS_GPU({ 1 }, 1, "low_mem"),  // Computing Service
        false                       // Randomly initialize network weights
    );

    // View model
    summary(net);
    plot(net, "model.pdf");
    setlogfile(net, "skin_lesion_classification_inference");

    // Set size from the size of the input layer of the network
    std::vector<int> size{ net->layers[0]->input->shape[2], net->layers[0]->input->shape[3] };

    auto test_augs = make_shared<SequentialAugmentationContainer>(AugResizeDim(size));
    DatasetAugmentations dataset_augmentations{ { nullptr, nullptr, test_augs } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d("D:/dataset/isic_classification/isic_classification.yml", batch_size, dataset_augmentations);

    if (save_images) {
        result_dir = "../output_images_classification_inference";
        create_directory(result_dir);
        for (const auto& c : d.classes_) {
            create_directories(result_dir / path(c));
        }
    }

    d.SetSplit(SplitType::test);
    int num_samples_test = vsize(d.GetSplit());
    int num_batches_test = num_samples_test / batch_size;
    DataGenerator d_generator(&d, batch_size, size, size, 5);

    float sum = 0., ca = 0., mean_metric;
    Tensor* output, * target, * result, * single_image;
    View<DataType::float32> img_t;
    vector<float> total_metric;
    Metric* m = getMetric("categorical_accuracy");
    d_generator.Start();

    // Test for each batch
    cout << "Starting test:" << endl;
    for (int i = 0, n = 0; d_generator.HasNext(); ++i) {
        cout << "Test: (batch " << i << "/" << num_batches_test - 1 << ") - ";
        cout << "|fifo| " << d_generator.Size() << " ";
        Tensor* x, * y;

        // Load a batch
        if (d_generator.PopBatch(x, y)) {
            // Preprocessing
            x->div_(255.0);

            forward(net, { x });
            output = getOutput(getOut(net)[0]);

            // Compute accuracy and optionally save the output images
            sum = 0.;
            for (int j = 0; j < batch_size; ++j, ++n) {
                result = output->select({ to_string(j) });
                target = y->select({ to_string(j) });

                ca = m->value(target, result);

                total_metric.push_back(ca);
                sum += ca;

                if (save_images) {
                    float max = std::numeric_limits<float>::min();
                    int classe = -1;
                    int gt_class = -1;
                    for (unsigned long k = 0; k < result->size; ++k) {
                        if (result->ptr[k] > max) {
                            max = result->ptr[k];
                            classe = k;
                        }

                        if (target->ptr[k] == 1.) {
                            gt_class = k;
                        }
                    }

                    single_image = x->select({ to_string(j) });
                    TensorToView(single_image, img_t);
                    img_t.colortype_ = ColorType::BGR;
                    single_image->mult_(255.);

                    path filename = d.samples_[d.GetSplit()[n]].location_[0].filename();

                    path cur_path = result_dir / d.classes_[classe] /
                        filename.replace_extension("_gt_class_" + to_string(gt_class) + ".png");
                    ImWrite(cur_path, img_t);
                    delete single_image;
                }

                delete result;
                delete target;
            }
            delete x;
            delete y;
            cout << "categorical_accuracy: " << static_cast<float>(sum) / batch_size << endl;
        }
    }
    d_generator.Stop();

    mean_metric = accumulate(total_metric.begin(), total_metric.end(), 0.0f) / total_metric.size();
    cout << "Total categorical accuracy: " << mean_metric << endl;

    ofstream of("output_classification_inference.txt");
    of << "Total categorical accuracy: " << mean_metric << endl;
    of.close();

    delete output;

    return EXIT_SUCCESS;
}