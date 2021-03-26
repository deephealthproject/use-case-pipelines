#include "data_generator/data_generator.h"
#include "models/models.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

#include "ecvl/core/filesystem.h"
#include "eddl/serialization/onnx/eddl_onnx.h"
#include "cxxopts.hpp"

using namespace ecvl;
using namespace ecvl::filesystem;
using namespace eddl;
using namespace std;

int main(int argc, char* argv[])
{
    cxxopts::Options options("Skin lesion inference", "");
    options.add_options()
        ("d,dataset_path", "Dataset path", cxxopts::value<path>())
        ("c,checkpoint", "Path to the onnx checkpoint file", cxxopts::value<string>())
        ("b,batch_size", "Number of images for each batch", cxxopts::value<int>()->default_value("12"))
        ("save_images", "Save validation images or not", cxxopts::value<bool>()->default_value("false"));

    auto args = options.parse(argc, argv);

    // Settings
    path dataset_path = args["dataset_path"].as<path>();
    int batch_size = args["batch_size"].as<int>();
    bool save_images = args["save_images"].as<bool>();
    string checkpoint = args["checkpoint"].as<string>();
    path result_dir;

    // Define network
    model net = import_net_from_onnx_file(checkpoint);

    // Build model
    build(net,
        adam(0.00001f),               // Optimizer
        { "soft_cross_entropy" },     // Losses
        { "categorical_accuracy" },   // Metrics
        CS_GPU({ 1 }, "low_mem"),     // Computing Service
        false                         // Randomly initialize network weights
    );

    // View model
    /*summary(net);
    plot(net, "model.pdf");*/
    setlogfile(net, "skin_lesion_classification_inference");

    // Set size from the size of the input layer of the network
    std::vector<int> size{ net->layers[0]->input->shape[2], net->layers[0]->input->shape[3] };

    auto augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(size, InterpolationType::cubic),
        AugToFloat32(255),
        //AugNormalize({ 0.6681, 0.5301, 0.5247 }, { 0.1337, 0.1480, 0.1595 }) // isic stats
        AugNormalize({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }) // imagenet stats
        );
    DatasetAugmentations dataset_augmentations{ { nullptr, nullptr, augs } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d(dataset_path, batch_size, dataset_augmentations, ColorType::RGB);

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

    float sum = 0., ca = 0., mean_metric;
    Tensor* output, * target, * result, * single_image;
    View<DataType::float32> img_t;
    vector<float> total_metric;
    Metric* metric_fn = getMetric("accuracy");

    // Test for each batch
    cout << "Starting test:" << endl;
    Tensor* x_test = new Tensor({ batch_size, d.n_channels_, size[0], size[1] });
    Tensor* y_test = new Tensor({ batch_size, static_cast<int>(d.classes_.size()) });
    layer out = getOut(net)[0];

    for (int i = 0, n = 0; i < num_batches_test; ++i) {
        cout << "Test: (batch " << i << "/" << num_batches_test - 1 << ") - ";

        // Load a batch
        d.LoadBatch(x_test, y_test);
        forward(net, { x_test });
        output = getOutput(out);
        ca = metric_fn->value(y_test, output);

        total_metric.push_back(ca);

        // Compute accuracy and optionally save the output images
        if (save_images) {
            for (int j = 0; j < batch_size; ++j, ++n) {
                result = output->select({ to_string(j) });
                target = y_test->select({ to_string(j) });

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

                single_image = x_test->select({ to_string(j) });
                TensorToView(single_image, img_t);
                img_t.colortype_ = ColorType::BGR;
                single_image->mult_(255.);

                path filename = d.samples_[d.GetSplit()[n]].location_[0].filename();

                path cur_path = result_dir / d.classes_[classe] /
                    filename.replace_extension("_gt_class_" + to_string(gt_class) + ".png");
                ImWrite(cur_path, img_t);
                delete single_image;
                delete result;
                delete target;
            }
        }
        cout << "categorical_accuracy: " << ca / batch_size << endl;
        delete output;
    }

    mean_metric = accumulate(total_metric.begin(), total_metric.end(), 0.0f) / (total_metric.size() * batch_size);
    cout << "Total categorical accuracy: " << mean_metric << endl;

    ofstream of("output_classification_inference.txt", ios::out | ios::app);
    of << "Total categorical accuracy: " << mean_metric << endl;
    of.close();

    delete x_test;
    delete y_test;
    return EXIT_SUCCESS;
}