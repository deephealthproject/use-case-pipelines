#include "metrics/metrics.h"
#include "models/models.h"
#include "cxxopts.hpp"

#include "eddl/serialization/onnx/eddl_onnx.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

using namespace ecvl;
using namespace eddl;
using namespace std;
using namespace std::filesystem;

int main(int argc, char** argv)
{
    cxxopts::Options options("DeepHealth pipeline pneumothorax validation inference", "");

    options.add_options()
        ("b,batch_size", "Number of images for each batch", cxxopts::value<int>()->default_value("2"))
        ("n,num_classes", "Number of output classes", cxxopts::value<int>()->default_value("1"))
        ("save_images", "Save validation images or not", cxxopts::value<bool>()->default_value("false"))
        ("save_gt", "Save validation ground truth or not", cxxopts::value<bool>()->default_value("false"))
        ("g,gpu", "Which GPUs to use", cxxopts::value<vector<int>>()->default_value("1"))
        ("lsb", "How many batches are processed before synchronizing the model weights", cxxopts::value<int>()->default_value("1"))
        ("m,mem", "GPU memory usage configuration", cxxopts::value<string>()->default_value("low_mem"))
        ("r,result_dir", "Directory where the output images are stored", cxxopts::value<path>()->default_value("../output_images_pneumothorax"))
        ("t,gt_dir", "Directory where the ground_truth images are stored", cxxopts::value<path>()->default_value("../ground_truth_images_pneumothorax"))
        ("d,dataset_path", "Dataset path", cxxopts::value<path>())
        ("c,checkpoint", "Path to the onnx checkpoint file", cxxopts::value<string>())
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        cout << options.help() << endl;
        return EXIT_SUCCESS;
    }

    // Settings
    mt19937 g(std::random_device{}());
    int batch_size = result["batch_size"].as<int>();
    int num_classes = result["num_classes"].as<int>();
    bool save_images = result["save_images"].as<bool>();
    bool save_gt = result["save_gt"].as<bool>();
    vector<int> gpu;
    int lsb;
    string mem, checkpoint;
    compserv cs;
    Net* net;

    path result_dir, ground_truth_dir, dataset_path;

    if (result.count("dataset_path")) {
        dataset_path = result["dataset_path"].as<path>();
    }
    else {
        cout << ECVL_ERROR_MSG "'d,dataset_path' is a required argument." << endl;
        return EXIT_FAILURE;
    }

    // Import onnx file
    if (result.count("checkpoint")) {
        checkpoint = result["checkpoint"].as<string>();
        net = import_net_from_onnx_file(checkpoint);
    }
    else {
        cout << ECVL_ERROR_MSG "'c,checkpoint' is a required argument." << endl;
        return EXIT_FAILURE;
    }

    // Set size from the size of the input layer of the network
    vector<int> size{ net->layers[0]->input->shape[2], net->layers[0]->input->shape[3] };

    // Print a summary of all the options used
    cout << "Options used: \n";
    cout << "batch_size: " << batch_size << "\n";
    cout << "num_classes: " << num_classes << "\n";
    cout << "size: (" << size[0] << ", " << size[1] << ")\n";
    cout << "dataset_path: " << dataset_path << "\n";
    cout << "pretrained weight: " << checkpoint << "\n";

    if (result.count("gpu")) {
        gpu = result["gpu"].as<vector<int>>();
        lsb = result["lsb"].as<int>();
        mem = result["mem"].as<string>();
        cs = CS_GPU(gpu, lsb, mem);
        cout << "Model running on GPU: {";
        for (auto& x : gpu) {
            cout << x << ",";
        }
        cout << "}\n";
        cout << "lsb: " << lsb << "\n";
        cout << "mem: " << mem << "\n";
    }
    else {
        cs = CS_CPU();
        cout << "Model running on CPU\n";
    }

    if (save_images) {
        result_dir = result["result_dir"].as<path>();
        create_directory(result_dir);
        cout << "save_images: true\n";
        cout << "result output folder: " << result_dir << "\n";
    }
    else {
        cout << "save_images: false\n";
    }
    if (save_gt) {
        ground_truth_dir = result["gt_dir"].as<path>();
        create_directory(ground_truth_dir);
        cout << "save_ground_truth: true\n";
        cout << "ground_truth output folder: " << ground_truth_dir << "\n";
    }
    else {
        cout << "save_ground_truth: false\n";
    }
    cout << endl;

    // Build model
    build(net,
        adam(0.00001f), // Useless in inference but required by the constructor
        cs,
        false
    );

    // View model
    summary(net);
    plot(net, "model.pdf");
    setlogfile(net, "pneumothorax_segmentation");

    // Set augmentations for training and validation
    auto validation_augs = make_shared<SequentialAugmentationContainer>(AugResizeDim(size, InterpolationType::nearest));

    DatasetAugmentations dataset_augmentations{ { nullptr, validation_augs, nullptr } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d(dataset_path, batch_size, dataset_augmentations, ColorType::GRAY);

    // Prepare tensors which store batch
    tensor x = new Tensor({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y = new Tensor({ batch_size, d.n_channels_gt_, size[0], size[1] });
    tensor output;

    // Retrieve indices of images with a black ground truth
    vector<int> total_indices(d.samples_.size());
    iota(total_indices.begin(), total_indices.end(), 0);
    vector<int> training_validation_test_indices(d.split_.training_);
    training_validation_test_indices.insert(training_validation_test_indices.end(), d.split_.test_.begin(), d.split_.test_.end());
    training_validation_test_indices.insert(training_validation_test_indices.end(), d.split_.validation_.begin(), d.split_.validation_.end());
    sort(training_validation_test_indices.begin(), training_validation_test_indices.end());
    vector<int> black;
    set_difference(total_indices.begin(), total_indices.end(), training_validation_test_indices.begin(), training_validation_test_indices.end(), std::inserter(black, black.begin()));

    // Get number of validation samples.
    d.SetSplit(SplitType::validation);
    int num_samples_validation = static_cast<int>(d.GetSplit().size() * 1.25);
    int num_batches_validation = num_samples_validation / batch_size;

    vector<int> black_validation(black.end() - (num_samples_validation - d.GetSplit().size()), black.end());
    d.GetSplit().insert(d.split_.validation_.end(), black_validation.begin(), black_validation.end());

    View<DataType::float32> pred_ecvl;
    View<DataType::float32> gt_ecvl;
    Eval evaluator;
    cout << "Starting validation:" << endl;
    net->resize(batch_size);

    // Validation for each batch
    for (int i = 0, n = 0; i < num_batches_validation; ++i) {
        cout << "Validation - (batch " << i << "/" << num_batches_validation << ") " << endl;

        // Load a batch
        d.LoadBatch(x, y);

        // Preprocessing
        x->div_(255.);
        y->div_(255.);

        evaluate(net, { x }, { y });
        output = getOutput(getOut(net)[0]);

        // Save the output images
        for (int k = 0; k < batch_size; ++k, ++n) {
            tensor pred = output->select({ to_string(k) });
            TensorToView(pred, pred_ecvl);
            pred_ecvl.colortype_ = ColorType::GRAY;
            pred_ecvl.channels_ = "xyc";

            tensor gt = y->select({ to_string(k) });
            TensorToView(gt, gt_ecvl);
            gt_ecvl.colortype_ = ColorType::GRAY;
            gt_ecvl.channels_ = "xyc";

            path filename = d.samples_[d.GetSplit()[n]].location_[0].filename();

            if (save_images) {
                pred->mult_(255.);
                ImWrite(result_dir / filename.replace_extension(".png"), pred_ecvl);
                pred->div_(255.);
            }
            if (save_gt) {
                gt->mult_(255.);
                ImWrite(ground_truth_dir / filename.replace_extension(".png"), gt_ecvl);
                gt->div_(255.);
            }

            cout << "Dice " << filename << ": " << evaluator.DiceCoefficient(pred_ecvl, gt_ecvl) << endl;

            delete pred;
            delete gt;
        }
        cout << endl;
    }

    cout << "----------------------------" << endl;
    cout << "Mean Dice Coefficient: " << evaluator.MeanMetric() << endl;
    cout << "----------------------------" << endl;

    delete x;
    delete output;

    return EXIT_SUCCESS;
}