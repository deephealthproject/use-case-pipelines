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

// Custom LoadBatch for pneumothorax specific problem.
vector<path> PneumothoraxLoadBatch(DLDataset& d, tensor& images, tensor& labels, const vector<int>& mask_indices, const vector<int>& black_indices, int& m_i, int& b_i)
{
    int& bs = d.batch_size_;
    Image img, gt;
    int offset = 0, start = 0;
    vector<path> names;
    bool expr = true;
    static std::mt19937 g(std::random_device{}());

    // Move to next samples
    start = d.current_batch_[+d.current_split_] * bs;
    ++d.current_batch_[+d.current_split_];

    int index = 0;
    // Fill tensors with data
    for (int i = start, j = 0; i < start + bs; ++i, ++j) {
        if (d.current_split_ == SplitType::training) {
            // in training, check if you can take other black ground truth images..
            // b_i < mask_indices.size() * 0.25
            if (mask_indices.size() * 1.25 - i > mask_indices.size() - m_i) {
                // generate a random value between 0 and 1. With a 80% probability we take a sample with a ground truth with mask if there are still some available.
                auto prob = std::uniform_real_distribution<>(0, 1)(g);
                expr = prob >= 0.2 && m_i < mask_indices.size();
            }
            // ..otherwise, you have to take a ground truth with mask
        }
        else {
            // in validation, first take all the samples with the ground truth with mask and then all those with the black ground truth.
            expr = m_i < mask_indices.size();
        }

        if (expr) {
            index = mask_indices[m_i++];
        }
        else {
            index = black_indices[b_i++];
        }

        // insert the original name of images and ground truth in case you want to save predictions during validation
        Sample& elem = d.samples_[index];
        names.emplace_back(elem.location_[0]);
        names.emplace_back(elem.label_path_.value());

        // Read the image
        img = elem.LoadImage(d.ctype_, false);

        // Read the ground truth
        gt = elem.LoadImage(d.ctype_gt_, true);

        // Apply chain of augmentations to sample image and corresponding ground truth
        d.augs_.Apply(d.current_split_, img, gt);

        // Copy image into tensor (images)
        ImageToTensor(img, images, offset);

        // Copy label into tensor (labels)
        ImageToTensor(gt, labels, offset);

        ++offset;
    }

    return names;
}

int main(int argc, char** argv)
{
    cxxopts::Options options("DeepHealth pipeline pneumothorax training", "");

    options.add_options()
        ("e,epochs", "Number of training epochs", cxxopts::value<int>()->default_value("50"))
        ("b,batch_size", "Number of images for each batch", cxxopts::value<int>()->default_value("2"))
        ("n,num_classes", "Number of output classes", cxxopts::value<int>()->default_value("1"))
        ("save_images", "Save validation images or not", cxxopts::value<bool>()->default_value("false"))
        ("s,size", "Size to which resize the input images", cxxopts::value<vector<int>>()->default_value("512,512"))
        ("loss", "Loss function", cxxopts::value<string>()->default_value("cross_entropy"))
        ("l,learning_rate", "Learning rate", cxxopts::value<float>()->default_value("0.0001"))
        ("model", "Model of the network", cxxopts::value<string>()->default_value("SegNetBN"))
        ("g,gpu", "Which GPUs to use", cxxopts::value<vector<int>>()->default_value("1"))
        ("lsb", "How many batches are processed before synchronizing the model weights", cxxopts::value<int>()->default_value("1"))
        ("m,mem", "GPU memory usage configuration", cxxopts::value<string>()->default_value("low_mem"))
        ("r,result_dir", "Directory where the output images are stored", cxxopts::value<path>()->default_value("../output_images_pneumothorax"))
        ("checkpoint_dir", "Directory where the checkpoints are stored", cxxopts::value<path>()->default_value("../checkpoints_pneumothorax"))
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
    int epochs = result["epochs"].as<int>();
    int batch_size = result["batch_size"].as<int>();
    int num_classes = result["num_classes"].as<int>();
    bool save_images = result["save_images"].as<bool>();
    vector<int> size = result["size"].as<vector<int>>();
    string model = result["model"].as<string>();
    string loss = result["loss"].as<string>();
    float lr = result["learning_rate"].as<float>();
    vector<int> gpu;
    int lsb;
    string mem;
    bool random_weights = true;
    compserv cs;
    Net* net;

    path result_dir, checkpoint_dir, dataset_path;

    if (result.count("dataset_path")) {
        dataset_path = result["dataset_path"].as<path>();
    }
    else {
        cout << ECVL_ERROR_MSG "'d,dataset_path' is a required argument." << endl;
        return EXIT_FAILURE;
    }

    cout << "Options used: \n";
    cout << "epochs: " << epochs << "\n";
    cout << "batch_size: " << batch_size << "\n";
    cout << "model: " << model << "\n";
    cout << "loss: " << loss << "\n";
    cout << "learning rate: " << lr << "\n";
    cout << "num_classes: " << num_classes << "\n";
    cout << "size: (" << size[0] << ", " << size[1] << ")\n";

    checkpoint_dir = result["checkpoint_dir"].as<path>();
    create_directory(checkpoint_dir);
    cout << "dataset_path: " << dataset_path << "\n";
    cout << "checkpoint_dir: " << checkpoint_dir << "\n";

    if (result.count("checkpoint")) {
        random_weights = false;
        string checkpoint = result["checkpoint"].as<string>();
        net = import_net_from_onnx_file(checkpoint);
        cout << "pretrained weight: " << checkpoint << "\n";
    }
    else { // Define the network
        layer in, out, out_sigm;
        in = Input({ 1, size[0], size[1] });

        if (!model.compare("SegNetBN")) {
            out = SegNetBN(in, num_classes);
        }
        else if (!model.compare("UNetWithPaddingBN")) {
            out = UNetWithPaddingBN(in, num_classes);
        }
        else if (!model.compare("SegNet")) {
            out = SegNet(in, num_classes);
        }
        else if (!model.compare("UNetWithPadding")) {
            out = UNetWithPadding(in, num_classes);
        }
        else {
            cout << ECVL_ERROR_MSG << "You must specify one of these models: SegNetBN, UNetWithPaddingBN, SegNet, UNetWithPadding" << endl;
            return EXIT_FAILURE;
        }

        out_sigm = Sigmoid(out);
        net = Model({ in }, { out_sigm });
    }

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
        cout << "save_images: false\n" << endl;
    }

    // Build model
    build(net,
        adam(lr), // Optimizer
        { loss }, // Loss
        { "dice" }, // Metric
        cs,
        random_weights
    );

    // View model
    summary(net);
    plot(net, "model.pdf");
    setlogfile(net, "pneumothorax_segmentation");

    // Set augmentations for training and validation
    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(size, InterpolationType::nearest),
        AugMirror(.5),
        OneOfAugmentationContainer(
            0.3,
            AugGammaContrast({ 0,3 }),
            AugBrightness({ 0, 30 })
        ),
        OneOfAugmentationContainer(
            0.3,
            AugElasticTransform({ 30, 120 }, { 3, 6 }),
            AugGridDistortion({ 2, 5 }, { -0.3f, 0.3f }),
            AugOpticalDistortion({ -0.3f, 0.3f }, { -0.1f, 0.1f })
        ),
        AugRotate({ -30, 30 }));

    auto validation_augs = make_shared<SequentialAugmentationContainer>(AugResizeDim(size, InterpolationType::nearest));

    DatasetAugmentations dataset_augmentations{ { training_augs, validation_augs, nullptr } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    //Training split is set by default
    DLDataset d(dataset_path, batch_size, dataset_augmentations, ColorType::GRAY);

    // Prepare tensors which store batch
    tensor x = new Tensor({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y = new Tensor({ batch_size, d.n_channels_gt_, size[0], size[1] });

    // Retrieve indices of images with a black ground truth
    vector<int> total_indices(d.samples_.size());
    iota(total_indices.begin(), total_indices.end(), 0);
    vector<int> training_validation_test_indices(d.split_.training_);
    training_validation_test_indices.insert(training_validation_test_indices.end(), d.split_.test_.begin(), d.split_.test_.end());
    training_validation_test_indices.insert(training_validation_test_indices.end(), d.split_.validation_.begin(), d.split_.validation_.end());
    sort(training_validation_test_indices.begin(), training_validation_test_indices.end());
    vector<int> black;
    set_difference(total_indices.begin(), total_indices.end(), training_validation_test_indices.begin(), training_validation_test_indices.end(), std::inserter(black, black.begin()));

    // Get number of training samples. Add a 25% of training samples with black ground truth.
    int num_samples = static_cast<int>(d.GetSplit().size() * 1.25);
    int num_batches = num_samples / batch_size;

    d.SetSplit(SplitType::validation);

    // Get number of validation samples. Add a 25% of validation samples with black ground truth.
    int num_samples_validation = static_cast<int>(d.GetSplit().size() * 1.25);
    int num_batches_validation = num_samples_validation / batch_size;

    // Split indices of images with a black ground truth for training and validation
    vector<int> black_training(black.begin(), black.end() - (num_samples_validation - d.GetSplit().size()));
    vector<int> black_validation(black.end() - (num_samples_validation - d.GetSplit().size()), black.end());

    vector<int> indices(batch_size);
    iota(indices.begin(), indices.end(), 0);

    View<DataType::float32> pred_ecvl;
    View<DataType::float32> gt_ecvl;
    Image orig_img, orig_gt;
    ofstream of;
    Eval evaluator;
    cv::TickMeter tm;

    // Indices to track mask and black vector in PneumothoraxLoadBatch
    int m_i = 0, b_i = 0;

    cout << "Starting training" << endl;
    for (int i = 0; i < epochs; ++i) {
        d.ResetAllBatches();
        m_i = 0, b_i = 0;

        auto current_path{ result_dir / path("Epoch_" + to_string(i)) };
        if (save_images) {
            create_directory(current_path);
        }

        d.SetSplit(SplitType::training);

        // Reset errors
        reset_loss(net);

        // Shuffle training list
        shuffle(d.GetSplit().begin(), d.GetSplit().end(), g);
        shuffle(black_training.begin(), black_training.end(), g);

        // Feed batches to the model
        for (int j = 0; j < num_batches; ++j) {
            cout << "Epoch " << i << "/" << epochs << " (batch " << j << "/" << num_batches << ") - ";
            tm.reset();
            tm.start();

            // Load a batch
            PneumothoraxLoadBatch(d, x, y, d.GetSplit(), black_training, m_i, b_i);
            tm.stop();
            cout << "Load time: " << tm.getTimeSec() << " - ";
            tm.reset();
            tm.start();

            // Preprocessing
            x->div_(255.);
            y->div_(255.);

            // Train batch
            train_batch(net, { x }, { y }, indices);

            print_loss(net, j);
            tm.stop();
            cout << "Train time: " << tm.getTimeSec() << "\n";
        }

        cout << "Saving weights..." << endl;
        save_net_to_onnx_file(net, (checkpoint_dir / path("pneumothorax_model_" + model + "_loss_" + loss + "_lr_" + to_string(lr) + "_size_" + to_string(size[0]) + "_epoch_" + to_string(i) + ".onnx")).string());

        cout << "Starting validation:" << endl;
        d.SetSplit(SplitType::validation);

        evaluator.ResetEval();
        m_i = 0, b_i = 0;

        // Validation for each batch
        for (int j = 0; j < num_batches_validation; ++j) {
            cout << "Validation - Epoch " << i << "/" << epochs << " (batch " << j << "/" << num_batches_validation << ") " << endl;

            // Load a batch
            vector<path> names = PneumothoraxLoadBatch(d, x, y, d.GetSplit(), black_validation, m_i, b_i);

            // Preprocessing
            x->div_(255.);
            y->div_(255.);

            evaluate(net, { x }, { y });

            tensor output = getOutput(getOut(net)[0]);

            // Compute Dice metric and optionally save the output images
            for (int k = 0, n = 0; k < batch_size; ++k, n += 2) {
                tensor pred = output->select({ to_string(k) });
                TensorToView(pred, pred_ecvl);
                pred_ecvl.colortype_ = ColorType::GRAY;
                pred_ecvl.channels_ = "xyc";

                tensor gt = y->select({ to_string(k) });
                TensorToView(gt, gt_ecvl);
                gt_ecvl.colortype_ = ColorType::GRAY;
                gt_ecvl.channels_ = "xyc";

                cout << "Dice " << names[n].filename() << ": " << evaluator.DiceCoefficient(pred_ecvl, gt_ecvl) << " ";

                if (save_images) {
                    pred->mult_(255.);

                    // Save original image fused together with prediction (red mask) and ground truth (green mask)
                    ImRead(names[n], orig_img);
                    ImRead(names[n + 1], orig_gt, ImReadMode::GRAYSCALE);
                    ChangeColorSpace(orig_img, orig_img, ColorType::BGR);

                    ResizeDim(pred_ecvl, pred_ecvl, { orig_img.Width(), orig_img.Height() }, InterpolationType::nearest);

                    View<DataType::uint8> v_orig(orig_img);
                    auto i_pred = pred_ecvl.Begin();
                    auto i_gt = orig_gt.Begin<uint8_t>();

                    for (int c = 0; c < pred_ecvl.Width(); ++c) {
                        for (int r = 0; r < pred_ecvl.Height(); ++r, ++i_pred, ++i_gt) {
                            // Replace in the green channel of the original image pixels that are 255 in the ground truth mask
                            if (*i_gt == 255) {
                                v_orig({ r, c, 1 }) = 255;
                            }
                            // Replace in the red channel of the original image pixels that are 255 in the prediction mask
                            if (*i_pred == 255) {
                                v_orig({ r, c, 2 }) = 255;
                            }
                        }
                    }

                    ImWrite(current_path / names[n].filename().replace_extension(".png"), orig_img);
                }
                cout << endl;

                delete pred;
                delete gt;
            }
            cout << endl;
            delete output;
        }
        cout << "----------------------------" << endl;
        cout << "Mean Dice Coefficient: " << evaluator.MeanMetric() << endl;
        cout << "----------------------------" << endl;

        // Save metric values on file
        of.open("output_evaluate_pneumothorax_segmentation.txt", ios::out | ios::app);
        of << "Epoch " << i << " - Mean Dice Coefficient: " << evaluator.MeanMetric() << endl;
        of.close();
    }

    delete x;
    delete y;
    delete cs;

    return EXIT_SUCCESS;
}