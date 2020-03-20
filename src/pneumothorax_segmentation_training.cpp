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

        if(d.current_split_ == SplitType::training) {
            // in training, check if you can take other black ground truth images..
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
        const Sample& elem = d.samples_[index];
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

int main()
{
    // Settings
    int epochs = 50;
    int batch_size = 2;
    int num_classes = 1;
    std::vector<int> size{ 512, 512 }; // Size of images

    std::mt19937 g(std::random_device{}());

    bool save_images = true;
    path output_path;

    if (save_images) {
        output_path = "../output_images_pneumothorax";
        create_directory(output_path);
    }

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

    // Set augmentations for training and validation
    auto training_augs = make_unique<SequentialAugmentationContainer>(
        AugResizeDim(size, InterpolationType::nearest),
        AugMirror(.5),
        AugRotate({ -10, 10 }));

    auto validation_augs = make_unique<SequentialAugmentationContainer>(AugResizeDim(size, InterpolationType::nearest));

    DatasetAugmentations dataset_augmentations{ {move(training_augs), move(validation_augs), nullptr } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    //Training split is set by default
    DLDataset d("/path/to/siim/pneumothorax.yml", batch_size, move(dataset_augmentations), ColorType::GRAY);

    // Prepare tensors which store batch
    tensor x = eddlT::create({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y = eddlT::create({ batch_size, d.n_channels_gt_, size[0], size[1] });

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
    int num_samples = d.GetSplit().size() * 1.25;
    int num_batches = num_samples / batch_size;

    d.SetSplit(SplitType::validation);

    // Get number of validation samples. Add a 25% of validation samples with black ground truth.
    int num_samples_validation = d.GetSplit().size() * 1.25;
    int num_batches_validation = num_samples_validation / batch_size;

    // Split indices of images with a black ground truth for training and validation
    vector<int> black_training(black.begin(), black.end() - (num_samples_validation - d.GetSplit().size()));
    vector<int> black_validation(black.end() - (num_samples_validation - d.GetSplit().size()), black.end());

    vector<int> indices(batch_size);
    iota(indices.begin(), indices.end(), 0);

    View<DataType::float32> img_t;
    View<DataType::float32> gt_t;
    Image orig_img_t, labels, tmp;
    vector<vector<Point2i>> contours;
    ofstream of;
    Eval evaluator;
    cv::TickMeter tm;

    // Indices to track mask and black vector in PneumothoraxLoadBatch
    int m_i = 0, b_i = 0;

    cout << "Starting training" << endl;
    for (int i = 0; i < epochs; ++i) {

        d.ResetAllBatches();
        m_i = 0, b_i = 0;

        auto current_path{ output_path / path("Epoch_" + to_string(i)) };
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

            // Preprocessing
            x->div_(255.);
            y->div_(255.);

            // Train batch
            train_batch(net, { x }, { y }, indices);

            print_loss(net, j);
            tm.stop();
            cout << "- Elapsed time: " << tm.getTimeSec() << endl;
        }

        cout << "Saving weights..." << endl;
        save(net, "pneumothorax_segnetBN_adam_lr_0.0001_loss_ce_size_512_epoch_" + to_string(i) + ".bin", "bin");

        cout << "Starting validation:" << endl;
        d.SetSplit(SplitType::validation);

        evaluator.ResetEval();
        m_i = 0, b_i = 0;

        // Validation for each batch
        for (int j = 0; j < num_batches_validation; ++j) {
            cout << "Validation - Epoch " << i << "/" << epochs << " (batch " << j << "/" << num_batches_validation << ") ";

            // Load a batch
            vector<path> names = PneumothoraxLoadBatch(d, x, y, d.GetSplit(), black_validation, m_i, b_i);

            // Preprocessing
            x->div_(255.);
            y->div_(255.);

            forward(net, { x });
            tensor output = getTensor(out_sigm);

            // Compute Dice metric and optionally save the output images
            for (int k = 0, n = 0; k < batch_size; ++k, ++n) {
                tensor img = eddlT::select(output, k);
                TensorToView(img, img_t);
                img_t.colortype_ = ColorType::GRAY;
                img_t.channels_ = "xyc";

                tensor gt = eddlT::select(y, k);
                TensorToView(gt, gt_t);
                gt_t.colortype_ = ColorType::GRAY;
                gt_t.channels_ = "xyc";

                cout << "- Dice: " << evaluator.DiceCoefficient(img_t, gt_t) << " ";

                if (save_images) {
                    path filename = names[n].filename();
                    path filename_gt = names[++n].filename();

                    img->mult_(255.);
                    ImWrite(current_path / filename.replace_extension(".jpg"), img_t);

                    if (filename_gt.compare("black.png") == 0) {
                        continue;
                    }
                    gt->mult_(255.);
                    ImWrite(current_path / filename_gt, gt_t);
                }
            }
            cout << endl;
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

    return EXIT_SUCCESS;
}