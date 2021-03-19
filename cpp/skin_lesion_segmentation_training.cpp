#include "data_generator/data_generator.h"
#include "metrics/metrics.h"
#include "models/models.h"
#include "utils/utils.h"

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

int main(int argc, char* argv[])
{
    // Settings
    Settings s(1, { 192,192 }, "SegNetBN", "cross_entropy", 0.0001f);
    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
    }

    // Build model
    build(s.net,
        adam(s.lr),                 // Optimizer
        { s.loss },                 // Loss
        { "mean_squared_error" }, // Metric
        s.cs,                       // Computing Service
        s.random_weights            // Randomly initialize network weights
    );

    // View model
    summary(s.net);
    plot(s.net, "model.pdf");
    setlogfile(s.net, "skin_lesion_segmentation");

    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size),
        AugMirror(.5),
        AugFlip(.5),
        AugRotate({ -180, 180 }),
        AugAdditivePoissonNoise({ 0, 10 }),
        AugGammaContrast({ .5, 1.5 }),
        AugGaussianBlur({ .0, .8 }),
        AugCoarseDropout({ 0, 0.3 }, { 0.02, 0.05 }, 0.5));

    auto validation_augs = make_shared<SequentialAugmentationContainer>(AugResizeDim(s.size));

    DatasetAugmentations dataset_augmentations{ { training_augs, validation_augs, nullptr } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d(s.dataset_path, s.batch_size, dataset_augmentations);
    // Create producer thread with 'DLDataset d' and 'std::queue q'
    int num_samples = vsize(d.GetSplit());
    int num_batches = num_samples / s.batch_size;
    DataGenerator d_generator_t(&d, s.batch_size, s.size, { vsize(d.classes_) }, 5);

    d.SetSplit(SplitType::validation);
    int num_samples_validation = vsize(d.GetSplit());
    int num_batches_validation = num_samples_validation / s.batch_size;
    DataGenerator d_generator_v(&d, s.batch_size, s.size, { vsize(d.classes_) }, 5);

    View<DataType::float32> img_t, gt_t;
    Image orig_img_t, labels, tmp;
    vector<vector<Point2i>> contours;
    float best_metric = 0.;
    ofstream of;
    Eval evaluator;
    mt19937 g(random_device{}());

    vector<int> indices(s.batch_size);
    iota(indices.begin(), indices.end(), 0);
    cv::TickMeter tm;
    cv::TickMeter tm_epoch;

    cout << "Starting training" << endl;
    for (int i = 0; i < s.epochs; ++i) {
        tm_epoch.reset();
        tm_epoch.start();

        d.SetSplit(SplitType::training);
        // Reset errors
        reset_loss(s.net);

        // Shuffle training list
        shuffle(std::begin(d.GetSplit()), std::end(d.GetSplit()), g);
        d.ResetAllBatches();

        d_generator_t.Start();

        // Feed batches to the model
        for (int j = 0; d_generator_t.HasNext() /* j < num_batches */; ++j) {
            tm.reset();
            tm.start();
            cout << "Epoch " << i << "/" << s.epochs - 1 << " (batch " << j << "/" << num_batches - 1 << ") - ";
            cout << "|fifo| " << d_generator_t.Size() << " - ";

            tensor x, y;

            // Load a batch
            if (d_generator_t.PopBatch(x, y)) {
                // Preprocessing
                x->div_(255.);
                y->div_(255.);

                // Train batch
                train_batch(s.net, { x }, { y }, indices);

                print_loss(s.net, j);

                delete x;
                delete y;
            }
            tm.stop();

            cout << "- Elapsed time: " << tm.getTimeSec() << endl;
        }

        d_generator_t.Stop();
        tm_epoch.stop();
        cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;

        // Validation
        d.SetSplit(SplitType::validation);
        d_generator_v.Start();
        evaluator.ResetEval();

        cout << "Starting validation:" << endl;
        for (int j = 0, n = 0; d_generator_v.HasNext(); ++j) {
            cout << "Validation - Epoch " << i << "/" << s.epochs - 1 << " (batch " << j << "/" << num_batches_validation - 1
                << ") ";

            tensor x, y;

            // Load a batch
            if (d_generator_v.PopBatch(x, y)) {
                // Preprocessing
                x->div_(255.);
                y->div_(255.);

                // Evaluate batch
                forward(s.net, { x });
                tensor output = getOutput(getOut(s.net)[0]);

                // Compute IoU metric and optionally save the output images
                for (int k = 0; k < s.batch_size; ++k, ++n) {
                    tensor img = output->select({ to_string(k) });
                    TensorToView(img, img_t);
                    img_t.colortype_ = ColorType::GRAY;
                    img_t.channels_ = "xyc";

                    tensor gt = y->select({ to_string(k) });
                    TensorToView(gt, gt_t);
                    gt_t.colortype_ = ColorType::GRAY;
                    gt_t.channels_ = "xyc";

                    cout << "- IoU: " << evaluator.BinaryIoU(img_t, gt_t) << " ";

                    if (s.save_images) {
                        tensor orig_img = x->select({ to_string(k) });
                        orig_img->mult_(255.);
                        TensorToImage(orig_img, orig_img_t);
                        orig_img_t.colortype_ = ColorType::BGR;
                        orig_img_t.channels_ = "xyc";

                        img->mult_(255.);
                        CopyImage(img_t, tmp, DataType::uint8);
                        ConnectedComponentsLabeling(tmp, labels);
                        CopyImage(labels, tmp, DataType::uint8);
                        FindContours(tmp, contours);
                        CopyImage(orig_img_t, tmp, DataType::uint8);

                        for (auto& contour : contours) {
                            for (auto c : contour) {
                                *tmp.Ptr({ c[0], c[1], 0 }) = 0;
                                *tmp.Ptr({ c[0], c[1], 1 }) = 0;
                                *tmp.Ptr({ c[0], c[1], 2 }) = 255;
                            }
                        }

                        path filename = d.samples_[d.GetSplit()[n]].location_[0].filename();
                        path filename_gt = d.samples_[d.GetSplit()[n]].label_path_.value().filename();

                        ImWrite(s.result_dir / filename.replace_extension(".png"), tmp);

                        if (i == 0) {
                            gt->mult_(255.);
                            ImWrite(s.result_dir / filename_gt, gt_t);
                        }

                        delete orig_img;
                    }

                    delete img;
                    delete gt;
                }
                cout << endl;
            }
        }

        d_generator_v.Stop();
        float mean_metric = evaluator.MeanMetric();
        cout << "----------------------------" << endl;
        cout << "MIoU: " << mean_metric << endl;
        cout << "----------------------------" << endl;

        if (mean_metric > best_metric) {
            cout << "Saving weights..." << endl;
            save_net_to_onnx_file(s.net, (s.checkpoint_dir / path("isic_segmentation_checkpoint_epoch_" + to_string(i) + ".onnx")).string());
            best_metric = mean_metric;
        }

        of.open("output_evaluate_isic_segmentation.txt", ios::out | ios::app);
        of << "Epoch " << i << " - MIoU: " << evaluator.MeanMetric() << endl;
        of.close();
    }

    return EXIT_SUCCESS;
}