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
    constexpr int workers = 6;
    constexpr float lr_step = 0.1f;

    Settings s(1, { 224,224 }, "UNetWithPaddingBN", "binary_cross_entropy", 0.001f, "skin_lesion_segmentation");
    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
    }

    // Build model
    build(s.net,
        //        sgd(s.lr, 0.9f), // Optimizer
        adam(s.lr),                 // Optimizer
        { s.loss },                 // Loss
        { "dice" },                 // Metric
        s.cs,                       // Computing Service
        s.random_weights            // Randomly initialize network weights
    );

    // View model
    summary(s.net);
    plot(s.net, "model.pdf");
    setlogfile(s.net, "skin_lesion_segmentation");

    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size, InterpolationType::cubic),
        AugMirror(.5),
        AugFlip(.5),
        AugRotate({ -180, 180 }),
        AugAdditivePoissonNoise({ 0, 10 }),
        AugGammaContrast({ .5, 1.5 }),
        AugGaussianBlur({ .0, .8 }),
        AugCoarseDropout({ 0, 0.03 }, { 0.02, 0.05 }, 0.25),
        AugToFloat32(255, 255),
        AugNormalize({ 0.6681, 0.5301, 0.5247 }, { 0.1337, 0.1480, 0.1595 }) // isic stats
        //AugNormalize({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }) // imagenet stats
        );

    auto validation_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size, InterpolationType::cubic),
        AugToFloat32(255, 255),
        AugNormalize({ 0.6681, 0.5301, 0.5247 }, { 0.1337, 0.1480, 0.1595 }) // isic stats
        //AugNormalize({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }) // imagenet stats
        );

    DatasetAugmentations dataset_augmentations{ { training_augs, validation_augs, nullptr } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d(s.dataset_path, s.batch_size, dataset_augmentations, ColorType::RGB, ColorType::GRAY);
    // Create producer thread with 'DLDataset d' and 'std::queue q'
    int num_samples = vsize(d.GetSplit());
    int num_batches = num_samples / s.batch_size;
    DataGenerator d_generator_t(&d, s.batch_size, s.size, { vsize(d.classes_) }, workers);

    int num_samples_validation = vsize(d.GetSplit(SplitType::validation));
    int num_batches_validation = num_samples_validation / s.batch_size;

    View<DataType::float32> img_t, gt_t;
    Image orig_img_t, labels, tmp;
    vector<vector<Point2i>> contours;
    float best_metric = 0.;
    ofstream of;
    Eval evaluator;
    mt19937 g(random_device{}());

    cv::TickMeter tm;
    cv::TickMeter tm_epoch;
    layer out = getOut(s.net)[0];

    Tensor* x_val = new Tensor({ s.batch_size, d.n_channels_, s.size[0], s.size[1] });
    Tensor* y_val = new Tensor({ s.batch_size, 1, s.size[0], s.size[1] });

    cout << "Starting training" << endl;
    unsigned long long it = 0;
    for (int e = 0; e < s.epochs; ++e) {
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
            cout << "Epoch " << e << "/" << s.epochs - 1 << " (batch " << j << "/" << num_batches - 1 << ") - ";
            cout << "|fifo| " << d_generator_t.Size() << " - ";

            Tensor* x, * y;

            // Load a batch
            if (d_generator_t.PopBatch(x, y)) {
                // Train batch
                train_batch(s.net, { x }, { y });

                // Check input images
                //unique_ptr<Tensor> out(getOutput(out));
                //int ind = 0;
                //{
                //    unique_ptr<Tensor> tmp(y->select({ to_string(ind), ":", ":", ":" }));
                //    //tmp->normalize_(0.f, 1.f);
                //    tmp->clamp_(0.f, 1.f);
                //    tmp->mult_(255.f);
                //    tmp->save("images/train_gt_" + to_string(j) + "_" + to_string(ind) + ".png");
                //}
                //{
                //    unique_ptr<Tensor> tmp(x->select({ to_string(ind), ":", ":", ":" }));
                //    //tmp->normalize_(0.f, 1.f);
                //    tmp->clamp_(0.f, 1.f);
                //    tmp->mult_(255.f);
                //    tmp->save("images/train_image_" + to_string(j) + "_" + to_string(ind) + ".png");
                //}

                print_loss(s.net, j);

                delete x;
                delete y;
            }
            tm.stop();
            cout << "- Elapsed time: " << tm.getTimeSec() << endl;
            ++it;
        }

        // Change the learning rate after 10'000 iterations
        //if (it > 1e4) {
        //    s.lr *= lr_step;
        //    setlr(s.net, { s.lr, 0.9f });

        //    it = 0;
        //}

        d_generator_t.Stop();
        tm_epoch.stop();
        cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;

        // Validation
        d.SetSplit(SplitType::validation);
        evaluator.ResetEval();

        cout << "Starting validation:" << endl;
        for (int j = 0, n = 0; j < num_batches_validation; ++j) {
            cout << "Validation: Epoch " << e << "/" << s.epochs - 1 << " (batch " << j << "/" << num_batches_validation - 1
                << ")";

            // Load a batch
            d.LoadBatch(x_val, y_val);
            // Evaluate batch
            forward(s.net, { x_val }); // forward does not require reset_loss

            unique_ptr<Tensor> output(getOutput(out));

            // Compute IoU metric and optionally save the output images
            for (int k = 0; k < s.batch_size; ++k, ++n) {
                Tensor* img = output->select({ to_string(k) });
                TensorToView(img, img_t);
                img_t.colortype_ = ColorType::GRAY;
                img_t.channels_ = "xyc";

                Tensor* gt = y_val->select({ to_string(k) });
                TensorToView(gt, gt_t);
                gt_t.colortype_ = ColorType::GRAY;
                gt_t.channels_ = "xyc";

                cout << " - IoU: " << evaluator.BinaryIoU(img_t, gt_t);

                if (s.save_images) {
                    Tensor* orig_img = x_val->select({ to_string(k) });
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

                    if (e == 0) {
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

        float mean_metric = evaluator.MeanMetric();
        cout << "----------------------------" << endl;
        cout << "Validation MIoU: " << mean_metric << endl;
        cout << "----------------------------" << endl;

        if (mean_metric > best_metric) {
            cout << "Saving weights..." << endl;
            save_net_to_onnx_file(s.net, (s.checkpoint_dir / path(s.exp_name + "_epoch_" + to_string(e) + ".onnx")).string());
            best_metric = mean_metric;
        }

        of.open(s.exp_name + "_stats.txt", ios::out | ios::app);
        of << "Epoch " << e << " - MIoU: " << mean_metric << endl;
        of.close();
    }

    return EXIT_SUCCESS;
}