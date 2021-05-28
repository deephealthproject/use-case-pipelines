#include "metrics/metrics.h"
#include "utils/utils.h"

#include <iostream>

#include "ecvl/core/filesystem.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace ecvl;
using namespace ecvl::filesystem;
using namespace eddl;
using namespace std;

int main(int argc, char* argv[])
{
    // Default settings, they can be changed from command line
    Settings s(1, { 224,224 }, "UNetWithPaddingBN", "binary_cross_entropy", 0.001f, "skin_lesion_segmentation", "", 100, 2, 6, 6, { 1 });
    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
    }
    constexpr float lr_step = 0.1f; // step for the learning rate scheduler

    // Build model
    build(s.net,
        adam(s.lr),      // Optimizer
        { s.loss },      // Loss
        { "dice" },      // Metric
        s.cs,            // Computing Service
        s.random_weights // Randomly initialize network weights
    );
    layer out = getOut(s.net)[0];

    // View model
    summary(s.net);
    plot(s.net, s.exp_name + ".pdf");
    setlogfile(s.net, s.exp_name);

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

    // Replace the random seed with a fixed one to have reproducible experiments
    // AugmentationParam::SetSeed(50);

    DatasetAugmentations dataset_augmentations{ { training_augs, validation_augs, validation_augs } }; // use the same augmentations for validation and test

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d(s.dataset_path, s.batch_size, dataset_augmentations, ColorType::RGB, ColorType::GRAY, s.workers, s.queue_ratio, { true, false });

    // int num_batches_training = d.GetNumBatches("training");  // or
    // int num_batches_training = d.GetNumBatches(0);           // where 0 is the split index, or
    int num_batches_training = d.GetNumBatches(SplitType::training);
    int num_batches_validation = d.GetNumBatches(SplitType::validation);
    int num_batches_test = d.GetNumBatches(SplitType::test);

    float best_metric = 0.f, mean_metric;
    vector<vector<Point2i>> contours;
    Image labels, tmp;
    View<DataType::float32> pred_t, target_t, img_t;
    Eval evaluator;
    ofstream of;
    cv::TickMeter tm, tm_epoch;

    if (!s.skip_train) {
        cout << "Starting training" << endl;
        unsigned long long it = 0; // iteration counter for the learning rate scheduler
        for (int e = s.resume; e < s.epochs; ++e) {
            tm_epoch.reset();
            tm_epoch.start();
            d.SetSplit(SplitType::training);
            auto current_path{ s.result_dir / ("Epoch_" + to_string(e)) };

            // Reset errors for train_batch
            reset_loss(s.net);

            // Resize to batch size if we have done a previous resize
            if (d.split_[d.current_split_].last_batch_ != s.batch_size) {
                s.net->resize(s.batch_size);
            }

            // Reset and shuffle training list
            d.ResetBatch(d.current_split_, true);

            d.Start();
            // Feed batches to the model
            for (int j = 0; j < num_batches_training; ++j) {
                tm.reset();
                tm.start();
                cout << "Epoch " << e << "/" << s.epochs - 1 << " (batch " << j << "/" << num_batches_training - 1 << ") - ";
                cout << "|fifo| " << d.GetQueueSize() << " - ";

                // Load a batch
                auto [x, y] = d.GetBatch();

                // Check input images
                //for (int ind = 0; ind < s.batch_size; ++ind) {
                //    {
                //        unique_ptr<Tensor> tmp(x->select({ to_string(ind), ":", ":", ":" }));
                //        //tmp->normalize_(0.f, 1.f);
                //        tmp->clamp_(0.f, 1.f);
                //        tmp->mult_(255.f);
                //        tmp->save("../images/train_image_" + to_string(j) + "_" + to_string(ind) + ".png");
                //    }
                //    {
                //        unique_ptr<Tensor> tmp(y->select({ to_string(ind), ":", ":", ":" }));
                //        //tmp->normalize_(0.f, 1.f);
                //        tmp->clamp_(0.f, 1.f);
                //        tmp->mult_(255.f);
                //        tmp->save("../images/train_gt_" + to_string(j) + "_" + to_string(ind) + ".png");
                //    }
                //}

                auto current_bs = x->shape[0];
                // if it's the last batch and the number of samples doesn't fit the batch size, resize the network
                if (j == num_batches_validation - 1 && current_bs != s.batch_size) {
                    s.net->resize(current_bs);
                }

                // Train batch
                train_batch(s.net, { x.get() }, { y.get() });

                // Print errors
                print_loss(s.net, j);

                tm.stop();
                cout << "- Elapsed time: " << tm.getTimeSec() << endl;
                ++it;
            }
            d.Stop();

            // Validation
            cout << "Starting validation:" << endl;
            // Resize to batch size if we have done a previous resize
            if (d.split_[d.current_split_].last_batch_ != s.batch_size) {
                s.net->resize(s.batch_size);
            }
            d.SetSplit(SplitType::validation);
            d.ResetBatch(d.current_split_);
            evaluator.ResetEval();

            d.Start();
            for (int j = 0, n = 0; j < num_batches_validation; ++j) {
                cout << "Validation: Epoch " << e << "/" << s.epochs - 1 << " (batch " << j << "/" << num_batches_validation - 1 << ") - ";
                cout << "|fifo| " << d.GetQueueSize() << " - ";

                // Load a batch
                auto [x, y] = d.GetBatch();

                auto current_bs = x->shape[0];
                // if it's the last batch and the number of samples doesn't fit the batch size, resize the network
                if (j == num_batches_validation - 1 && current_bs != s.batch_size) {
                    s.net->resize(current_bs);
                }

                // Evaluate batch
                forward(s.net, { x.get() }); // forward does not require reset_loss
                unique_ptr<Tensor> output(getOutput(out));

                // Compute IoU metric and optionally save the output images
                for (int k = 0; k < current_bs; ++k, ++n) {
                    unique_ptr<Tensor> pred(output->select({ to_string(k) }));
                    TensorToView(pred.get(), pred_t);
                    unique_ptr<Tensor> target(y->select({ to_string(k) }));
                    TensorToView(target.get(), target_t);

                    cout << " - IoU: " << evaluator.BinaryIoU(pred_t, target_t);

                    if (s.save_images) {
                        unique_ptr<Tensor> single_image(x->select({ to_string(k) }));
                        single_image->mult_(255.);
                        single_image->normalize_(0.f, 255.f);
                        TensorToView(single_image.get(), img_t);
                        img_t.colortype_ = ColorType::RGB;
                        img_t.channels_ = "xyc";

                        pred->mult_(255.);
                        pred_t.colortype_ = ColorType::GRAY;
                        pred_t.channels_ = "xyc";
                        ConvertTo(pred_t, tmp, DataType::uint8);
                        ConnectedComponentsLabeling(tmp, labels);
                        ConvertTo(labels, tmp, DataType::uint8);
                        FindContours(tmp, contours);
                        ConvertTo(img_t, tmp, DataType::uint8);

                        for (auto& contour : contours) {
                            for (auto c : contour) {
                                *tmp.Ptr({ c[0], c[1], 0 }) = 0;
                                *tmp.Ptr({ c[0], c[1], 1 }) = 0;
                                *tmp.Ptr({ c[0], c[1], 2 }) = 255;
                            }
                        }

                        auto filename = "image_" + to_string(n) + ".png";
                        auto filename_gt = "gt_" + to_string(n) + ".png";
                        ImWrite(current_path / filename, tmp);

                        if (e == 0) {
                            target->mult_(255.);
                            target_t.colortype_ = ColorType::GRAY;
                            target_t.channels_ = "xyc";
                            ImWrite(s.result_dir / "Validation Ground Truth" / filename_gt, target_t);
                        }
                    }
                }
                cout << endl;
            }
            d.Stop();

            // Change the learning rate after 10'000 iterations
            if (it > 1e4) {
                s.lr *= lr_step;
                setlr(s.net, { s.lr });

                it = 0;
            }

            mean_metric = evaluator.MeanMetric();
            cout << "----------------------------------------" << endl;
            cout << "Epoch " << e << " - Mean validation IoU: " << mean_metric << endl;
            cout << "----------------------------------------" << endl;

            if (mean_metric > best_metric) {
                cout << "Saving weights..." << endl;
                save_net_to_onnx_file(s.net, (s.checkpoint_dir / (s.exp_name + "_epoch_" + to_string(e) + ".onnx")).string());
                best_metric = mean_metric;
            }

            of.open(s.exp_name + "_stats.txt", ios::out | ios::app);
            of << "Epoch " << e << " - Total validation IoU: " << mean_metric << endl;
            of.close();

            tm_epoch.stop();
            cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;
        }
    }

    //Test
    cout << "Starting test:" << endl;
    // Resize to batch size if we have done a previous resize
    if (d.split_[d.current_split_].last_batch_ != s.batch_size) {
        s.net->resize(s.batch_size);
    }
    d.SetSplit(SplitType::test);
    evaluator.ResetEval();

    string epoch = to_string(s.skip_train ? s.resume : s.epochs);
    auto current_path{ s.result_dir / ("Test - epoch " + epoch) };

    d.Start();
    for (int i = 0, n = 0; i < num_batches_test; ++i) {
        cout << "Test: (batch " << i << "/" << num_batches_test - 1 << ") - ";
        cout << "|fifo| " << d.GetQueueSize() << " - ";

        // Load a batch
        auto [x, y] = d.GetBatch();

        auto current_bs = x->shape[0];
        // if it's the last batch and the number of samples doesn't fit the batch size, resize the network
        if (i == num_batches_validation - 1 && current_bs != s.batch_size) {
            s.net->resize(current_bs);
        }

        // Evaluate batch
        forward(s.net, { x.get() }); // forward does not require reset_loss
        unique_ptr<Tensor> output(getOutput(out));

        // Compute IoU metric and optionally save the output images
        for (int j = 0; j < current_bs; ++j, ++n) {
            unique_ptr<Tensor> pred(output->select({ to_string(j) }));
            TensorToView(pred.get(), pred_t);
            unique_ptr<Tensor> target(y->select({ to_string(j) }));
            TensorToView(target.get(), target_t);

            cout << " - IoU: " << evaluator.BinaryIoU(pred_t, target_t);

            if (s.save_images) {
                unique_ptr<Tensor> single_image(x->select({ to_string(j) }));
                single_image->mult_(255.);
                single_image->normalize_(0.f, 255.f);
                TensorToView(single_image.get(), img_t);
                img_t.colortype_ = ColorType::RGB;
                img_t.channels_ = "xyc";

                pred->mult_(255.);
                pred_t.colortype_ = ColorType::GRAY;
                pred_t.channels_ = "xyc";
                ConvertTo(pred_t, tmp, DataType::uint8);
                ConnectedComponentsLabeling(tmp, labels);
                ConvertTo(labels, tmp, DataType::uint8);
                FindContours(tmp, contours);
                ConvertTo(img_t, tmp, DataType::uint8);

                for (auto& contour : contours) {
                    for (auto c : contour) {
                        *tmp.Ptr({ c[0], c[1], 0 }) = 0;
                        *tmp.Ptr({ c[0], c[1], 1 }) = 0;
                        *tmp.Ptr({ c[0], c[1], 2 }) = 255;
                    }
                }

                auto filename = "image_" + to_string(n) + ".png";
                auto filename_gt = "gt_" + to_string(n) + ".png";
                ImWrite(current_path / filename, tmp);

                target->mult_(255.);
                target_t.colortype_ = ColorType::GRAY;
                target_t.channels_ = "xyc";
                ImWrite(s.result_dir / "Test Ground Truth" / filename_gt, target_t);
            }
        }
        cout << endl;
    }
    d.Stop();

    mean_metric = evaluator.MeanMetric();
    cout << "-------------------------" << endl;
    cout << "Mean test IoU: " << mean_metric << endl;
    cout << "-------------------------" << endl;

    of.open(s.exp_name + "_stats.txt", ios::out | ios::app);
    of << "Epoch " << epoch << " - Total test IoU: " << mean_metric << endl;
    of.close();

    delete s.net;
    return EXIT_SUCCESS;
}