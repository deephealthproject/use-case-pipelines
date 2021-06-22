#include "models/models.h"
#include "utils/utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <cmath>

#include "ecvl/core/filesystem.h"
#include "eddl/serialization/onnx/eddl_onnx.h"
#include "eddl/losses/loss.h"

using namespace ecvl;
using namespace ecvl::filesystem;
using namespace eddl;
using namespace std;

int main(int argc, char* argv[])
{
    // Download pretrained resnet50 created with pytorch
    if (!filesystem::exists("resnet50_pytorch_imagenet.onnx")) {
        system("curl -O -J -L \"https://drive.google.com/u/1/uc?id=1jVVVgJcImHit9xkzxpu4I9Rho4Yh2k2H&export=download\"");
    }

    // Settings
    Settings s(8, { 224,224 }, "", "sce", 1e-5f, "skin_lesion_classification");
    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
    }

    // onnx resnet generated by pytorch
//    removeLayer(s.net, "Gemm_68"); // remove last Linear of resnet18
    removeLayer(s.net, "Gemm_174"); // remove last Linear of resnet50
//    auto top = getLayer(s.net, "Flatten_67"); // get flatten of resnet18
    auto top = getLayer(s.net, "Flatten_173"); // get flatten of resnet50

    layer out = Softmax(Dense(top, s.num_classes, true, "classifier")); // true is for the bias.
    auto data_input = getLayer(s.net, "input"); // input of pytorch onnxes

    s.net = Model({ data_input }, { out });

    build(s.net,
        //sgd(s.lr, s.momentum),      // Optimizer
        adam(s.lr),      // Optimizer
        { s.loss },                 // Loss
        { "accuracy" }, // Metric
        s.cs,                       // Computing Service
        false
    );

    initializeLayer(s.net, "classifier");
    //for (auto l : s.net->layers) {
    //    if (l->name != "classifier")
    //        setTrainable(s.net, l->name, false);
    //}

    // View model
    summary(s.net);
    setlogfile(s.net, s.exp_name);

    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size, InterpolationType::cubic),
        AugMirror(.5),
        AugFlip(.5),
        AugRotate({ -180, 180 }),
        AugAdditivePoissonNoise({ 0, 10 }),
        AugGammaContrast({ .5, 1.5 }),
        AugGaussianBlur({ .0, .8 }),
        AugCoarseDropout({ 0, 0.03 }, { 0, 0.05 }, 0.25),
        AugToFloat32(255),
        //AugNormalize({ 0.6681, 0.5301, 0.5247 }, { 0.1337, 0.1480, 0.1595 }) // isic stats
        AugNormalize({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }) // imagenet stats
        );

    auto validation_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size, InterpolationType::cubic),
        AugToFloat32(255),
        //AugNormalize({ 0.6681, 0.5301, 0.5247 }, { 0.1337, 0.1480, 0.1595 }) // isic stats
        AugNormalize({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }) // imagenet stats
        );

    // Replace the random seed with a fixed one
    AugmentationParam::SetSeed(50);

    DatasetAugmentations dataset_augmentations{ { training_augs, validation_augs, validation_augs } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d(s.dataset_path, s.batch_size, dataset_augmentations, ColorType::RGB, ColorType::none, s.workers, s.queue_ratio, { true, false, false });

    int num_batches_training = d.GetNumBatches(SplitType::training);
    int num_batches_validation = d.GetNumBatches(SplitType::validation);
    //int num_batches_test = d.GetNumBatches(SplitType::test);

    Tensor* output, * target, * result, * single_image;
    float sum = 0.f, ca = 0.f, best_metric = 0.f, mean_metric;

    vector<float> total_metric;
    Metric* metric_fn = getMetric("accuracy");
    View<DataType::float32> img_t;
    ofstream of;
    mt19937 g(random_device{}());

    vector<int> indices(s.batch_size);
    iota(indices.begin(), indices.end(), 0);
    cv::TickMeter tm;
    cv::TickMeter tm_epoch;
    Tensor* x_val = new Tensor({ s.batch_size, d.n_channels_, s.size[0], s.size[1] });
    Tensor* y_val = new Tensor({ s.batch_size, static_cast<int>(d.classes_.size()) });
    cout << "Starting training" << endl;
    for (int i = 0; i < s.epochs; ++i) {
        tm_epoch.reset();
        tm_epoch.start();

        /*if (freeze && i >= frozen_epochs) {
            freeze = false;
            for (auto l : s.net->layers) {
                setTrainable(s.net, l->name, true);
            }
        }*/

        d.SetSplit(SplitType::training);
        auto current_path{ s.result_dir / path("Epoch_" + to_string(i)) };
        if (s.save_images) {
            for (const auto& c : d.classes_) {
                create_directories(current_path / path(c));
            }
        }

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
            cout << "Epoch " << i << "/" << s.epochs - 1 << " (batch " << j << "/" << num_batches_training - 1 << ") - ";
            cout << "|fifo| " << d.GetQueueSize() << " - ";

            // Load a batch
            auto [samples, x, y] = d.GetBatch();

            // Check input images
            //for (int ind = 0; ind < s.batch_size; ++ind) {
            //    unique_ptr<Tensor> tmp(x->select({ to_string(ind), ":", ":", ":" }));
            //    tmp->mult_(255.);
            //    tmp->normalize_(0.f, 255.f);
            //    tmp->save("../images/train_image_" + to_string(j) + "_" + to_string(ind) + ".png");
            //}

            // if it's the last batch and the number of samples doesn't fit the batch size, resize the network
            if (j == num_batches_training - 1 && x->shape[0] != s.batch_size) {
                s.net->resize(x->shape[0]);
            }

            // Train batch
            train_batch(s.net, { x.get() }, { y.get() });

            // Print errors
            print_loss(s.net, j);

            tm.stop();
            cout << "Elapsed time: " << tm.getTimeSec() << endl;
        }
        d.Stop();
        tm_epoch.stop();
        cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;

        // Validation
        d.SetSplit(SplitType::validation);
        cout << "Starting validation:" << endl;
        // Resize to batch size if we have done a previous resize
        if (d.split_[d.current_split_].last_batch_ != s.batch_size) {
            s.net->resize(s.batch_size);
        }
        d.Start();
        for (int j = 0, n = 0; j < num_batches_validation; ++j) {
            cout << "validation: Epoch " << i << '/' << s.epochs - 1 << " (batch " << j << "/" << num_batches_validation - 1 << ") - ";
            cout << "|fifo| " << d.GetQueueSize() << " - ";

            // Load a batch
            auto [samples, x, y] = d.GetBatch();

            auto current_bs = x->shape[0];
            // if it's the last batch and the number of samples doesn't fit the batch size, resize the network
            if (j == num_batches_validation - 1 && current_bs != s.batch_size) {
                s.net->resize(current_bs);
            }

            // Evaluate batch
            forward(s.net, { x_val }); // forward does not require reset_loss
            output = getOutput(out);
            ca = metric_fn->value(y_val, output);

            total_metric.push_back(ca);
            if (s.save_images) {
                for (int k = 0; k < s.batch_size; ++k, ++n) {
                    result = output->select({ to_string(k) });
                    target = y_val->select({ to_string(k) });
                    //result->toGPU();
                    //target->toGPU();
                    float max = std::numeric_limits<float>::min();
                    int classe = -1;
                    int gt_class = -1;
                    for (unsigned c = 0; c < result->size; ++c) {
                        if (result->ptr[c] > max) {
                            max = result->ptr[c];
                            classe = c;
                        }

                        if (target->ptr[c] == 1.) {
                            gt_class = c;
                        }
                    }

                    single_image = x_val->select({ to_string(k) });
                    TensorToView(single_image, img_t);
                    img_t.colortype_ = ColorType::BGR;
                    single_image->mult_(255.);

                    path filename = d.samples_[d.GetSplit()[n]].location_[0].filename();

                    path cur_path = current_path / d.classes_[classe] /
                        filename.replace_extension("_gt_class_" + to_string(gt_class) + ".png");
                    ImWrite(cur_path, img_t);
                    delete single_image;
                    delete result;
                    delete target;
                }
            }
            cout << " categorical_accuracy: " << ca / s.batch_size << endl;

            delete output;
        }
        d.Stop();

        mean_metric = accumulate(total_metric.begin(), total_metric.end(), 0.0f) / (total_metric.size() * s.batch_size);
        cout << "Validation categorical accuracy: " << mean_metric << endl;

        if (mean_metric > best_metric) {
            cout << "Saving weights..." << endl;
            save_net_to_onnx_file(s.net, (s.checkpoint_dir / path(s.exp_name + "_epoch_" + to_string(i) + ".onnx")).string());
            best_metric = mean_metric;
        }

        of.open(s.exp_name + "_stats.txt", ios::out | ios::app);
        of << "Epoch " << i << " - Total categorical accuracy: " << mean_metric << endl;
        of.close();
    }

    delete x_val;
    delete y_val;
    return EXIT_SUCCESS;
}