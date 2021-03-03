#include "data_generator/data_generator.h"
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
    Settings s(8, { 224,224 }, "ResNet50", "sce", 0.0001f, 0.9f);
    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
    }
    int workers = 2;


    // onnx resnet50
    //removeLayer(s.net, "resnetv17_dense0_fwd");
    removeLayer(s.net, "resnetv25_dense0_fwd");
    //auto top = getLayer(s.net, "flatten_473");
    auto top = getLayer(s.net, "resnetv25_flatten0_reshape0");
    layer out = Softmax(Dense(top, s.num_classes, true, "classifier")); // true is for the bias.
    auto data_input = getLayer(s.net, "data");
    s.net = Model({ data_input }, { out });

    // Build model
    build(s.net,
        adam(s.lr),      // Optimizer
        { s.loss },                 // Loss
        { "categorical_accuracy" }, // Metric
        s.cs,                       // Computing Service
        s.random_weights            // Randomly initialize network weights
    );

    initializeLayer(s.net, "classifier");
    for (auto l : s.net->layers) {
        if (l->name != "classifier")
            setTrainable(s.net, l->name, false);
    }
    auto trainable = false;

    // View model
    summary(s.net);
    plot(s.net, "model.pdf");
    setlogfile(s.net, "skin_lesion_classification");

    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size),
        AugMirror(.5),
        AugFlip(.5),
        AugRotate({ -180, 180 }),
        AugAdditivePoissonNoise({ 0, 10 }),
        AugGammaContrast({ .5, 1.5 }),
        AugGaussianBlur({ .0, .8 }),
        AugCoarseDropout({ 0, 0.03 }, { 0, 0.05 }, 0.25)
        );

    auto validation_augs = make_shared<SequentialAugmentationContainer>(AugResizeDim(s.size));

    DatasetAugmentations dataset_augmentations{ { training_augs, validation_augs, nullptr } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d(s.dataset_path, s.batch_size, dataset_augmentations, ecvl::ColorType::RGB);
    // Create producer thread with 'DLDataset d' and 'std::queue q'
    int num_samples = vsize(d.GetSplit());
    int num_batches = num_samples / s.batch_size;
    DataGenerator d_generator_t(&d, s.batch_size, s.size, { vsize(d.classes_) }, workers);

    d.SetSplit(SplitType::validation);
    int num_samples_validation = vsize(d.GetSplit());
    int num_batches_validation = num_samples_validation / s.batch_size;
    DataGenerator d_generator_v(&d, s.batch_size, s.size, { vsize(d.classes_) }, workers);

    tensor output, target, result, single_image;
    float sum = 0., ca = 0., best_metric = 0., mean_metric;

    vector<float> total_metric;
    Metric* m = getMetric("categorical_accuracy");
    View<DataType::float32> img_t;
    ofstream of;
    mt19937 g(random_device{}());

    vector<int> indices(s.batch_size);
    iota(indices.begin(), indices.end(), 0);
    cv::TickMeter tm;
    cv::TickMeter tm_epoch;

    cout << "Starting training" << endl;
    for (int i = 0; i < s.epochs; ++i) {
        tm_epoch.reset();
        tm_epoch.start();

        if (!trainable && i > 4) {
            trainable = true;
            for (auto l : s.net->layers) {
                setTrainable(s.net, l->name, true);
            }
        }

        auto current_path{ s.result_dir / path("Epoch_" + to_string(i)) };
        if (s.save_images) {
            for (const auto& c : d.classes_) {
                create_directories(current_path / path(c));
            }
        }

        d.SetSplit(SplitType::training);
        // Reset errors
        reset_loss(s.net);
        total_metric.clear();

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
                x->div_(255.0);

                // Train batch
                train_batch(s.net, { x }, { y }, indices);

                // Print errors
                print_loss(s.net, j);

                delete x;
                delete y;
            }
            tm.stop();

            cout << "Elapsed time: " << tm.getTimeSec() << endl;
        }

        d_generator_t.Stop();
        tm_epoch.stop();
        cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;

        // Validation
        d.SetSplit(SplitType::validation);
        d_generator_v.Start();

        cout << "Starting validation:" << endl;
        for (int j = 0, n = 0; d_generator_v.HasNext(); ++j) {
            cout << "Validation: Epoch " << i << "/" << s.epochs - 1 << " (batch " << j << "/" << num_batches_validation - 1
                << ") - ";

            tensor x, y;

            // Load a batch
            if (d_generator_v.PopBatch(x, y)) {
                // Preprocessing
                x->div_(255.0);

                // Evaluate batch
                forward(s.net, { x });
                output = getOutput(getOut(s.net)[0]);

                sum = 0.;
                for (int k = 0; k < s.batch_size; ++k, ++n) {
                    result = output->select({ to_string(k) });
                    target = y->select({ to_string(k) });

                    ca = m->value(target, result);

                    total_metric.push_back(ca);
                    sum += ca;

                    if (s.save_images) {
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

                        single_image = x->select({ to_string(k) });
                        TensorToView(single_image, img_t);
                        img_t.colortype_ = ColorType::BGR;
                        single_image->mult_(255.);

                        path filename = d.samples_[d.GetSplit()[n]].location_[0].filename();

                        path cur_path = current_path / d.classes_[classe] /
                            filename.replace_extension("_gt_class_" + to_string(gt_class) + ".png");
                        ImWrite(cur_path, img_t);
                        delete single_image;
                    }

                    delete result;
                    delete target;
                }
                cout << " categorical_accuracy: " << static_cast<float>(sum) / s.batch_size << endl;

                delete x;
                delete y;
            }
        }

        d_generator_v.Stop();

        mean_metric = accumulate(total_metric.begin(), total_metric.end(), 0.0f) / total_metric.size();
        cout << "Validation categorical accuracy: " << mean_metric << endl;

        if (mean_metric > best_metric) {
            cout << "Saving weights..." << endl;
            save_net_to_onnx_file(s.net, (s.checkpoint_dir / path("isic_classification_checkpoint_epoch_" + to_string(i) + ".onnx")).string());
            best_metric = mean_metric;
        }

        of.open("output_evaluate_isic_classification.txt", ios::out | ios::app);
        of << "Epoch " << i << " - Total categorical accuracy: " << mean_metric << endl;
        of.close();
    }

    delete output;

    return EXIT_SUCCESS;
}