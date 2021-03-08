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
    Settings s(10, { 28,28 }, "LeNet", "mse", 0.0001f);
    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
    }

    // Build model
    build(s.net,
        adam(s.lr),                 // Optimizer
        { s.loss },                 // Loss
        { "categorical_accuracy" }, // Metric
        s.cs,                       // Computing Service
        s.random_weights            // Randomly initialize network weights
    );

    // View model
    summary(s.net);
    setlogfile(s.net, "mnist");

    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugRotate({ -5, 5 }),
        AugAdditivePoissonNoise({ 0, 10 }),
        AugGaussianBlur({ .0, .8 }),
        AugCoarseDropout({ 0, 0.3 }, { 0.02, 0.05 }, 0));

    DatasetAugmentations dataset_augmentations{ { move(training_augs), nullptr, nullptr } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d(s.dataset_path, s.batch_size, dataset_augmentations);
    // Create producer thread with 'DLDataset d' and 'std::queue q'
    int num_samples = vsize(d.GetSplit());
    int num_batches = num_samples / s.batch_size;
    DataGenerator d_generator_t(&d, s.batch_size, s.size, { vsize(d.classes_) }, 12);

    d.SetSplit(SplitType::test);
    int num_samples_validation = vsize(d.GetSplit());
    int num_batches_validation = num_samples_validation / s.batch_size;
    DataGenerator d_generator_v(&d, s.batch_size, s.size, { vsize(d.classes_) }, 12);

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
        d.SetSplit(SplitType::test);
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

                // Evaluate batch
                evaluate(s.net, { x }, { y });

                delete x;
                delete y;
            }
        }

        d_generator_v.Stop();
    }

    return EXIT_SUCCESS;
}