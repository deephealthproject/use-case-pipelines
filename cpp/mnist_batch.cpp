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
    // workers and queue_ratio will not be used because in this pipeline we use the "old" LoadBatch function
    // num_classes, size, model, loss, lr, exp_name, dataset_path, epochs, batch_size, workers, queue_ratio, gpus, input_channels 
    Settings s(10, { 28,28 }, "LeNet", "sce", 0.001f, "mnist_classification", "../data/mnist/mnist.yml", 5, 200, 1, 1, {}, 1);
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
    plot(s.net, s.exp_name + ".pdf");
    setlogfile(s.net, s.exp_name);

    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugRotate({ -5, 5 }),
        AugAdditivePoissonNoise({ 0, 10 }),
        AugGaussianBlur({ .0, .8 }),
        AugCoarseDropout({ 0, 0.3 }, { 0.02, 0.05 }, 0));

    DatasetAugmentations dataset_augmentations{ { training_augs, nullptr } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d(s.dataset_path, s.batch_size, dataset_augmentations, ColorType::GRAY);

    // Prepare tensors which store batch
    Tensor* x = new Tensor({ s.batch_size, d.n_channels_, s.size[0], s.size[1] });
    Tensor* y = new Tensor({ s.batch_size, static_cast<int>(d.classes_.size()) });

    // int num_batches_training = d.GetNumBatches("training");  // or
    // int num_batches_training = d.GetNumBatches(0);           // where 0 is the split index, or
    int num_batches_training = d.GetNumBatches(SplitType::training);
    int num_batches_test = d.GetNumBatches(SplitType::test);

    cv::TickMeter tm, tm_epoch;

    if (!s.skip_train) {
        d.SetSplit(SplitType::training);
        cout << "Starting training" << endl;
        for (int e = s.resume; e < s.epochs; ++e) {
            tm_epoch.reset();
            tm_epoch.start();

            // Reset errors
            reset_loss(s.net);

            // Reset and shuffle training list
            d.ResetBatch(d.current_split_, true);

            // Feed batches to the model
            for (int j = 0; j < num_batches_training; ++j) {
                tm.reset();
                tm.start();
                cout << "Epoch " << e << "/" << s.epochs << " (batch " << j << "/" << num_batches_training - 1 << ") - ";

                // Load a batch
                d.LoadBatch(x, y);

                // Preprocessing
                x->div_(255.0);

                // Train batch
                train_batch(s.net, { x }, { y });

                // Print errors
                print_loss(s.net, j);

                tm.stop();
                cout << "- Elapsed time: " << tm.getTimeSec() << endl;
            }

            tm_epoch.stop();
            cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;

            cout << "Saving weights..." << endl;
            save_net_to_onnx_file(s.net, (s.checkpoint_dir / (s.exp_name + "_epoch_" + to_string(e) + ".onnx")).string());
        }
    }

    // Test
    cout << "Starting test" << endl;
    d.SetSplit(SplitType::test);

    for (int i = 0; i < num_batches_test; ++i) {
        cout << "Test - (batch " << i << "/" << num_batches_test - 1 << ") - ";

        // Load a batch
        d.LoadBatch(x, y);

        // Preprocessing
        x->div_(255.0);

        // Evaluate batch
        evaluate(s.net, { x }, { y });
    }

    delete x;
    delete y;
    return EXIT_SUCCESS;
}