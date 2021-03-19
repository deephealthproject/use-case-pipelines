#include "models/models.h"

#include <algorithm>
#include <iostream>
#include <random>

using namespace ecvl;
using namespace eddl;
using namespace std;

int main()
{
    // Settings
    int epochs = 5;
    int batch_size = 64;
    int num_classes = 10;
    std::vector<int> size{ 28,28 }; // Size of images

    ColorType ctype = ColorType::GRAY;
    std::mt19937 g(std::random_device{}());

    // Define network
    layer in = Input({ 1, size[0],  size[1] });
    layer out = LeNet(in, num_classes); // Model LeNet
    model net = Model({ in }, { out });

    // Build model
    build(net,
        sgd(0.001f, 0.9f), // Optimizer
        { "soft_cross_entropy" }, // Loss
        { "categorical_accuracy" } // Metric
    );

    toGPU(net);

    // View model
    summary(net);
    plot(net, "model.pdf");
    setlogfile(net, "mnist");

    auto training_augs = make_unique<SequentialAugmentationContainer>(
        AugRotate({ -5, 5 }),
        AugAdditivePoissonNoise({ 0, 10 }),
        AugGaussianBlur({ .0, .8 }),
        AugCoarseDropout({ 0, 0.3 }, { 0.02, 0.05 }, 0));

    DatasetAugmentations dataset_augmentations{ {move(training_augs), nullptr, nullptr} };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d("../data/mnist/mnist.yml", batch_size, move(dataset_augmentations), ctype);

    // Prepare tensors which store batch
    tensor x = new Tensor({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y = new Tensor({ batch_size, static_cast<int>(d.classes_.size()) });

    int num_samples = vsize(d.GetSplit());
    int num_batches = num_samples / batch_size;

    vector<int> indices(batch_size);
    iota(indices.begin(), indices.end(), 0);
    cv::TickMeter tm;

    cout << "Starting training" << endl;
    for (int i = 0; i < epochs; ++i) {
        // Reset errors
        reset_loss(net);

        // Shuffle training list
        shuffle(std::begin(d.GetSplit()), std::end(d.GetSplit()), g);
        d.ResetCurrentBatch();

        // Feed batches to the model
        for (int j = 0; j < num_batches; ++j) {
            tm.reset();
            tm.start();
            cout << "Epoch " << i << "/" << epochs << " (batch " << j << "/" << num_batches << ") - ";

            // Load a batch
            d.LoadBatch(x, y);

            // Preprocessing
            x->div_(255.0);

            // Prepare data
            vtensor tx{ x };
            vtensor ty{ y };

            // Train batch
            train_batch(net, tx, ty, indices);

            // Print errors
            print_loss(net, j);
            tm.stop();

            cout << "- Elapsed time: " << tm.getTimeSec() << endl;
        }
    }
    cout << "Saving weights..." << endl;
    save(net, "mnist_checkpoint.bin", "bin");

    // Evaluation
    d.SetSplit(SplitType::test);
    num_samples = vsize(d.GetSplit());
    num_batches = num_samples / batch_size;

    cout << "Evaluate test:" << endl;
    for (int i = 0; i < num_batches; ++i) {
        cout << "Batch " << i << "/" << num_batches << ") - ";

        // Load a batch
        d.LoadBatch(x, y);

        // Preprocessing
        x->div_(255.0);

        // Evaluate batch
        evaluate(net, { x }, { y });
    }

    delete x;
    delete y;

    return EXIT_SUCCESS;
}