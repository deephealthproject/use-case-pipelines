#include "ecvl/core.h"
#include "ecvl/eddl.h"
#include "ecvl/dataset_parser.h"
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
    std::random_device rd;
    std::mt19937 g(rd());

    // Define network
    layer in = Input({ 1, size[0],  size[1] });
    layer out = LeNet(in, num_classes); // Model LeNet
    model net = Model({ in }, { out });

    // Build model
    build(net,
        sgd(0.001, 0.9), // Optimizer
        { "soft_cross_entropy" }, // Loss
        { "categorical_accuracy" } // Metric
    );

    toGPU(net);

    // View model
    summary(net);
    plot(net, "model.pdf");

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d("mnist/mnist.yml", batch_size, size, ctype);

    // Prepare tensors which store batch
    tensor x = eddlT::create({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y = eddlT::create({ batch_size, static_cast<int>(d.classes_.size()) });

    int num_samples = d.GetSplit().size();
    int num_batches = num_samples / batch_size;

    vector<int> indices(batch_size);
    iota(indices.begin(), indices.end(), 0);

    cout << "Starting training" << endl;
    for (int i = 0; i < epochs; ++i) {
        // Reset errors
        reset_loss(net);

        // Shuffle training list
        shuffle(std::begin(d.GetSplit()), std::end(d.GetSplit()), g);
        d.ResetCurrentBatch();

        // Feed batches to the model
        for (int j = 0; j < num_batches; ++j) {
            cout << "Epoch " << i + 1 << "/" << epochs << " (batch " << j + 1 << "/" << num_batches << ") - ";

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
            cout << endl;
        }
    }

    save(net, "mnist_checkpoint.bin", "bin");

    // Evaluation
    d.SetSplit("test");
    num_samples = d.GetSplit().size();
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