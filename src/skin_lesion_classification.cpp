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
    int epochs = 50;
    int batch_size = 12;
    int num_classes = 8;
    std::vector<int> size{ 224,224 }; // Size of images

    std::random_device rd;
    std::mt19937 g(rd());

    // Define network
    layer in = Input({ 3, size[0],  size[1] });
    layer out = VGG16(in, num_classes);
    model net = Model({ in }, { out });

    // View model
    cout << summary(net) << endl;
    plot(net, "model.pdf");

    // Build model
    build(net,
        sgd(0.001, 0.9), // Optimizer
        { "soft_cross_entropy" }, // Losses
        { "categorical_accuracy" }, // Metrics
        CS_GPU({ 1 })
        //CS_CPU(4) // CPU with 4 threads
    );

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d("D:/datasets/isic_2019/isic_skin_lesion/isic.yml", batch_size, "training");

    // Prepare tensors which store batch
    tensor x_train = eddlT::create({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y_train = eddlT::create({ batch_size, static_cast<int>(d.classes_.size()) });

    // Set batch size
    resize_model(net, batch_size);
    // Set training mode
    set_mode(net, TRMODE);

    // Store errors of each output layer
    verr total_loss = { 0 };
    verr total_metric = { 0 };

    int num_samples = d.GetSplit().size();
    int num_batches = num_samples / batch_size;

    vector<int> indices(batch_size);
    iota(indices.begin(), indices.end(), 0);

    cout << "Starting training" << endl;
    for (int i = 0; i < epochs; ++i) {
        // Reset errors
        total_loss[0] = 0.0;
        total_metric[0] = 0.0;

        // Shuffle training list
        shuffle(std::begin(d.GetSplit()), std::end(d.GetSplit()), g);
        d.current_batch_ = 0;

        // Feed batches to the model
        for (int j = 0; j < num_batches; ++j, ++d.current_batch_) {
            cout << "Epoch " << i + 1 << "/" << epochs << " (batch " << j + 1 << "/" << num_batches << ") - ";

            // Load a batch
            LoadBatch(d, size, x_train, y_train);

            // Preprocessing
            x_train->div_(255.0);

            // Prepare data
            vtensor tx{ x_train };
            vtensor ty{ y_train };

            // Train batch
            train_batch(net, tx, ty, indices);

            // Print errors
            int p = 0;
            for (int k = 0; k < ty.size(); k++, p += 2) {
                total_loss[k] += net->fiterr[p];  // loss
                total_metric[k] += net->fiterr[p + 1];  // metric

                cout << net->lout[k]->name.c_str() << "(" << net->losses[k]->name.c_str() << "=" << total_loss[k] / (batch_size * (j + 1)) << "," <<
                    net->metrics[k]->name.c_str() << "=" << total_metric[k] / (batch_size * (j + 1)) << ")" << endl;

                net->fiterr[p] = net->fiterr[p + 1] = 0.0;
            }
        }
    }

    save(net, "isic_classification_checkpoint.bin");
    delete x_train;
    delete y_train;

    // Evaluation
    // TODO evaluation does not yet allow batched execution
    d.SetSplit("test");
    tensor x_test;
    tensor y_test;
    TestToTensor(d, size, x_test, y_test);

    // Preprocessing
    x_test->div_(255.0);
    cout << "Evaluate test:" << endl;
    evaluate(net, { x_test }, { y_test });

    return EXIT_SUCCESS;
}