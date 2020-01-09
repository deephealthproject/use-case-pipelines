#include "ecvl/core.h"
#include "ecvl/eddl.h"
#include "ecvl/dataset_parser.h"
#include "models/models.h"

#include <iostream>

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

    // Define network
    layer in = Input({ 1, size[0], size[1] });
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

    cout << "Reading dataset" << endl;
    Dataset d("mnist/mnist.yml");
    tensor x_train;
    tensor y_train;

    cout << "Creating EDDL tensor" << endl;
    TrainingToTensor(d, size, x_train, y_train, ctype);

    // Preprocessing
    x_train->div_(255.0);
    cout << "Starting training" << endl;
    fit(net, { x_train }, { y_train }, batch_size, epochs);
    save(net, "mnist_checkpoint.bin", "bin");

    // Evaluation
    tensor x_test;
    tensor y_test;
    TestToTensor(d, size, x_test, y_test, ctype);

    // Preprocessing
    x_test->div_(255.0);
    cout << "Evaluate test:" << endl;
    evaluate(net, { x_test }, { y_test });

    return EXIT_SUCCESS;
}