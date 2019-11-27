#include "models/models.h"
#include "metrics/metrics.h"
#include "utils/utils.h"

#include <algorithm>
#include <iostream>
#include <random>

using namespace ecvl;
using namespace eddl;
using namespace std;

int main()
{
    // Settings
    int epochs = 10;
    int batch_size = 2;
    int num_classes = 1;
    std::vector<int> size{ 192, 192 }; // Size of images

    std::random_device rd;
    std::mt19937 g(rd());

    // Define network
    layer in = Input({ 3, size[0], size[1] });
    //layer out = UNetWithPadding(in, num_classes);
    layer out = SegNet(in, num_classes);
    layer out_sigm = Sigmoid(out);
    model net = Model({ in }, { out_sigm });

    // Build model
    build(net,
        //sgd(0.001, 0.9), // Optimizer
        adam(0.0001), //Optimizer
        //{ "cross_entropy" }, // Losses
        { "cross_entropy" }, // Losses
        //{ "categorical_accuracy" }, // Metrics
        { "mean_squared_error" } // Metrics
    );

    toGPU(net);

    // View model
    summary(net);
    plot(net, "model.pdf");

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d("D:/dataset/isic_2017/isic_segmentation.yml", batch_size, "training");

    // Prepare tensors which store batch
    tensor x = eddlT::create({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y = eddlT::create({ batch_size, 1, size[0], size[1] });

    // Set batch size
    resize_model(net, batch_size);
    // Set training mode
    set_mode(net, TRMODE);

    d.SetSplit("validation");
    int num_samples_validation = d.GetSplit().size();
    int num_batches_validation = num_samples_validation / batch_size;

    d.SetSplit("training");
    int num_samples = d.GetSplit().size();
    int num_batches = num_samples / batch_size;

    vector<int> indices(batch_size);
    iota(indices.begin(), indices.end(), 0);

    Eval evaluator;
    cout << "Starting training" << endl;
    for (int i = 0; i < epochs; ++i) {
        d.SetSplit("training");
        // Reset errors
        reset_loss(net);

        // Shuffle training list
        shuffle(std::begin(d.GetSplit()), std::end(d.GetSplit()), g);
        d.current_batch_ = 0;

        // Feed batches to the model
        for (int j = 0; j < num_batches; ++j, ++d.current_batch_) {
            cout << "Epoch " << i + 1 << "/" << epochs << " (batch " << j + 1 << "/" << num_batches << ") - ";

            // Load a batch
            LoadBatch(d, size, x, y);

            // Preprocessing
            x->div_(255.);
            y->div_(255.);

            // Prepare data
            vtensor tx{ x };
            vtensor ty{ y };

            // Train batch
            train_batch(net, tx, ty, indices);
            print_loss(net, j);
            cout << endl;
        }

        cout << "Validation: " << endl;
        d.SetSplit("validation");
        d.current_batch_ = 0;

        evaluator.ResetEval();

        // Validation for each batch
        for (int j = 0; j < num_batches_validation; ++j, ++d.current_batch_) {
            cout << "Validation - Epoch " << i + 1 << "/" << epochs << " (batch " << j + 1 << "/" << num_batches_validation << ") - ";

            // Load a batch
            LoadBatch(d, size, x, y);

            // Preprocessing
            x->div_(255.);
            y->div_(255.);

            // Prepare data
            vtensor tsx{ x };
            vtensor tsy{ y };

            forward(net, tsx);
            tensor output = getTensor(out_sigm);

            for (int k = 0; k < batch_size; ++k) {

                tensor img = eddlT::select(output, k);
                Image img_t = TensorToView(img);

                tensor gt = eddlT::select(y, k);
                Image gt_t = TensorToView(gt);

                cout << "IoU: " << evaluator.BinaryIoU(img_t, gt_t) << endl;

                ImageSqueeze(img_t);
                ImWrite("batch_" + to_string(j) + "_output.png", img_t);

                if (i == 0) {
                    ImageSqueeze(gt_t);
                    ImWrite("batch_" + to_string(j) + "_gt.png", gt_t);
                }

            }
        }
        cout << "----------------------------" << endl;
        cout << "MIoU: " << evaluator.MIoU() << endl;
        cout << "----------------------------" << endl;
    }

    save(net, "isic_segmentation_checkpoint.bin");
    delete x;
    delete y;

    return EXIT_SUCCESS;
}