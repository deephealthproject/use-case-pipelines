#include "metrics/metrics.h"
#include "models/models.h"
#include "utils/utils.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <random>

using namespace ecvl;
using namespace eddl;
using namespace std;

int main()
{
    // Settings
    int epochs = 20;
    int batch_size = 2;
    int num_classes = 1;
    std::vector<int> size{ 192, 192 }; // Size of images

    std::random_device rd;
    std::mt19937 g(rd());

    bool save_images = true;

    if (save_images) {
        filesystem::create_directory("output_images");
    }

    // Define network
    layer in = Input({ 3, size[0], size[1] });
    //layer out = UNetWithPadding(in, num_classes);
    layer out = SegNet(in, num_classes);
    layer out_sigm = Sigmoid(out);
    model net = Model({ in }, { out_sigm });

    // Build model
    build(net,
        adam(0.0001), //Optimizer
        { "cross_entropy" }, // Losses
        { "mean_squared_error" } // Metrics
    );

    toGPU(net);

    // View model
    summary(net);
    plot(net, "model.pdf");

    // Read the dataset
    cout << "Reading dataset" << endl;

    //Training split is set by default
    DLDataset d("D:/dataset/isic_segmentation/isic_segmentation.yml", batch_size, size);

    // Prepare tensors which store batch
    tensor x = eddlT::create({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y = eddlT::create({ batch_size, 1, size[0], size[1] });

    // Get number of training samples
    int num_samples = d.GetSplit().size();
    int num_batches = num_samples / batch_size;

    // Get number of validation samples
    d.SetSplit("validation");
    int num_samples_validation = d.GetSplit().size();
    int num_batches_validation = num_samples_validation / batch_size;

    vector<int> indices(batch_size);
    iota(indices.begin(), indices.end(), 0);
    View<DataType::float32> img_t;
    View<DataType::float32> gt_t;

    Eval evaluator;
    cout << "Starting training" << endl;
    for (int i = 0; i < epochs; ++i) {
        d.SetSplit("training");
        // Reset errors
        reset_loss(net);

        // Shuffle training list
        shuffle(std::begin(d.GetSplit()), std::end(d.GetSplit()), g);
        d.ResetAllBatches();

        // Feed batches to the model
        for (int j = 0; j < num_batches; ++j) {
            cout << "Epoch " << i + 1 << "/" << epochs << " (batch " << j + 1 << "/" << num_batches << ") - ";

            // Load a batch
            d.LoadBatch(x, y);

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

        cout << "Starting validation:" << endl;
        d.SetSplit("validation");

        evaluator.ResetEval();

        // Validation for each batch
        for (int j = 0; j < num_batches_validation; ++j) {
            cout << "Validation - Epoch " << i + 1 << "/" << epochs << " (batch " << j + 1 << "/" << num_batches_validation << ") ";

            // Load a batch
            d.LoadBatch(x, y);

            // Preprocessing
            x->div_(255.);
            y->div_(255.);

            forward(net, { x });
            tensor output = getTensor(out_sigm);

            // Compute IoU metric and optionally save the output images
            for (int k = 0; k < batch_size; ++k) {
                tensor img = eddlT::select(output, k);
                TensorToView(img, img_t);

                tensor gt = eddlT::select(y, k);
                TensorToView(gt, gt_t);

                cout << "- IoU: " << evaluator.BinaryIoU(img_t, gt_t) << " ";

                if (save_images) {
                    ImageSqueeze(img_t);
                    img->mult_(255.);
                    ImWrite("output_images/batch_" + to_string(j) + "_output.png", img_t);

                    if (i == 0) {
                        ImageSqueeze(gt_t);
                        gt->mult_(255.);
                        ImWrite("output_images/batch_" + to_string(j) + "_gt.png", gt_t);
                    }
                }
            }
            cout << endl;
        }
        cout << "----------------------------" << endl;
        cout << "MIoU: " << evaluator.MIoU() << endl;
        cout << "----------------------------" << endl;
    }

    save(net, "isic_segmentation_checkpoint.bin", "bin");
    delete x;
    delete y;

    return EXIT_SUCCESS;
}