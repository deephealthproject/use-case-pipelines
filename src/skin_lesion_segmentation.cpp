#include "metrics/metrics.h"
#include "models/models.h"
#include "utils/utils.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

using namespace ecvl;
using namespace eddl;
using namespace std;
using namespace std::filesystem;

int main()
{
    // Settings
    int epochs = 20;
    int batch_size = 2;
    int num_classes = 1;
    std::vector<int> size{ 192, 192 }; // Size of images

    std::mt19937 g(std::random_device{}());

    bool save_images = true;
    path output_path;

    if (save_images) {
        output_path = "../output_images";
        create_directory(output_path);
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

    auto training_augs = make_unique<SequentialAugmentationContainer>(
        AugMirror(.5),
        AugFlip(.5),
        AugRotate({ -180, 180 }),
        AugAdditivePoissonNoise({ 0, 10 }),
        AugGammaContrast({ .5, 1.5 }),
        AugGaussianBlur({ .0, .8 }),
        AugCoarseDropout({ 0, 0.3 }, { 0.02, 0.05 }, 0.5),
        AugResizeDim(size));

    auto validation_augs = make_unique<SequentialAugmentationContainer>(AugResizeDim(size));

    DatasetAugmentations dataset_augmentations{ {move(training_augs), move(validation_augs), nullptr } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    //Training split is set by default
    DLDataset d("D:/dataset/isic_segmentation/isic_segmentation.yml", batch_size, move(dataset_augmentations));

    // Prepare tensors which store batch
    tensor x = eddlT::create({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y = eddlT::create({ batch_size, 1, size[0], size[1] });

    // Get number of training samples
    int num_samples = d.GetSplit().size();
    int num_batches = num_samples / batch_size;

    // Get number of validation samples
    d.SetSplit(SplitType::validation);

    int num_samples_validation = d.GetSplit().size();
    int num_batches_validation = num_samples_validation / batch_size;

    vector<int> indices(batch_size);
    iota(indices.begin(), indices.end(), 0);
    cv::TickMeter tm;

    View<DataType::float32> img_t;
    View<DataType::float32> gt_t;
    ofstream of;

    Eval evaluator;
    cout << "Starting training" << endl;
    for (int i = 0; i < epochs; ++i) {
        d.SetSplit(SplitType::training);
        // Reset errors
        reset_loss(net);

        // Shuffle training list
        shuffle(std::begin(d.GetSplit()), std::end(d.GetSplit()), g);
        d.ResetAllBatches();

        // Feed batches to the model
        for (int j = 0; j < num_batches; ++j) {
            cout << "Epoch " << i << "/" << epochs << " (batch " << j << "/" << num_batches << ") - ";
            tm.reset();
            tm.start();
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
            tm.stop();

            cout << "- Elapsed time: " << tm.getTimeSec() << endl;
        }

        cout << "Saving weights..." << endl;
        save(net, "isic_segmentation_checkpoint_epoch_" + to_string(i) + ".bin", "bin");

        cout << "Starting validation:" << endl;
        d.SetSplit(SplitType::validation);

        evaluator.ResetEval();

        // Validation for each batch
        for (int j = 0; j < num_batches_validation; ++j) {
            cout << "Validation - Epoch " << i << "/" << epochs << " (batch " << j << "/" << num_batches_validation << ") ";

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
                img_t.colortype_ = ColorType::GRAY;

                tensor gt = eddlT::select(y, k);
                TensorToView(gt, gt_t);
                gt_t.colortype_ = ColorType::GRAY;

                cout << "- IoU: " << evaluator.BinaryIoU(img_t, gt_t) << " ";

                if (save_images) {
                    img->mult_(255.);
                    ImWrite(output_path / path("batch_" + to_string(j) + "_output_" + to_string(k) + ".png"), img_t);

                    if (i == 0) {
                        gt->mult_(255.);
                        ImWrite(output_path / path("batch_" + to_string(j) + "_gt_" + to_string(k) + ".png"), gt_t);
                    }
                }
            }
            cout << endl;
        }
        cout << "----------------------------" << endl;
        cout << "MIoU: " << evaluator.MIoU() << endl;
        cout << "----------------------------" << endl;

        of.open("output_evaluate_isic_segmentation.txt", ios::out | ios::app);
        of << "Epoch " << i << " - MIoU: " << evaluator.MIoU() << endl;
        of.close();
    }

    delete x;
    delete y;

    return EXIT_SUCCESS;
}