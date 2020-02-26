#include "metrics/metrics.h"
#include "models/models.h"

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
    int batch_size = 12;
    int num_classes = 1;
    std::vector<int> size{ 192, 192 }; // Size of images

    std::mt19937 g(std::random_device{}());

    bool save_images = true;
    path output_path;

    if (save_images) {
        output_path = "../output_images_segmentation_inference";
        create_directory(output_path);
    }

    // Define network
    layer in = Input({ 3, size[0], size[1] });
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
    setlogfile(net, "skin_lesion_segmentation_inference");

    auto training_augs = make_unique<SequentialAugmentationContainer>(AugResizeDim(size));
    auto test_augs = make_unique<SequentialAugmentationContainer>(AugResizeDim(size));
    DatasetAugmentations dataset_augmentations{ {move(training_augs), nullptr, move(test_augs) } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d("D:/dataset/isic_segmentation/isic_segmentation.yml", batch_size, move(dataset_augmentations));

    // Prepare tensors which store batch
    tensor x = eddlT::create({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y = eddlT::create({ batch_size, 1, size[0], size[1] });
    tensor output;

    // Get number of test samples
    d.SetSplit(SplitType::test);
    int num_samples_test = d.GetSplit().size();
    int num_batches_test = num_samples_test / batch_size;

    View<DataType::float32> img_t;
    View<DataType::float32> gt_t;
    Image orig_img_t, labels, tmp;
    vector<vector<Point2i>> contours;

    Eval evaluator;
    load(net, "isic_segmentation_checkpoint_epoch_24.bin", "bin");

    cout << "Starting test:" << endl;
    evaluator.ResetEval();

    // Test for each batch
    for (int i = 0, n = 0; i < num_batches_test; ++i) {
        cout << "Test - (batch " << i << "/" << num_batches_test << ") ";

        // Load a batch
        d.LoadBatch(x, y);

        // Preprocessing
        x->div_(255.);
        y->div_(255.);

        forward(net, { x });
        output = getTensor(out_sigm);

        // Compute IoU metric and optionally save the output images
        for (int j = 0; j < batch_size; ++j, ++n) {
            tensor img = eddlT::select(output, j);
            TensorToView(img, img_t);
            img_t.colortype_ = ColorType::GRAY;
            img_t.channels_ = "xyc";

            tensor gt = eddlT::select(y, j);
            TensorToView(gt, gt_t);
            gt_t.colortype_ = ColorType::GRAY;
            gt_t.channels_ = "xyc";

            cout << "- IoU: " << evaluator.BinaryIoU(img_t, gt_t) << " ";

            if (save_images) {
                tensor orig_img = eddlT::select(x, j);
                orig_img->mult_(255.);
                TensorToImage(orig_img, orig_img_t);
                orig_img_t.colortype_ = ColorType::BGR;
                orig_img_t.channels_ = "xyc";

                img->mult_(255.);
                CopyImage(img_t, tmp, DataType::uint8);
                ConnectedComponentsLabeling(tmp, labels);
                CopyImage(labels, tmp, DataType::uint8);
                FindContours(tmp, contours);
                CopyImage(orig_img_t, tmp, DataType::uint8);

                for (auto c : contours[0]) {
                    *tmp.Ptr({ c[0], c[1], 0 }) = 0;
                    *tmp.Ptr({ c[0], c[1], 1 }) = 0;
                    *tmp.Ptr({ c[0], c[1], 2 }) = 255;
                }

                path filename = d.samples_[d.GetSplit()[n]].location_[0].filename();
                path filename_gt = d.samples_[d.GetSplit()[n]].label_path_.value().filename();

                ImWrite(output_path / filename.replace_extension(".png"), tmp);

                gt->mult_(255.);
                ImWrite(output_path / filename_gt, gt_t);
            }
        }
        cout << endl;
    }
    cout << "----------------------------" << endl;
    cout << "MIoU: " << evaluator.MIoU() << endl;
    cout << "----------------------------" << endl;

    ofstream of("output_segmentation_inference.txt");
    of << "MIoU: " << evaluator.MIoU() << endl;
    of.close();

    delete x;
    delete y;

    return EXIT_SUCCESS;
}