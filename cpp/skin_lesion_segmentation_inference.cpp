#include "data_generator/data_generator.h"
#include "metrics/metrics.h"
#include "models/models.h"

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

int main()
{
    // Settings
    int batch_size = 15;
    bool save_images = true;
    path result_dir;

    // Define network
    model net = import_net_from_onnx_file("isic_segmentation_checkpoint_epoch_29.onnx");

    // Build model
    build(net,
          adam(0.0001f),              // Optimizer
          { "cross_entropy" },        // Losses
          { "mean_squared_error" },   // Metrics
          CS_GPU({1}, 1, "low_mem"),  // Computing Service
          false                       // Randomly initialize network weights
    );

    // View model
    summary(net);
    plot(net, "model.pdf");
    setlogfile(net, "skin_lesion_segmentation_inference");

    // Set size from the size of the input layer of the network
    std::vector<int> size{ net->layers[0]->input->shape[2], net->layers[0]->input->shape[3] };

    auto test_augs = make_shared<SequentialAugmentationContainer>(AugResizeDim(size));
    DatasetAugmentations dataset_augmentations{ {nullptr, nullptr, test_augs } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d("D:/dataset/isic_segmentation/isic_segmentation.yml", batch_size, dataset_augmentations);

    if (save_images) {
        result_dir = "../output_images_segmentation_inference";
        create_directory(result_dir);
    }

    d.SetSplit(SplitType::test);
    int num_samples_test = vsize(d.GetSplit());
    int num_batches_test = num_samples_test / batch_size;
    DataGenerator d_generator(&d, batch_size, size, size, 5);

    tensor output;
    View<DataType::float32> img_t;
    View<DataType::float32> gt_t;
    Image orig_img_t, labels, tmp;
    vector<vector<Point2i>> contours;
    Eval evaluator;

    evaluator.ResetEval();
    d_generator.Start();

    // Test for each batch
    cout << "Starting test:" << endl;
    for (int i = 0, n = 0; d_generator.HasNext(); ++i) {
        cout << "Test (batch " << i << "/" << num_batches_test - 1 << ") ";
        cout << "|fifo| " << d_generator.Size() << " ";
        tensor x, y;

        // Load a batch
        if (d_generator.PopBatch(x, y)) {
            // Preprocessing
            x->div_(255.);
            y->div_(255.);

            forward(net, { x });
            output = getOutput(getOut(net)[0]);

            // Compute IoU metric and optionally save the output images
            for (int j = 0; j < batch_size; ++j, ++n) {
                tensor img = output->select({to_string(j)});
                TensorToView(img, img_t);
                img_t.colortype_ = ColorType::GRAY;
                img_t.channels_ = "xyc";

                tensor gt = y->select({to_string(j)});
                TensorToView(gt, gt_t);
                gt_t.colortype_ = ColorType::GRAY;
                gt_t.channels_ = "xyc";

                cout << "- IoU: " << evaluator.BinaryIoU(img_t, gt_t) << " ";

                if (save_images) {
                    tensor orig_img = x->select({to_string(j)});
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

                    ImWrite(result_dir / filename.replace_extension(".png"), tmp);

                    gt->mult_(255.);
                    ImWrite(result_dir / filename_gt, gt_t);

                    delete orig_img;
                }

                delete img;
                delete gt;
            }
            delete x;
            delete y;
            cout << endl;
        }
    }
    d_generator.Stop();

    cout << "----------------------------" << endl;
    cout << "MIoU: " << evaluator.MeanMetric() << endl;
    cout << "----------------------------" << endl;

    ofstream of("output_segmentation_inference.txt");
    of << "MIoU: " << evaluator.MeanMetric() << endl;
    of.close();

    delete output;

    return EXIT_SUCCESS;
}