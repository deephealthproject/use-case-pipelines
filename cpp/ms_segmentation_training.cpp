#include "metrics/metrics.h"
#include "models/models.h"
#include "utils/utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>

#include "ecvl/core/filesystem.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace ecvl;
using namespace ecvl::filesystem;
using namespace eddl;
using namespace std;

// MSVolume manages loading of volumes and their slices
class MSVolume
{
public:
    DLDataset& d_;
    const int n_channels_ = 1; // Number of slices to provide as input together
    const int stride_; // Stride represents the jump to make to reach the next volume slice. Default value in constructor
    Image volume_, gt_;
    std::mt19937 g_;

    // Volume related variables
    int slices_; // Number of slices in the current volume
    int current_slice_ = 0; // Index of the current slice
    int current_volume_ = -1; // Index of the current volume of the dataset
    vector<int> indices_; // vector with random indices in [0,slices_ / stride_]
    vector<string> names_; // names of current volume and its ground truth

    // Default: stride = n_channels (means no overlapping)
    MSVolume(DLDataset& d, int n_channels) : d_{ d }, n_channels_{ n_channels }, stride_{ n_channels }, g_(std::random_device{}()) {}

    MSVolume(DLDataset& d, int n_channels, int stride) : d_{ d }, n_channels_{ n_channels }, stride_{ stride }, g_(std::random_device{}()) {}

    // Open a new volume, its ground truth and reset slice variables
    void Init()
    {
        current_volume_++;
        // Get next volume from DLDataset
        const int index = d_.GetSplit()[current_volume_];
        Sample& elem = d_.samples_[index];

        // Load a volume and its gt in memory
        volume_ = elem.LoadImage(d_.ctype_, false);
        Image tmp = elem.LoadImage(d_.ctype_gt_, true);
        CopyImage(tmp, gt_, DataType::float32);

        current_slice_ = 0;
        slices_ = volume_.Channels();

        // indices created as random between 0 and slices_ / stride_
        indices_ = vector<int>(slices_ / stride_);
        iota(indices_.begin(), indices_.end(), 0);
        shuffle(indices_.begin(), indices_.end(), g_);

        // Save names of current volume and gt --> for save_images
        names_.clear();
        string sample_name = elem.location_[0].parent_path().stem().string() + "_" + elem.location_[0].stem().string() + "_";
        names_.emplace_back(sample_name);
        sample_name = elem.label_path_.value().parent_path().stem().string() + "_" + elem.label_path_.value().stem().string() + "_";
        names_.emplace_back(sample_name);
    }

    // Reset variables at each training or validation
    void Reset()
    {
        current_volume_ = -1;
        current_slice_ = 0;
        indices_.clear();
        d_.ResetAllBatches();
    }

    // "Override" the LoadBatch of DLDataset
    // return false if every volume has been processed, true otherwise
    bool LoadBatch(tensor& images, tensor& labels)
    {
        int& bs = d_.batch_size_;

        int offset = 0, start = 0;
        start = d_.current_batch_[+d_.current_split_] * bs;

        // Load a new volume if we already loaded all slices of the current one
        if (current_slice_ >= vsize(indices_) || vsize(indices_) < start + bs) {
            // Stop training/validation if there are no more volume to read
            if (current_volume_ >= vsize(d_.GetSplit()) - 1) {
                return false;
            }
            // Load a new volume
            Init();
            d_.ResetCurrentBatch();
        }

        ++d_.current_batch_[+d_.current_split_];

        // Fill tensors with data
        for (int i = start; i < start + bs; ++i) {
            // Read slices and their ground truth
            View<DataType::float32> v_volume_(volume_, { 0, 0, indices_[current_slice_] * stride_ }, { volume_.Width(), volume_.Height(), n_channels_ });
            View<DataType::float32> v_gt_(gt_, { 0, 0, indices_[current_slice_] * stride_ }, { gt_.Width(), gt_.Height(), n_channels_ });

            // Apply chain of augmentations to sample image and corresponding ground truth
            d_.augs_.Apply(d_.current_split_, v_volume_, v_gt_);

            // Copy label into tensor (labels)
            ImageToTensor(v_gt_, labels, offset);
            // Copy image into tensor (images)
            ImageToTensor(v_volume_, images, offset);

            ++offset;
            ++current_slice_;
        }

        return true;
    }
};

int main(int argc, char* argv[])
{
    // Settings parsed from command-line arguments
    Settings s;
    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
    }

    // Build model
    build(s.net,
        adam(s.lr),         // Optimizer
        { s.loss },         // Loss
        { "dice" },         // Metric
        s.cs,               // Computing Service
        s.random_weights    // Randomly initialize network weights
    );

    // View model
    summary(s.net);
    plot(s.net, "model.pdf");
    setlogfile(s.net, "ms_segmentation");

    // Set augmentations for training and validation
    auto training_augs = make_shared<SequentialAugmentationContainer>(AugResizeDim(s.size),
                                                                      //AugMirror(.5),
                                                                      //AugFlip(.5),
                                                                      AugRotate({ -180, 180 }));
                                                                      //AugAdditivePoissonNoise({ 0, 10 }),
                                                                      //AugGaussianBlur({ .0, .8 }),
                                                                      //AugCoarseDropout({ 0, 0.3 }, { 0.02, 0.05 }, 0.5));

    auto validation_augs = make_shared<SequentialAugmentationContainer>(AugResizeDim(s.size));
    auto testing_augs = make_shared<SequentialAugmentationContainer>(AugResizeDim(s.size));

    DatasetAugmentations dataset_augmentations{ {training_augs, validation_augs, testing_augs } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    //Training split is set by default
    DLDataset d(s.dataset_path, s.batch_size, dataset_augmentations, ColorType::none, ColorType::none);
    MSVolume v(d, s.n_channels); // MSVolume takes a reference to DLDataset

    // Prepare tensors which store batch
    tensor x = new Tensor({ d.batch_size_, s.n_channels, s.size[0], s.size[1] });
    tensor y = new Tensor({ d.batch_size_, s.n_channels, s.size[0], s.size[1] });

    // Get number of training samples.
    int num_samples = vsize(d.GetSplit()) * s.n_channels;
    int num_batches = num_samples / s.batch_size;

    d.SetSplit(SplitType::validation);

    // Get number of validation samples.
    int num_samples_validation = vsize(d.GetSplit()) * s.n_channels;
    int num_batches_validation = num_samples_validation / s.batch_size;

    vector<int> indices(s.batch_size);
    iota(indices.begin(), indices.end(), 0);

    std::mt19937 g(std::random_device{}());
    //    View<DataType::float32> pred_ecvl;
    //    View<DataType::float32> gt_ecvl;
    Image input_image_ecvl;
    Image pred_ecvl;
    Image gt_ecvl;
    Image orig_img, orig_gt;
    float best_metric = 0;
    ofstream of;
    Eval evaluator;
    cv::TickMeter tm;

    if (s.do_training) {
        cout << "Starting training" << endl;
        for (int i = 0; i < s.epochs; ++i) {
            v.Reset();

            auto current_path{ s.result_dir / path("Epoch_" + to_string(i)) };
            if (s.save_images) {
                create_directory(current_path);
            }

            d.SetSplit(SplitType::training);

            // Reset errors
            reset_loss(s.net);

            // Shuffle training list
            shuffle(d.GetSplit().begin(), d.GetSplit().end(), g);

            // Feed batches to the model
            int j = 0, old_volume = 0;
            while (true) {
                tm.reset();
                tm.start();
                if (!v.LoadBatch(x, y)) {
                    break; // All volumes have been processed
                }

                if (old_volume != v.current_volume_) {
                    j = 0; // Current volume ended
                    old_volume = v.current_volume_;
                }

                cout << "Epoch " << i << "/" << s.epochs - 1 << \
                    " - volume " << v.current_volume_ << "/" << vsize(d.GetSplit()) - 1 << \
                    " - batch " << j << "/" << v.slices_ / (v.n_channels_ * d.batch_size_) - 1;

                tm.stop();
                cout << " - Load time: " << tm.getTimeSec() << " - ";
                tm.reset();
                tm.start();

                // Train batch
                train_batch(s.net, { x }, { y }, indices);

                print_loss(s.net, j);
                tm.stop();
                cout << "Train time: " << tm.getTimeSec() << endl;
                ++j;
            }

            cout << "Starting validation:" << endl;
            d.SetSplit(SplitType::validation);

            v.Reset();
            evaluator.ResetEval();

            // Validation for each batch
            j = 0, old_volume = 0;
            while (true) {
                tm.reset();
                tm.start();
                if (!v.LoadBatch(x, y)) {
                    break; // All volumes have been processed
                }

                if (old_volume != v.current_volume_) {
                    j = 0; // Current volume ended
                    old_volume = v.current_volume_;
                }

                cout << "Validation - Epoch " << i << "/" << s.epochs - 1 << \
                    " - volume " << v.current_volume_ << "/" << vsize(d.GetSplit()) - 1 << \
                    " - batch " << j << "/" << v.slices_ / (v.n_channels_ * d.batch_size_) - 1;

                tm.stop();
                cout << " - Load time: " << tm.getTimeSec() << " - ";
                tm.reset();
                tm.start();

                forward(s.net, { x });
                unique_ptr<Tensor> output(getOutput(getOut(s.net)[0]));

                // Compute Dice metric and optionally save the output images
                for (int k = 0; k < s.batch_size; ++k) {
                    tensor pred = output->select({ to_string(k) });
                    TensorToImage(pred, pred_ecvl);
                    pred_ecvl.colortype_ = ColorType::GRAY;
                    pred_ecvl.channels_ = "xyc";

                    tensor gt = y->select({ to_string(k) });
                    TensorToImage(gt, gt_ecvl);
                    gt_ecvl.colortype_ = ColorType::GRAY;
                    gt_ecvl.channels_ = "xyc";

                    for (int m = 0; m < s.n_channels; ++m) {
                        View<DataType::float32> view(pred_ecvl, { 0, 0, m }, { pred_ecvl.Width(), pred_ecvl.Height(), 1 });
                        View<DataType::float32> view_gt(gt_ecvl, { 0, 0, m }, { gt_ecvl.Width(), gt_ecvl.Height(), 1 });

                        // NOTE: dice computed on downsampled images
                        cout << "- Dice: " << evaluator.DiceCoefficient(view, view_gt) << " ";

                        if (s.save_images) {
                            Mul(gt_ecvl, 255, gt_ecvl);
                            Mul(pred_ecvl, 255, pred_ecvl);
                            ImWrite(current_path / path(v.names_[0] + to_string(v.indices_[v.current_slice_ - s.batch_size + k] * v.stride_ + m) + ".png"), view);
                            ImWrite(current_path / path(v.names_[1] + to_string(v.indices_[v.current_slice_ - s.batch_size + k] * v.stride_ + m) + ".png"), view_gt);
                        }
                    }
                    delete pred;
                    delete gt;
                }
                tm.stop();
                cout << " - Validation time: " << tm.getTimeSec() << endl;

                ++j;
            }

            float mean_metric = evaluator.MeanMetric();
            cout << "----------------------------" << endl;
            cout << "Mean Dice Coefficient: " << mean_metric << endl;
            cout << "----------------------------" << endl;

            if (mean_metric > best_metric) {
                cout << "Saving weights..." << endl;
                save_net_to_onnx_file(s.net, (s.checkpoint_dir / path("ms_segmentation_checkpoint_epoch_" + to_string(i) + ".onnx")).string());
                best_metric = mean_metric;
            }

            // Save metric values on file
            of.open("output_evaluate_ms_segmentation.txt", ios::out | ios::app);
            of << "Epoch " << i << " - Mean Dice Coefficient: " << evaluator.MeanMetric() << endl;
            of.close();
        }
    }
    if (s.do_test) {
        cout << "Starting test:" << endl;
        d.SetSplit(SplitType::test);

        auto current_path{ s.result_dir / path("testing_") };
        if (s.save_images) {
            create_directory(current_path);
        }

        v.Reset();
        evaluator.ResetEval();

        // Testing for each batch
        int j = 0, old_volume = 0;
        while (true) {
            tm.reset();
            tm.start();
            if (!v.LoadBatch(x, y)) {
                break; // All volumes have been processed
            }

            if (old_volume != v.current_volume_) {
                j = 0; // Current volume ended
                old_volume = v.current_volume_;
            }

            cout << "Testing "
                << " - volume " << v.current_volume_ << "/" << vsize(d.GetSplit()) - 1
                << " - batch " << j << "/" << v.slices_ / (v.n_channels_ * d.batch_size_) - 1;

            tm.stop();
            cout << " - Load time: " << tm.getTimeSec() << " - ";
            tm.reset();
            tm.start();

            forward(s.net, { x });
            unique_ptr<Tensor> output(getOutput(getOut(s.net)[0]));

            // Compute Dice metric and optionally save the output images
            for (int k = 0; k < s.batch_size; ++k) {
                tensor input_image = x->select({ to_string(k) });
                //input_image->print();
                TensorToImage(input_image, input_image_ecvl);
                input_image_ecvl.colortype_ = ColorType::GRAY;
                input_image_ecvl.channels_ = "xyc";

                tensor pred = output->select({ to_string(k) });
                TensorToImage(pred, pred_ecvl);
                pred_ecvl.colortype_ = ColorType::GRAY;
                pred_ecvl.channels_ = "xyc";

                tensor gt = y->select({ to_string(k) });
                TensorToImage(gt, gt_ecvl);
                gt_ecvl.colortype_ = ColorType::GRAY;
                gt_ecvl.channels_ = "xyc";

                for (int m = 0; m < s.n_channels; ++m) {
                    View<DataType::float32> view_input_image(input_image_ecvl, { 0, 0, m }, { input_image_ecvl.Width(), input_image_ecvl.Height(), 1 });
                    View<DataType::float32> view(pred_ecvl, { 0, 0, m }, { pred_ecvl.Width(), pred_ecvl.Height(), 1 });
                    View<DataType::float32> view_gt(gt_ecvl, { 0, 0, m }, { gt_ecvl.Width(), gt_ecvl.Height(), 1 });

                    // NOTE: dice computed on downsampled images
                    cout << "- Dice: " << evaluator.DiceCoefficient(view, view_gt) << " ";

                    if (s.save_images) {
                        Mul(gt_ecvl, 255, gt_ecvl);
                        Mul(pred_ecvl, 255, pred_ecvl);
                        ImWrite(current_path / path(v.names_[0] + to_string(v.indices_[v.current_slice_ - s.batch_size + k] * v.stride_ + m) + "-img.png"), view_input_image);
                        ImWrite(current_path / path(v.names_[0] + to_string(v.indices_[v.current_slice_ - s.batch_size + k] * v.stride_ + m) + ".png"), view);
                        ImWrite(current_path / path(v.names_[1] + to_string(v.indices_[v.current_slice_ - s.batch_size + k] * v.stride_ + m) + ".png"), view_gt);
                    }
                }
                delete input_image;
                delete pred;
                delete gt;
            }
            tm.stop();
            cout << " - Testing time: " << tm.getTimeSec() << endl;

            ++j;
        }

        float mean_metric = evaluator.MeanMetric();
        cout << "----------------------------" << endl;
        cout << "Mean Dice Coefficient in test: " << mean_metric << endl;
        cout << "----------------------------" << endl;
    }

    delete x;
    delete y;

    return EXIT_SUCCESS;
}
