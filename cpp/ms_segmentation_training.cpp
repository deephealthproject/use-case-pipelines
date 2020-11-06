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

class MSVolume
{
public:
    DLDataset& d_;
    const int n_channels_ = 1;
    const int stride_;
    Image volume_, gt_;

    // Volume related variables
    int slices_;
    int current_slice_ = 0;
    int current_volume_ = 0;
    vector<int> indices_;
    vector<string> names_;
    std::mt19937 g_;

    // Default: stride = n_channels means no overlapping
    MSVolume(DLDataset& d, int n_channels) : d_{ d }, n_channels_{ n_channels }, stride_{ n_channels }, g_(std::random_device{}()) {}
    MSVolume(DLDataset& d, int n_channels, int stride) : d_{ d }, n_channels_{ n_channels }, stride_{ stride }, g_(std::random_device{}()) {}

    // Open a new volume, its ground truth and reset slice variables
    void Init()
    {
        const int index = d_.GetSplit()[current_volume_];
        current_volume_++;
        Sample& elem = d_.samples_[index];

        volume_ = elem.LoadImage(d_.ctype_, false);
        Image tmp = elem.LoadImage(d_.ctype_gt_, true);
        volume_.channels_ = "zxy";
        tmp.channels_ = "zxy";
        CopyImage(tmp, gt_, DataType::float32);
        
        current_slice_ = 0;
        slices_ = volume_.Channels();

        // [1 2] [3 4] [5 6]
        // [1 2] [2 3] [3 4] [4 5] [5 6]

        // 0 1 2 3 4 5 -> channel 2, stride 2, slices 12
        // 0 1 2 3 -> channel 3, stride 3, slices 12
        // 
        // 0 1 2 3 4 5 6 7 8 9 10 11 -> channel 1, stride 1, slices 12
        // 0 1 2 3 4 5 6 7 8 9 10 11 -> channel 2, stride 1, slices 12

        // 5    1 4 ...
        indices_ = vector<int>(slices_ / stride_);
        iota(indices_.begin(), indices_.end(), 0);
        shuffle(indices_.begin(), indices_.end(), g_);

        // Save names of current volume (volume and gt)
        names_.clear();
        string sample_name = elem.location_[0].parent_path().stem().string() + "_" + elem.location_[0].stem().string() + "_";
        names_.emplace_back(sample_name);
        sample_name = elem.label_path_.value().parent_path().stem().string() + "_" + elem.label_path_.value().stem().string() + "_";
        names_.emplace_back(sample_name);

        // TODO initialize indices for random slices sampling
    }

    void Reset()
    {
        current_volume_ = 0;
        current_slice_ = 0;
        d_.ResetAllBatches();
    }

    bool LoadBatch(tensor& images, tensor& labels)
    {
        if (current_slice_ >= vsize(indices_)) {
            if (current_volume_ >= vsize(d_.GetSplit())) {
                return false;
            }
            // Load a new volume
            Init();
        }

        int& bs = d_.batch_size_;

        //Image img, gt;
        int offset = 0, start = 0;
        //vector<path> names;

        // Check if tensors size matches with batch dimensions
        // size of images tensor must be equal to batch_size * number_of_image_channels * image_width * image_height
        //if (images->size != bs * d.n_channels_ * d.resize_dims_[0] * d.resize_dims_[1]) {
        //    cerr << ECVL_ERROR_MSG "images tensor must have N = batch_size, C = number_of_image_channels, H = image_height, W = image_width" << endl;
        //    ECVL_ERROR_INCOMPATIBLE_DIMENSIONS
        //}
        //
        // segmentation problem so size of labels tensor must be equal to batch_size * number_of_label_channels * image_width * image_height
        //if (labels->size != bs * d.n_channels_gt_ * d.resize_dims_[0] * d.resize_dims_[1]) {
        //    cerr << ECVL_ERROR_MSG "labels tensor must have N = batch_size, C = number_of_label_channels, H = image_height, W = image_width" << endl;
        //    ECVL_ERROR_INCOMPATIBLE_DIMENSIONS
        //}

        // Move to next samples
        //{ // CRITICAL REGION STARTS
            //std::unique_lock<std::mutex> lck(d_.mutex_current_batch_);

        start = d_.current_batch_[+d_.current_split_] * bs;
        ++d_.current_batch_[+d_.current_split_];
        //} // CRITICAL REGION ENDS

        //if (vsize(d.GetSplit()) < start + bs) {
        //    cerr << ECVL_ERROR_MSG "Batch size is not even with the number of samples. Hint: loop through `num_batches = num_samples / batch_size;`" << endl;
        //    ECVL_ERROR_CANNOT_LOAD_IMAGE
        //}

        // Fill tensors with data
        for (int i = start; i < start + bs; ++i) {
            // Read slices and their ground truth
            View<DataType::float32> v_volume_(volume_, { indices_[current_slice_] * stride_,0,0 }, { n_channels_, volume_.Width(), volume_.Height() });
            View<DataType::float32> v_gt_(gt_, { indices_[current_slice_] * stride_,0,0 }, { n_channels_, gt_.Width(), gt_.Height() });

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
    // Settings
    Settings s;
    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
    }

    // Build model
    build(s.net,
        adam(s.lr),      // Optimizer
        { s.loss },                 // Loss
        { "dice" }, // Metric
        s.cs,                       // Computing Service
        s.random_weights            // Randomly initialize network weights
    );

    // View model
    summary(s.net);
    plot(s.net, "model.pdf");
    setlogfile(s.net, "ms_segmentation");

    // Set augmentations for training and validation
    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size)//,
        //AugMirror(.5),
        //AugRotate({ -10, 10 }),
        //AugBrightness({ 0, 30 }),
        //AugGammaContrast({ 0,3 })
        );

    auto validation_augs = make_shared<SequentialAugmentationContainer>(AugResizeDim(s.size));

    DatasetAugmentations dataset_augmentations{ {training_augs, validation_augs, nullptr } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    //Training split is set by default
    DLDataset d(s.dataset_path, s.batch_size, dataset_augmentations, ColorType::none, ColorType::none);
    MSVolume v(d, s.n_channels);

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
    View<DataType::float32> pred_ecvl;
    View<DataType::float32> gt_ecvl;
    Image orig_img, orig_gt;
    float best_metric = 0;
    ofstream of;
    Eval evaluator;
    cv::TickMeter tm;

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
        int j = 0;
        while (true) {
            break;
            if (v.current_slice_ >= vsize(v.indices_)) {
                j = 0; // Current volume ended
            }

            tm.reset();
            tm.start();
            if (!v.LoadBatch(x, y)) {
                 break; // All volumes have been processed
            }
            cout << "Epoch " << i << "/" << s.epochs - 1 << \
            " - volume "<< v.current_volume_ << "/"<< vsize(d.GetSplit()) << \
            " - batch " << j << "/" << v.slices_ / (v.n_channels_ * d.batch_size_) - 1;

            tm.stop();
            cout << " - Load time: " << tm.getTimeSec() << " - ";
            tm.reset();
            tm.start();

            // Preprocessing
            //x->div_(255.);

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
        j = 0;
        while (true) {
            if (v.current_slice_ >= vsize(v.indices_)) {
                j = 0; // Current volume ended
            }

            tm.reset();
            tm.start();
            if (!v.LoadBatch(x, y)) {
                break; // All volumes have been processed
            }

            cout << "Validation - Epoch " << i << "/" << s.epochs - 1 << \
                " - volume " << v.current_volume_ << "/" << vsize(d.GetSplit()) << \
                " - batch " << j << "/" << v.slices_ / (v.n_channels_ * d.batch_size_) - 1;

            tm.stop();
            cout << " - Load time: " << tm.getTimeSec() << " - ";
            tm.reset();
            tm.start();

            // Preprocessing
            // x->div_(255.);

            forward(s.net, { x });
            unique_ptr<Tensor> output(getOutput(getOut(s.net)[0]));

            // Compute Dice metric and optionally save the output images
            for (int k = 0; k < s.batch_size; ++k) {
                tensor pred = output->select({ to_string(k) });
                TensorToView(pred, pred_ecvl);
                pred_ecvl.colortype_ = ColorType::GRAY;
                pred_ecvl.channels_ = "xyc";

                tensor gt = y->select({ to_string(k) });
                TensorToView(gt, gt_ecvl);
                gt_ecvl.colortype_ = ColorType::GRAY;
                gt_ecvl.channels_ = "xyc";

                // NOTE: dice computed on downsampled images
                cout << "- Dice: " << evaluator.DiceCoefficient(pred_ecvl, gt_ecvl) << " ";

                if (s.save_images) {
                    pred->mult_(255.);
                    gt->mult_(255.);
                    for (int m = 0; m < s.n_channels; ++m) {

                        View<DataType::float32> view(pred_ecvl, {0, 0, m}, { pred_ecvl.Width(), pred_ecvl.Height(), 1 });
                        ImWrite(current_path / path(v.names_[0] + to_string(v.indices_[v.current_slice_ - s.batch_size + k] * v.stride_ + m) + ".png"), view);

                        View<DataType::float32> view_gt(gt_ecvl, { 0, 0, m }, { gt_ecvl.Width(), gt_ecvl.Height(), 1 });
                        ImWrite(current_path / path(v.names_[1] + to_string(v.indices_[v.current_slice_ - s.batch_size + k] * v.stride_ + m) + ".png"), view_gt);
                        //ImWrite(current_path / path(v.names_[1] + to_string(k) + "_" + to_string(m) + ".png"), view_gt);

                        //    // Save original image fused together with prediction (red mask) and ground truth (green mask)
                        //    ImRead(names[n], orig_img);
                        //ImRead(names[n + 1], orig_gt, ImReadMode::GRAYSCALE);
                        //ChangeColorSpace(orig_img, orig_img, ColorType::BGR);
                        //
                        //ResizeDim(pred_ecvl, pred_ecvl, { orig_img.Width(), orig_img.Height() }, InterpolationType::nearest);
                        //
                        //View<DataType::uint8> v_orig(orig_img);
                        //auto i_pred = pred_ecvl.Begin();
                        //auto i_gt = orig_gt.Begin<uint8_t>();
                        //
                        //for (int c = 0; c < pred_ecvl.Width(); ++c) {
                        //    for (int r = 0; r < pred_ecvl.Height(); ++r, ++i_pred, ++i_gt) {
                        //        // Replace in the green channel of the original image pixels that are 255 in the ground truth mask
                        //        if (*i_gt == 255) {
                        //            v_orig({ r, c, 1 }) = 255;
                        //        }
                        //        // Replace in the red channel of the original image pixels that are 255 in the prediction mask
                        //        if (*i_pred == 255) {
                        //            v_orig({ r, c, 2 }) = 255;
                        //        }
                        //    }
                        //}
                        //
                        //ImWrite(current_path / names[n].filename().replace_extension(".png"), orig_img);
                    }
                }
                delete pred;
                delete gt;
            }
            tm.stop();

            //cout << endl;
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

    delete x;
    delete y;

    return EXIT_SUCCESS;
}