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
class MSVolume : public DLDataset
{
public:
    const int stride_; // Stride represents the jump to make to reach the next volume slice. Default value in constructor
    std::mt19937 g_;

    // Volume related variables
    array<int, 3> total_slices_{ 0,0,0 };
    float best_metric_ = 0;

    MSVolume(const int in_channels, const filesystem::path& filename,
        const int batch_size,
        const DatasetAugmentations& augs,
        ColorType ctype = ColorType::RGB,
        ColorType ctype_gt = ColorType::GRAY,
        unsigned num_workers = 1,
        double queue_ratio_size = 1.,
        const unordered_map<string, bool>& drop_last = {}) :

        DLDataset{ filename, batch_size, augs, ctype, ctype_gt, num_workers, queue_ratio_size, drop_last },
        stride_{ in_channels }, g_(std::random_device{}())
    {
        n_channels_ = in_channels;
        n_channels_gt_ = in_channels;
        SetTensorsShape();
        for (int i = 0; i < vsize(split_); ++i) {
            for (auto& j : GetSplit(i)) {
                total_slices_[i] += stoi(samples_[j].values_.value()[0]);
            }
        }
    }

    MSVolume(const int in_channels, const int stride, const filesystem::path& filename,
        const int batch_size,
        const DatasetAugmentations& augs,
        ColorType ctype = ColorType::RGB,
        ColorType ctype_gt = ColorType::GRAY,
        unsigned num_workers = 1,
        double queue_ratio_size = 1.,
        const unordered_map<string, bool>& drop_last = {}) :

        DLDataset{ filename, batch_size, augs, ctype, ctype_gt, num_workers, queue_ratio_size, drop_last },
        stride_{ stride }, g_(std::random_device{}())
    {
        n_channels_ = in_channels;
        n_channels_gt_ = in_channels;
        SetTensorsShape();
    }

    template<DataType DT>
    void RealProduce(DatasetAugmentations& augs, Sample& elem)
    {
        Tensor* label_tensor = nullptr, * image_tensor = nullptr;

        Image img = elem.LoadImage(ctype_, false);
        Image gt = elem.LoadImage(ctype_gt_, true);
        auto slices = img.Channels();
        vector<int> indices = vector<int>(slices / stride_);
        iota(indices.begin(), indices.end(), 0);

        augs.Apply(current_split_, img, gt);

        // Apply chain of augmentations to sample image and corresponding ground truth
        for (int cur_slice = 0; cur_slice < vsize(indices); ++cur_slice) {
            View<DataType::float32> v_volume(img, { 0, 0, indices[cur_slice] * stride_ }, { img.Width(), img.Height(), n_channels_ });
            View<DT> v_gt(gt, { 0, 0, indices[cur_slice] * stride_ }, { gt.Width(), gt.Height(), n_channels_ });

            ImageToTensor(v_volume, image_tensor);
            ImageToTensor(v_gt, label_tensor);
            queue_.Push(elem, image_tensor, label_tensor);
        }
    }

    void ProduceImageLabel(DatasetAugmentations& augs, Sample& elem) override
    {
        if (elem.values_.value()[1] == "float32") {
            RealProduce<DataType::float32>(augs, elem);
        }
        else if (elem.values_.value()[1] == "uint16") {
            RealProduce<DataType::uint16>(augs, elem);
        }
    }
};

void Inference(const string& type, MSVolume& d, const Settings& s, const int epoch, const path& current_path)
{
    float mean_metric = 0;
    View<DataType::float32> pred_t, target_t;
    Image orig_img, orig_gt;
    Eval evaluator;
    ofstream of;
    layer out = getOut(s.net)[0];

    cout << "Starting " << type << ":" << endl;
    d.SetSplit(type);
    d.ResetBatch(d.current_split_);

    auto str = type == "validation" ? "/" + to_string(s.epochs - 1) : "";
    auto index = type == "validation" ? 1 : 2;

    // Validation for each batch
    int n = 0;
    auto num = d.total_slices_[index] / (d.n_channels_ * s.batch_size);

    d.Start();
    for (int j = 0; j < num; ++j) {
        // Load a batch
        auto [samples, x, y] = d.GetBatch();
        cout << type << ": Epoch " << epoch << str << " (batch " << j << "/" << num - 1 << ") - ";
        cout << "|fifo| " << d.GetQueueSize() << " - ";

        auto current_bs = x->shape[0];

        // Evaluate batch
        forward(s.net, { x.get() }); // forward does not require reset_loss

        unique_ptr<Tensor> output(getOutput(out));

        // Compute Dice metric and optionally save the output images
        for (int k = 0; k < current_bs; ++k, ++n) {
            unique_ptr<Tensor> pred(output->select({ to_string(k) }));
            TensorToView(pred.get(), pred_t);
            pred_t.colortype_ = ColorType::GRAY;
            pred_t.channels_ = "xyc";

            unique_ptr<Tensor> target(y->select({ to_string(k) }));
            TensorToView(target.get(), target_t);
            target_t.colortype_ = ColorType::GRAY;
            target_t.channels_ = "xyc";

            cout << "Dice: " << evaluator.DiceCoefficient(pred_t, target_t) << " ";

            if (s.save_images) {
                Image pred_i, target_i;
                pred->mult_(255.);
                target->mult_(255.);
                unique_ptr<Tensor> orig_img_t(x->select({ to_string(k) }));
                TensorToImage(orig_img_t.get(), orig_img);
                orig_img.colortype_ = ColorType::GRAY;
                orig_img.channels_ = "xyc";
                ResizeDim(orig_img, orig_img, { samples[k].size_[0], samples[k].size_[1] }, InterpolationType::nearest);
                ResizeDim(pred_t, pred_i, { orig_img.Width(), orig_img.Height() }, InterpolationType::nearest);
                ChangeColorSpace(pred_i, pred_i, ColorType::BGR);

                // Save original image fused together with prediction (red mask) and ground truth (green mask)
                ResizeDim(target_t, target_i, { orig_img.Width(), orig_img.Height() }, InterpolationType::nearest);
                ChangeColorSpace(orig_img, orig_img, ColorType::BGR);
                ChangeColorSpace(target_i, target_i, ColorType::BGR);
                ScaleTo(orig_img, orig_img, 0, 255);

                View<DataType::float32> v_orig(orig_img);
                auto i_pred = pred_i.Begin<float>();
                auto i_gt = target_i.Begin<float>();

                for (int r = 0; r < pred_i.Height(); ++r) {
                    for (int c = 0; c < pred_i.Width(); ++c, ++i_pred, ++i_gt) {
                        // Replace in the green channel of the original image pixels that are 255 in the ground truth mask
                        if (*i_gt == 255) {
                            v_orig({ c, r, 1 }) = 255;
                        }
                        // Replace in the red channel of the original image pixels that are 255 in the prediction mask
                        if (*i_pred == 255) {
                            v_orig({ c, r, 2 }) = 255;
                        }
                    }
                }

                auto filename = samples[k].location_[0].parent_path().stem();
                filename += "_" + to_string(n) + ".png";
                ImWrite(current_path / filename, orig_img);
            }
        }
        cout << endl;
    }
    d.Stop();

    mean_metric = evaluator.MeanMetric();
    cout << "----------------------------------------" << endl;
    cout << "Epoch " << epoch << " - Mean " << type << " Dice Coefficient: " << mean_metric << endl;
    cout << "----------------------------------------" << endl;

    if (type == "validation") {
        if (mean_metric > d.best_metric_) {
            cout << "Saving weights..." << endl;
            save_net_to_onnx_file(s.net, (s.checkpoint_dir / (s.exp_name + "_epoch_" + to_string(epoch) + ".onnx")).string());
            d.best_metric_ = mean_metric;
        }

        of.open(s.exp_name + "_stats.txt", ios::out | ios::app);
        of << "Epoch " << epoch << " - Total " << type << " Dice Coefficient: " << mean_metric << endl;
        of.close();
    }
}

int main(int argc, char* argv[])
{
    time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    cout << "Start at " << ctime(&now) << endl;

    // Default settings, they can be changed from command line
    // num_classes, size, model, loss, lr, exp_name, dataset_path, epochs, batch_size, workers, queue_ratio, gpu, input_channels
    Settings s(1, { 256,256 }, "Nabla", "bce", 0.0001f, "ms_segmentation", "", 100, 16, 6, 20, {}, 1);
    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
    }

    layer out = getOut(s.net)[0];
    if (typeid(*out) != typeid(LActivation)) {
        out = Sigmoid(out);
        s.net = Model(s.net->lin, { out });
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
    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size, InterpolationType::cubic),
        AugMirror(.5),
        AugRotate({ -30, 30 })
        );

    auto validation_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size, InterpolationType::cubic)
        );

    DatasetAugmentations dataset_augmentations{ {training_augs, validation_augs, validation_augs } }; // use the same augmentations for validation and test

    // Read the dataset
    cout << "Reading dataset" << endl;
    MSVolume d(s.in_channels, s.dataset_path, s.batch_size, dataset_augmentations, ColorType::none, ColorType::none, s.workers, s.queue_ratio, { {"training", false}, {"validation", false},{"test", false} });

    auto num_batches_training = d.total_slices_[0] / (d.n_channels_ * s.batch_size);

    vector<int> indices(s.batch_size);
    iota(indices.begin(), indices.end(), 0);

    Image input_image_ecvl;
    Image pred_ecvl;
    Image gt_ecvl;
    Image orig_img, orig_gt;
    float best_metric = 0;
    ofstream of;
    Eval evaluator;
    cv::TickMeter tm, tm_epoch;

    if (!s.skip_train) {
        cout << "Starting training" << endl;
        for (int e = s.resume; e < s.epochs; ++e) {
            //v.Reset();

            tm_epoch.reset();
            tm_epoch.start();
            d.SetSplit(SplitType::training);
            auto current_path{ s.result_dir / ("Epoch_" + to_string(e)) };
            if (s.save_images) {
                create_directory(current_path);
            }

            // Reset errors
            reset_loss(s.net);

            // Reset and shuffle training list
            d.ResetBatch(d.current_split_, true);

            d.Start();
            // Feed batches to the model
            for (int j = 0; j < num_batches_training; ++j) {
                tm.reset();
                tm.start();

                // Load a batch
                auto [samples, x, y] = d.GetBatch();

                cout << "Epoch " << e << "/" << s.epochs - 1 << " (batch " << j << "/" << num_batches_training - 1 << ") - ";
                cout << "|fifo| " << d.GetQueueSize();

                tm.stop();
                cout << " - Load time: " << tm.getTimeSec() << " - ";
                tm.reset();
                tm.start();

                // Train batch
                train_batch(s.net, { x.get() }, { y.get() });

                print_loss(s.net, j);
                tm.stop();
                cout << "Train time: " << tm.getTimeSec() << endl;
            }
            d.Stop();

            set_mode(s.net, TSMODE);
            Inference("validation", d, s, e, current_path);

            tm_epoch.stop();
            cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;
        }
    }
    int epoch = s.skip_train ? s.resume : s.epochs;
    auto current_path{ s.result_dir / ("Test - epoch " + to_string(epoch)) };
    if (s.save_images) {
        create_directory(current_path);
    }
    set_mode(s.net, TSMODE);
    Inference("test", d, s, epoch, current_path);

    now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    cout << "End at " << ctime(&now) << endl;

    return EXIT_SUCCESS;
}