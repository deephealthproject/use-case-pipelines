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

class KidneyDataset : public DLDataset
{
public:
    array<int, 3> total_slices_{ 0,0,0 };
    float best_metric_ = 0;

    KidneyDataset(const int in_channels, const filesystem::path& filename,
        const int batch_size,
        DatasetAugmentations augs,
        ColorType ctype = ColorType::RGB,
        ColorType ctype_gt = ColorType::GRAY,
        unsigned num_workers = 1,
        double queue_ratio_size = 1.,
        vector<bool> drop_last = {},
        bool verify = false) :

        DLDataset{ filename, batch_size, augs, ctype, ctype_gt, num_workers, queue_ratio_size, drop_last, verify }
    {
        SetNumChannels(in_channels, in_channels);
        // Compute the total number of slices for each image
        for (int i = 0; i < vsize(split_); ++i) {
            for (auto& j : GetSplit(i)) {
                total_slices_[i] += vsize(samples_[j].location_);
            }
        }
    }

    void ProduceImageLabel(DatasetAugmentations& augs, Sample& elem) override
    {
        Tensor* label_tensor = nullptr, * image_tensor = nullptr;
        Image img = elem.LoadImage(ctype_, false);
        Image gt = elem.LoadImage(ctype_gt_, true);
        const int slices = img.Channels();

        // Reverse the order of the image slices - due to the inverse order of the ground truth
        vector<Image> tmp;
        for (int i = slices - 1; i >= 0; --i) {
            View<DataType::int16> img_v(img, { 0,0,i }, { img.Width(),img.Height(),1 });
            img_v.channels_ = "xyc";
            tmp.push_back(img_v);
        }
        Stack(tmp, img);

        // Apply chain of augmentations to sample image and corresponding ground truth
        augs.Apply(current_split_, img, gt);

        // Push the slice and its ground truth to the queue
        for (int cur_slice = 0; cur_slice < slices; ++cur_slice) {
            View<DataType::int16> v_volume(img, { 0, 0, cur_slice}, { img.Width(), img.Height(), n_channels_ });
            View<DataType::int16> v_gt(gt, { 0, 0, cur_slice}, { gt.Width(), gt.Height(), n_channels_ });
            ImageToTensor(v_volume, image_tensor);
            ImageToTensor(v_gt, label_tensor);

            queue_.Push(elem, image_tensor, label_tensor);
        }
    }
};

void Inference(const string& type, KidneyDataset& d, const Settings& s, const int epoch, const path& current_path)
{
    float mean_metric = 0;
    Image orig_img_i, pred_i, target_i;
    Eval evaluator;
    ofstream of;
    layer out = getOut(s.net)[0];

    cout << "Starting " << type << ":" << endl;

    d.SetSplit(type);
    d.ResetBatch(d.current_split_);
    evaluator.ResetEval();

    auto str = type == "validation" ? "/" + to_string(s.epochs - 1) : "";
    auto index = type == "validation" ? 1 : 2;
    d.Start();

    // Validation for each batch
    int n = 0;
    auto num_batches = d.total_slices_[index] / (d.n_channels_ * s.batch_size);

    for (int j = 0; j < num_batches; ++j) {
        // Load a batch
        auto [samples, x, y] = d.GetBatch();
        cout << type << ": Epoch " << epoch << str << " (batch " << j << "/" << num_batches - 1 << ") - ";
        cout << "|fifo| " << d.GetQueueSize() << " - ";

        auto current_bs = x->shape[0];

        // Evaluate batch
        forward(s.net, { x.get() }); // forward does not require reset_loss
        unique_ptr<Tensor> output(getOutput(out));

        // Compute Dice metric and optionally save the output images
        for (int k = 0; k < current_bs; ++k, ++n) {
            unique_ptr<Tensor> pred(output->select({ to_string(k) }));
            TensorToImage(pred.get(), pred_i);
            pred_i.colortype_ = ColorType::GRAY;
            pred_i.channels_ = "xyc";

            unique_ptr<Tensor> target(y->select({ to_string(k) }));
            TensorToImage(target.get(), target_i);
            target_i.colortype_ = ColorType::GRAY;
            target_i.channels_ = "xyc";

            unique_ptr<Tensor> orig_img(x->select({ to_string(k) }));
            TensorToImage(orig_img.get(), orig_img_i);
            orig_img_i.colortype_ = ColorType::GRAY;
            orig_img_i.channels_ = "xyc";

            // Resize to original size
            ResizeDim(orig_img_i, orig_img_i, { samples[k].size_[0], samples[k].size_[1] }, InterpolationType::nearest);
            ResizeDim(pred_i, pred_i, { orig_img_i.Width(), orig_img_i.Height() }, InterpolationType::nearest);
            ResizeDim(target_i, target_i, { orig_img_i.Width(), orig_img_i.Height() }, InterpolationType::nearest);

            cout << "Dice: " << evaluator.DiceCoefficient(pred_i, target_i) << " ";

            if (s.save_images) {
                Mul(pred_i, 255, pred_i);
                Mul(target_i, 255, target_i);
                ChangeColorSpace(orig_img_i, orig_img_i, ColorType::BGR);
                ScaleTo(orig_img_i, orig_img_i, 0, 255);

                // Save original image fused together with prediction (red mask) and ground truth (green mask)
                View<DataType::float32> v_orig(orig_img_i);
                auto i_pred = pred_i.Begin<float>();
                auto i_gt = target_i.Begin<float>();

                for (int c = 0; c < pred_i.Width(); ++c) {
                    for (int r = 0; r < pred_i.Height(); ++r, ++i_pred, ++i_gt) {
                        // Replace in the green channel of the original image pixels that are 255 in the ground truth mask
                        if (*i_gt == 255) {
                            v_orig({ r, c, 1 }) = 255;
                        }
                        // Replace in the red channel of the original image pixels that are 255 in the prediction mask
                        if (*i_pred == 255) {
                            v_orig({ r, c, 2 }) = 255;
                        }
                    }
                }

                auto filename = samples[k].location_[0].parent_path().stem();
                filename += "_" + to_string(n) + ".png";
                ImWrite(current_path / filename, orig_img_i);
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
    Settings s(1, { 224,224 }, "UNet", "dice", 0.00001f, "kidney_segmentation", "", 100, 6, 6, 10, {}, 1);
    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
    }

    // if an imported model without final activation is employed, add the sigmoid
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
    setlogfile(s.net, "kidney_segmentation");

    // Set augmentations for training and validation
    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size, InterpolationType::cubic),
        AugMirror(.5),
        OneOfAugmentationContainer(
            0.3,
            AugGammaContrast({ 0,3 }),
            AugBrightness({ 0, 30 })
        ),
        OneOfAugmentationContainer(
            0.3,
            AugElasticTransform({ 30, 120 }, { 3, 6 }),
            AugGridDistortion({ 2, 5 }, { -0.3f, 0.3f }),
            AugOpticalDistortion({ -0.3f, 0.3f }, { -0.1f, 0.1f })
        ),
        AugRotate({ -30, 30 })
        );

    auto validation_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size, InterpolationType::cubic)
        );

    DatasetAugmentations dataset_augmentations{ {training_augs, validation_augs, validation_augs } }; // use the same augmentations for validation and test

    // Read the dataset
    cout << "Reading dataset" << endl;
    KidneyDataset d(s.in_channels, s.dataset_path, s.batch_size, dataset_augmentations, ColorType::none, ColorType::none, s.workers, s.queue_ratio, { true, false, false });

    cv::TickMeter tm, tm_epoch;

    if (!s.skip_train) {
        cout << "Starting training" << endl;
        for (int e = s.resume; e < s.epochs; ++e) {
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

            // Feed batches to the model
            auto num_batches_training = d.total_slices_[0] / (d.n_channels_ * s.batch_size);
            d.Start();
            for (int j = 0; j < num_batches_training; ++j) {
                tm.reset();
                tm.start();

                cout << "Epoch " << e << "/" << s.epochs - 1 << " (batch " << j << "/" << num_batches_training - 1 << ") - ";
                cout << "|fifo| " << d.GetQueueSize();

                // Load a batch
                auto [samples, x, y] = d.GetBatch();

                // Train batch
                // set_mode(s.net, TRMODE) // not necessary because it's already inside the train_batch
                train_batch(s.net, { x.get() }, { y.get() });

                // Print errors
                print_loss(s.net, j);

                tm.stop();
                cout << " - Elapsed time: " << tm.getTimeSec() << endl;
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