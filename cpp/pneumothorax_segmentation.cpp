#include "metrics/metrics.h"
#include "utils/utils.h"

#include <iostream>

#include "ecvl/core/filesystem.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace ecvl;
using namespace ecvl::filesystem;
using namespace eddl;
using namespace std;

class PneumoDataset : public DLDataset
{
    vector<int> original_training_indices_;
    vector<int> black_indices_train_;
    vector<int> black_indices_valid_;
    int num_samples_training_, num_samples_validation_;

public:
    float best_metric_ = 0;
    PneumoDataset(const filesystem::path& filename,
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
        original_training_indices_ = GetSplit(SplitType::training);

        // Get number of training and validation samples. Add a 25% of samples with black ground truth.
        num_samples_training_ = static_cast<int>(vsize(original_training_indices_) * 1.25);
        num_samples_validation_ = static_cast<int>(vsize(GetSplit(SplitType::validation)) * 1.25);
    }

    void SetTrainingBlackMasks()
    {
        auto& s = GetSplit(SplitType::training);
        s = original_training_indices_;

        shuffle(black_indices_train_.begin(), black_indices_train_.end(), re_);
        vector<int> black_needed(black_indices_train_.begin(), black_indices_train_.begin() + (num_samples_training_ - vsize(original_training_indices_)));
        s.insert(s.end(), black_needed.begin(), black_needed.end());

        //shuffle of entire training list is performed by ResetBatch
    }

    void InitDatasetWithBlackMasks()
    {
        auto& tr = GetSplit(SplitType::training);
        auto& val = GetSplit(SplitType::validation);
        auto& test = GetSplit(SplitType::test);

        // Retrieve indices of images with a black ground truth
        vector<int> total_indices(vsize(samples_));
        iota(total_indices.begin(), total_indices.end(), 0);
        vector<int> training_validation_test_indices(tr);
        training_validation_test_indices.insert(training_validation_test_indices.end(), test.begin(), test.end());
        training_validation_test_indices.insert(training_validation_test_indices.end(), val.begin(), val.end());
        sort(training_validation_test_indices.begin(), training_validation_test_indices.end());
        vector<int> black;
        set_difference(total_indices.begin(), total_indices.end(), training_validation_test_indices.begin(), training_validation_test_indices.end(), std::inserter(black, black.begin()));

        // Split indices of images with a black ground truth for training and validation
        black_indices_train_ = vector<int>(black.begin(), black.end() - (num_samples_validation_ - vsize(val)));
        black_indices_valid_ = vector<int>(black.end() - (num_samples_validation_ - vsize(val)), black.end());

        val.insert(val.end(), black_indices_valid_.begin(), black_indices_valid_.end());
        SetTrainingBlackMasks();

        for (int id = 0; id < vsize(split_); ++id) {
            split_[id].SetNumBatches(batch_size_);
            split_[id].SetLastBatch(batch_size_);
            InitTC(id);
        }
    }
};

void Inference(const string& type, PneumoDataset& d, const Settings& s, const int num_batches, const int epoch, const path& current_path)
{
    float mean_metric = 0;
    View<DataType::float32> pred_t, target_t;
    Image orig_img, orig_gt;
    Eval evaluator;
    ofstream of;
    layer out = getOut(s.net)[0];

    cout << "Starting " << type << ":" << endl;
    // Resize to batch size if we have done a previous resize
    if (d.split_[d.current_split_].last_batch_ != s.batch_size) {
        s.net->resize(s.batch_size);
    }
    d.SetSplit(type);
    d.ResetBatch(d.current_split_);
    evaluator.ResetEval();

    auto str = type == "validation" ? "/" + to_string(s.epochs - 1) : "";
    d.Start();
    // Validation for each batch
    for (int j = 0; j < num_batches; ++j) {
        cout << type << ": Epoch " << epoch << str << " (batch " << j << "/" << num_batches - 1 << ") - ";
        cout << "|fifo| " << d.GetQueueSize();

        // Load a batch
        auto [samples, x, y] = d.GetBatch();

        auto current_bs = x->shape[0];
        // if it's the last batch and the number of samples doesn't fit the batch size, resize the network
        if (j == num_batches - 1 && current_bs != s.batch_size) {
            s.net->resize(current_bs);
        }

        // Evaluate batch
        set_mode(s.net, TSMODE);
        forward(s.net, { x.get() }); // forward does not require reset_loss
        unique_ptr<Tensor> output(getOutput(out));

        // Compute Dice metric and optionally save the output images
        for (int k = 0, n = 0; k < current_bs; ++k, ++n) {
            unique_ptr<Tensor> pred(output->select({ to_string(k) }));
            TensorToView(pred.get(), pred_t);
            pred_t.colortype_ = ColorType::GRAY;
            pred_t.channels_ = "xyc";

            if (type == "validation") {
                unique_ptr<Tensor> target(y->select({ to_string(k) }));
                TensorToView(target.get(), target_t);
                target_t.colortype_ = ColorType::GRAY;
                target_t.channels_ = "xyc";

                cout << "- Dice: " << evaluator.DiceCoefficient(pred_t, target_t) << " ";
            }
            else {
                auto i_pred = pred_t.ContiguousBegin<float>(), e_pred = pred_t.ContiguousEnd<float>();

                for (; i_pred != e_pred; ++i_pred) {
                    *i_pred = ((*i_pred) < 0.50) ? 0.f : 1.f;
                }
            }

            if (s.save_images) {
                Image pred_i;
                pred->mult_(255.);
                ImRead(samples[k].location_[0], orig_img);
                ResizeDim(pred_t, pred_i, { orig_img.Width(), orig_img.Height() }, InterpolationType::nearest);

                if (type == "validation") {
                    // Save original image fused together with prediction (red mask) and ground truth (green mask)
                    ImRead(samples[k].label_path_.value(), orig_gt, ImReadMode::GRAYSCALE);
                    ChangeColorSpace(orig_img, orig_img, ColorType::BGR);

                    View<DataType::uint8> v_orig(orig_img);
                    auto i_pred = pred_i.Begin<float>();
                    auto i_gt = orig_gt.Begin<uint8_t>();

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

                    ImWrite(current_path / samples[k].location_[0].filename().replace_extension(".png"), orig_img);
                }
                else {
                    ImWrite(current_path / samples[k].location_[0].filename().replace_extension(".png"), pred_i);
                }
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
    }

    of.open(s.exp_name + "_stats.txt", ios::out | ios::app);
    of << "Epoch " << epoch << " - Total " << type << " Dice Coefficient: " << mean_metric << endl;
    of.close();
}

int main(int argc, char* argv[])
{
    // Default settings, they can be changed from command line
    // num_classes, size, model, loss, lr, exp_name, dataset_path, epochs, batch_size, workers, queue_ratio, gpu, input_channels
    Settings s(1, { 512,512 }, "SegNet", "dice", 0.0001f, "pneumothorax_segmentation", "", 50, 2, 6, 6, {}, 1);
    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
    }

    layer out = getOut(s.net)[0];
    if (typeid(out) != typeid(LActivation)){
        out = Sigmoid(out);
        s.net = Model({ s.net->lin[0] }, { out });
    }

    // Build model
    build(s.net,
        adam(s.lr),      // Optimizer
        { s.loss },      // Loss
        { "dice" },      // Metric
        s.cs,            // Computing Service
        s.random_weights // Randomly initialize network weights
    );

    // View model
    summary(s.net);
    plot(s.net, s.exp_name + ".pdf");
    setlogfile(s.net, s.exp_name);

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
        AugRotate({ -30, 30 }),
        AugToFloat32(255, 255)
        );

    auto validation_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size, InterpolationType::cubic),
        AugToFloat32(255, 255)
        );

    // Replace the random seed with a fixed one to have reproducible experiments
    // AugmentationParam::SetSeed(50);

    DatasetAugmentations dataset_augmentations{ {training_augs, validation_augs, validation_augs } }; // use the same augmentations for validation and test

    // Read the dataset
    cout << "Reading dataset" << endl;
    PneumoDataset d(s.dataset_path, s.batch_size, dataset_augmentations, ColorType::GRAY, ColorType::GRAY, s.workers, s.queue_ratio, { true, false, false });
    d.InitDatasetWithBlackMasks();

    int num_batches_training = d.GetNumBatches(SplitType::training);
    int num_batches_validation = d.GetNumBatches(SplitType::validation);
    int num_batches_test = d.GetNumBatches(SplitType::test);

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

            // Reset errors for train_batch
            reset_loss(s.net);

            // Resize to batch size if we have done a previous resize
            if (d.split_[d.current_split_].last_batch_ != s.batch_size) {
                s.net->resize(s.batch_size);
            }

            d.SetTrainingBlackMasks();

            // Reset and shuffle training list
            d.ResetBatch(d.current_split_, true);

            d.Start();
            // Feed batches to the model
            for (int j = 0; j < num_batches_training; ++j) {
                tm.reset();
                tm.start();
                cout << "Epoch " << e << "/" << s.epochs - 1 << " (batch " << j << "/" << num_batches_training - 1 << ") - ";
                cout << "|fifo| " << d.GetQueueSize() << " - ";

                // Load a batch
                auto [samples, x, y] = d.GetBatch();

                auto current_bs = x->shape[0];
                // if it's the last batch and the number of samples doesn't fit the batch size, resize the network
                if (j == num_batches_training - 1 && current_bs != s.batch_size) {
                    s.net->resize(current_bs);
                }

                // Train batch
                // set_mode(s.net, TRMODE) // not necessary because it's already inside the train_batch
                train_batch(s.net, { x.get() }, { y.get() });

                // Print errors
                print_loss(s.net, j);

                tm.stop();
                cout << " - Elapsed time: " << tm.getTimeSec() << endl;
            }
            d.Stop();

            Inference("validation", d, s, num_batches_validation, e, current_path);

            tm_epoch.stop();
            cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;
        }
    }

    int epoch = s.skip_train ? s.resume : s.epochs;
    auto current_path{ s.result_dir / ("Test - epoch " + to_string(epoch)) };
    if (s.save_images) {
        create_directory(current_path);
    }
    Inference("test", d, s, num_batches_test, epoch, current_path);

    return EXIT_SUCCESS;
}