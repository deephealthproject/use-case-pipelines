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
    vector<int> black_indices_train_;
    vector<int> black_indices_valid_;
    int m_i_ = 0, b_i_ = 0, i_ = 0;

public:
    float best_metric_ = 0;
    PneumoDataset(const filesystem::path& filename,
        const int batch_size,
        DatasetAugmentations augs = DatasetAugmentations(),
        ColorType ctype = ColorType::RGB,
        ColorType ctype_gt = ColorType::GRAY,
        int num_workers = 1,
        int queue_ratio_size = 1,
        vector<bool> drop_last = {},
        bool verify = false) :

        DLDataset{ filename, batch_size, augs, ctype, ctype_gt, num_workers, queue_ratio_size, drop_last, verify }
    {}

    void SetMasks(const vector<int>& black_indices_train, const vector<int>& black_indices_valid)
    {
        black_indices_train_ = black_indices_train;
        black_indices_valid_ = black_indices_valid;
    }

    void ResetIndices()
    {
        i_ = 0;
        m_i_ = 0;
        b_i_ = 0;
    }

    // Custom LoadBatch for pneumothorax specific problem.
    void ProduceImageLabel(DatasetAugmentations& augs, Sample& elem) override
    {
        if (!(split_[current_split_].split_type_ == SplitType::test)) {
            bool expr = true;
            vector<int> mask_indices = GetSplit();
            vector<int> black_indices;
            if (split_[current_split_].split_type_ == SplitType::training) {
                black_indices = black_indices_train_;

                // in training, check if you can take other black ground truth images..
                // b_i < mask_indices.size() * 0.25
                if (mask_indices.size() * 1.25 - i_ > mask_indices.size() - m_i_) {
                    // generate a random value between 0 and 1. With a 80% probability we take a sample with a ground truth with mask if there are still some available.
                    auto prob = std::uniform_real_distribution<>(0, 1)(re_);
                    expr = prob >= 0.2 && m_i_ < mask_indices.size();
                }
                // ..otherwise, you have to take a ground truth with mask
            }
            else {
                black_indices = black_indices_valid_;
                // in validation, first take all the samples with the ground truth with mask and then all those with the black ground truth.
                expr = m_i_ < mask_indices.size();
            }

            int index = expr ? mask_indices[m_i_++] : black_indices[b_i_++];

            // override the sample loaded by default
            elem = samples_[index];
        }

        // Read the image
        Image img = elem.LoadImage(ctype_, false);

        LabelImage* label = nullptr;
        // Read the ground truth
        if (!split_[current_split_].no_label_) {
            label = new LabelImage();
            Image gt = elem.LoadImage(ctype_gt_, true);
            // Apply chain of augmentations to sample image and corresponding ground truth
            augs.Apply(current_split_, img, gt);
            label->gt = gt;
        }
        else {
            augs.Apply(current_split_, img);
        }
        queue_.Push(elem, img, label);
        ++i_;
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
    d.ResetIndices();
    evaluator.ResetEval();

    auto str = !type.compare("validation") ? "/" + s.epochs - 1 : "";
    d.Start();
    // Validation for each batch
    for (int j = 0; j < num_batches; ++j) {
        cout << type << ": Epoch " << epoch << str << " (batch " << j << "/" << num_batches - 1 << ") - ";
        cout << "|fifo| " << d.GetQueueSize() << " - ";

        // Load a batch
        auto [samples, x, y] = d.GetBatch();

        auto current_bs = x->shape[0];
        // if it's the last batch and the number of samples doesn't fit the batch size, resize the network
        if (j == num_batches - 1 && current_bs != s.batch_size) {
            s.net->resize(current_bs);
        }

        // Evaluate batch
        forward(s.net, { x.get() }); // forward does not require reset_loss
        unique_ptr<Tensor> output(getOutput(out));

        // Compute Dice metric and optionally save the output images
        for (int k = 0, n = 0; k < s.batch_size; ++k, ++n) {
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
                pred->mult_(255.);
                if (type == "validation") {
                    // Save original image fused together with prediction (red mask) and ground truth (green mask)
                    ImRead(samples[k].location_[0], orig_img);
                    ImRead(samples[k].label_path_.value(), orig_gt, ImReadMode::GRAYSCALE);
                    ChangeColorSpace(orig_img, orig_img, ColorType::BGR);

                    ResizeDim(pred_t, pred_t, { orig_img.Width(), orig_img.Height() }, InterpolationType::nearest);

                    View<DataType::uint8> v_orig(orig_img);
                    auto i_pred = pred_t.Begin();
                    auto i_gt = orig_gt.Begin<uint8_t>();

                    for (int c = 0; c < pred_t.Width(); ++c) {
                        for (int r = 0; r < pred_t.Height(); ++r, ++i_pred, ++i_gt) {
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
                    pred_t = v_orig;
                }

                ImWrite(current_path / samples[k].location_[0].filename().replace_extension(".png"), pred_t);
            }
        }
        cout << endl;
    }
    d.Stop();

    mean_metric = evaluator.MeanMetric();
    cout << "----------------------------------------" << endl;
    cout << "Epoch " << epoch << " - Mean " << type << " Dice Coefficient: " << mean_metric << endl;
    cout << "----------------------------------------" << endl;

    if (!type.compare("validation")) {
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
    // num_classes, size, model, loss, lr, exp_name, dataset_path, epochs, batch_size, workers, queue_ratio
    Settings s(1, { 512,512 }, "SegNet", "cross_entropy", 0.0001f, "pneumothorax_segmentation", "", 50, 2, 6, 6, {}, 1);
    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
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
        AugRotate({ -10, 10 }),
        AugBrightness({ 0, 30 }),
        AugGammaContrast({ 0,3 }),
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
    PneumoDataset d(s.dataset_path, s.batch_size, dataset_augmentations, ColorType::GRAY, ColorType::GRAY, s.workers, s.queue_ratio, { true, true });

    // Retrieve indices of images with a black ground truth
    vector<int> total_indices(d.samples_.size());
    iota(total_indices.begin(), total_indices.end(), 0);
    vector<int> training_validation_test_indices(d.GetSplit(SplitType::training));
    training_validation_test_indices.insert(training_validation_test_indices.end(), d.GetSplit(SplitType::test).begin(), d.GetSplit(SplitType::test).end());
    training_validation_test_indices.insert(training_validation_test_indices.end(), d.GetSplit(SplitType::validation).begin(), d.GetSplit(SplitType::validation).end());
    sort(training_validation_test_indices.begin(), training_validation_test_indices.end());
    vector<int> black;
    set_difference(total_indices.begin(), total_indices.end(), training_validation_test_indices.begin(), training_validation_test_indices.end(), std::inserter(black, black.begin()));
    // Get number of training samples. Add a 25% of training samples with black ground truth.
    int num_samples_training = static_cast<int>(d.GetSplit().size() * 1.25);
    int num_batches_training = num_samples_training / s.batch_size;

    d.SetSplit(SplitType::validation);

    // Get number of validation samples. Add a 25% of validation samples with black ground truth.
    int num_samples_validation = static_cast<int>(d.GetSplit().size() * 1.25);
    int num_batches_validation = num_samples_validation / s.batch_size;

    // Split indices of images with a black ground truth for training and validation
    vector<int> black_training(black.begin(), black.end() - (num_samples_validation - d.GetSplit().size()));
    vector<int> black_validation(black.end() - (num_samples_validation - d.GetSplit().size()), black.end());
    d.SetMasks(black_training, black_validation);

    mt19937 g(std::random_device{}());
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

            // Reset and shuffle training list
            d.ResetBatch(d.current_split_, true);
            d.ResetIndices();

            // Shuffle training list
            shuffle(black_training.begin(), black_training.end(), g);

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
                if (j == num_batches_validation - 1 && current_bs != s.batch_size) {
                    s.net->resize(current_bs);
                }

                // Train batch
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

    // Get number of test samples.
    d.SetSplit(SplitType::test);
    int num_samples_test = vsize(d.GetSplit());
    int num_batches_test = num_samples_test / s.batch_size;
    int epoch = s.skip_train ? s.resume : s.epochs;
    auto current_path{ s.result_dir / ("Test - epoch " + to_string(epoch)) };
    if (s.save_images) {
        create_directory(current_path);
    }
    Inference("test", d, s, num_batches_test, epoch, current_path);

    return EXIT_SUCCESS;
}