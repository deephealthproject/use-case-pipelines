#include "utils/utils.h"

#include <iostream>

#include "ecvl/core/filesystem.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace ecvl;
using namespace ecvl::filesystem;
using namespace eddl;
using namespace std;

class KidneyDataset : public DLDataset
{
public:
    KidneyDataset(const filesystem::path& filename,
        const int batch_size,
        DatasetAugmentations augs,
        ColorType ctype = ColorType::none,
        ColorType ctype_gt = ColorType::none,
        unsigned num_workers = 1,
        double queue_ratio_size = 1.,
        const unordered_map<string, bool>& drop_last = {},
        bool verify = false) :

        DLDataset{ filename, batch_size, augs, ctype, ctype_gt, num_workers, queue_ratio_size, drop_last, verify }
    {}

    void ProduceImageLabel(DatasetAugmentations& augs, Sample& elem) override
    {
        Tensor* label_tensor = nullptr, * image_tensor = nullptr;
        Image img = elem.LoadImage(ctype_, false);
        Image dst(img.dims_, DataType::uint8, img.channels_, img.colortype_, img.spacings_);

        ConstView<DataType::int16> src_v(img);
        View<DataType::uint8> dst_v(dst);

        // Find min and max of the image
        auto max = *std::max_element(src_v.Begin(), src_v.End());
        auto min = *std::min_element(src_v.Begin(), src_v.End());

        // Bring the image in range 0-255
        /*  OldRange = (OldMax - OldMin)
            NewRange = (NewMax - NewMin)
            NewValue = int((((OldValue - OldMin) * NewRange) / OldRange) + NewMin)*/
        auto dst_it = dst_v.Begin();
        auto src_it = src_v.Begin();
        auto src_end = src_v.End();
        for (; src_it != src_end; ++src_it, ++dst_it) {
            (*dst_it) = (((*src_it) - min) * 255) / (max - min);
        }

        // Apply chain of augmentations to sample image
        augs.Apply(current_split_, dst);

        // Convert Image and label in EDDL Tensor
        ImageToTensor(dst, image_tensor);
        ToTensorPlane(elem.label_.value(), label_tensor);

        // Push them to the queue
        queue_.Push(elem, image_tensor, label_tensor);
    }
};

void Inference(const string& type, KidneyDataset& d, const Settings& s, const int num_batches, const int epoch, const path& current_path, float& best_metric, float& best_metric_patients)
{
    float ca = 0.f, mean_metric, mean_metric_patients;
    vector<float> total_metric;
    View<DataType::float32> img_t;
    Metric* metric_fn = getMetric("categorical_accuracy");
    ofstream of;
    layer out = getOut(s.net)[0];
    // Map in which the key is the patient name and the value is the pair <label, prediction>
    unordered_map<string, pair<vector<int>, vector<int>>> patients(vsize(d.split_[d.current_split_].samples_indices_));
    auto sum = 0;

    cout << "Starting " << type << ": " << endl;
    // Resize to batch size if we have done a previous resize
    if (d.split_[d.current_split_].last_batch_ != s.batch_size) {
        s.net->resize(s.batch_size);
    }
    d.SetSplit(type);
    d.ResetBatch(d.current_split_); // Reset batch without shuffling

    auto str = type == "validation" ? "/" + to_string(s.epochs - 1) : "";
    d.Start();
    for (int j = 0, n = 0; j < num_batches; ++j) {
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
        ca = metric_fn->value(y.get(), output.get());
        total_metric.push_back(ca);
        cout << "categorical_accuracy: " << ca / current_bs << endl;

        for (int k = 0; k < current_bs; ++k, ++n) {
            // Find the patient name of the current sample and save its label in the map
            string patient_name = samples[k].location_[0].parent_path().filename().string();
            auto gt_class = samples[k].label_.value()[0];
            patients[patient_name].first.push_back(gt_class);

            unique_ptr<Tensor> pred(output->select({ to_string(k) }));
            unique_ptr<Tensor> target(y->select({ to_string(k) }));

            // Find the predicted and the ground truth class
            float max = std::numeric_limits<float>::min();
            int pred_class = -1;
            for (unsigned c = 0; c < pred->size; ++c) {
                if (pred->ptr[c] > max) {
                    max = pred->ptr[c];
                    pred_class = c;
                }
            }

            // Save the class predicted in the map
            patients[patient_name].second.push_back(pred_class);

            if (s.save_images) {
                unique_ptr<Tensor> single_image(x->select({ to_string(k) }));
                single_image->mult_(255.);
                single_image->normalize_(0.f, 255.f);
                TensorToView(single_image.get(), img_t);
                img_t.colortype_ = ColorType::GRAY;
                img_t.channels_ = "xyc";
                auto filename = samples[k].location_[0].stem().concat("_gt_class_" + to_string(gt_class) + ".png");
                ImWrite(current_path / d.classes_[pred_class] / filename, img_t);
            }
        }
    }
    d.Stop();

    // Calculate the most voted label for each patient and assign it to all the slices of the patient
    for (auto& elem : patients) {
        auto mean_patient = accumulate(elem.second.second.begin(), elem.second.second.end(), 0.0f) / elem.second.second.size();
        int patient_sum = 0;
        if (mean_patient >= 0.5) {
            fill(elem.second.second.begin(), elem.second.second.end(), 1);
        }
        else {
            fill(elem.second.second.begin(), elem.second.second.end(), 0);
        }

        for (int i = 0; i < elem.second.first.size(); ++i) {
            if (elem.second.first[i] == elem.second.second[i]) {
                sum += 1;
                patient_sum += 1;
            }
        }
        cout << "Patient " << elem.first << " - accuracy: " << (float)patient_sum / elem.second.first.size() << endl;
    }

    // Calculate the mean of the metrics
    mean_metric = accumulate(total_metric.begin(), total_metric.end(), 0.0f) / ((num_batches - 1) * s.batch_size + d.split_[d.current_split_].last_batch_);
    mean_metric_patients = float(sum) / ((num_batches - 1) * s.batch_size + d.split_[d.current_split_].last_batch_);
    cout << "--------------------------------------------------" << endl;
    cout << "Epoch " << epoch << " - Mean " << type << " categorical accuracy PATIENTS : " << mean_metric_patients << endl;
    cout << "--------------------------------------------------" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << "Epoch " << epoch << " - Mean " << type << " categorical accuracy NO PATIENTS: " << mean_metric << endl;
    cout << "--------------------------------------------------" << endl;

    if (type == "validation") {
        if (mean_metric > best_metric) {
            cout << "Saving weights..." << endl;
            save_net_to_onnx_file(s.net, (s.checkpoint_dir / (s.exp_name + "_epoch_" + to_string(epoch) + ".onnx")).string());
            best_metric = mean_metric;
        }

        if (mean_metric_patients > best_metric_patients) {
            cout << "Saving weights..." << endl;
            save_net_to_onnx_file(s.net, (s.checkpoint_dir / (s.exp_name + "_epoch_" + to_string(epoch) + "_patients.onnx")).string());
            best_metric_patients = mean_metric_patients;
        }
    }

    of.open(s.exp_name + "_stats.txt", ios::out | ios::app);
    of << "Epoch " << epoch << " - Total " << type << " categorical accuracy: " << mean_metric << endl;
    of.close();

    of.open(s.exp_name + "_patients_stats.txt", ios::out | ios::app);
    of << "Epoch " << epoch << " - Total " << type << " categorical accuracy: " << mean_metric_patients << endl;
    of.close();

    delete metric_fn;
}

int main(int argc, char* argv[])
{
    time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    cout << "Start at " << ctime(&now) << endl;

    // Default settings, they can be changed from command line
    // num_classes, size, model, loss, lr, exp_name, dataset_path, epochs, batch_size, workers, queue_ratio, gpu, input_channels
    Settings s(2, { 224,224 }, "onnx::resnet101", "bce", 1e-4f, "right_kidney_classification", "", 100, 20, 4, 5, {}, 1);

    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
    }

    layer out = getOut(s.net)[0];
    if (typeid(*out) != typeid(LActivation)) {
        out = Softmax(out);
        s.net = Model({ s.net->lin[0] }, { out });
    }

    // Build model
    build(s.net,
        adam(s.lr),      // Optimizer
        { s.loss },                 // Loss
        { "categorical_accuracy" }, // Metric
        s.cs,                       // Computing Service
        s.random_weights            // Randomly initialize network weights
    );

    if (s.last_layer) {
        // Initialize last layer if it's been substituted
        initializeLayer(s.net, "last_layer");
    }

    // View model
    summary(s.net);
    plot(s.net, s.exp_name + ".pdf");
    setlogfile(s.net, s.exp_name);

    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size, InterpolationType::cubic),
        AugMirror(.5),
        AugRotate({ -30, 30 }),
        AugAdditivePoissonNoise({ 0, 10 }),
        AugGammaContrast({ .5, 1.5 }),
        AugGaussianBlur({ .0, .8 }),
        AugCoarseDropout({ 0, 0.03 }, { 0, 0.05 }, 0.25),
        AugToFloat32(255)
        );

    auto validation_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size, InterpolationType::cubic),
        AugToFloat32(255)
        );

    DatasetAugmentations dataset_augmentations{ { training_augs, validation_augs, validation_augs } }; // use the same augmentations for validation and test

    // Read the dataset
    cout << "Reading dataset" << endl;
    KidneyDataset d(s.dataset_path, s.batch_size, dataset_augmentations, ColorType::GRAY, ColorType::none, s.workers, s.queue_ratio, { {"training", false}, {"validation", false}, {"test", false} });

    // int num_batches_training = d.GetNumBatches("training");  // or
    // int num_batches_training = d.GetNumBatches(0);           // where 0 is the split index, or
    int num_batches_training = d.GetNumBatches(SplitType::training);
    int num_batches_validation = d.GetNumBatches(SplitType::validation);
    int num_batches_test = d.GetNumBatches(SplitType::test);

    float best_metric = 0.f, best_metric_patients = 0.f;
    cv::TickMeter tm, tm_epoch;

    if (!s.skip_train) {
        cout << "Starting training" << endl;
        for (int e = s.resume; e < s.epochs; ++e) {
            tm_epoch.reset();
            tm_epoch.start();
            d.SetSplit(SplitType::training);

            auto current_path{ s.result_dir / ("Epoch_" + to_string(e)) };
            if (s.save_images) {
                for (const auto& c : d.classes_) {
                    create_directories(current_path / c);
                }
            }

            // Reset errors for train_batch
            reset_loss(s.net);

            // Resize to batch size if we have done a previous resize
            if (d.split_[d.current_split_].last_batch_ != s.batch_size) {
                s.net->resize(s.batch_size);
            }

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

                // Check input images
                //for (int ind = 0; ind < s.batch_size; ++ind) {
                //    unique_ptr<Tensor> tmp(x->select({ to_string(ind), ":", ":", ":" }));
                //    tmp->mult_(255.);
                //    tmp->normalize_(0.f, 255.f);
                //    tmp->save("../images/train_image_" + to_string(j) + "_" + to_string(ind) + ".png");
                //}

                // if it's the last batch and the number of samples doesn't fit the batch size, resize the network
                if (j == num_batches_training - 1 && x->shape[0] != s.batch_size) {
                    s.net->resize(x->shape[0]);
                }

                // Train batch
                // set_mode(s.net, TRMODE) // not necessary because it's already inside the train_batch
                train_batch(s.net, { x.get() }, { y.get() });

                // Print errors
                print_loss(s.net, j);

                tm.stop();
                cout << "Elapsed time: " << tm.getTimeSec() << endl;
            }
            d.Stop();

            set_mode(s.net, TSMODE);
            Inference("validation", d, s, num_batches_validation, e, current_path, best_metric, best_metric_patients);

            tm_epoch.stop();
            cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;
        }
    }

    int epoch = s.skip_train ? s.resume : s.epochs;
    auto current_path{ s.result_dir / ("Test - epoch " + to_string(epoch)) };
    if (s.save_images) {
        for (const auto& c : d.classes_) {
            create_directories(current_path / c);
        }
    }

    set_mode(s.net, TSMODE);
    Inference("test", d, s, num_batches_test, epoch, current_path, best_metric, best_metric_patients);

    now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    cout << "End at " << ctime(&now) << endl;

    return EXIT_SUCCESS;
}