#include "data_generator/data_generator.h"
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

int main(int argc, char* argv[])
{
    // Settings
    int epochs = 50;
    int batch_size = 12;
    int num_classes = 8;
    std::vector<int> size{ 224,224 }; // Size of images

    vector<int> gpus = { 1 };
    int lsb = 1;
    string mem = "low_mem";
    string checkpoint = "";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--low-mem")) {
            mem = "low_mem";
        }
        else if (!strcmp(argv[i], "--mid-mem")) {
            mem = "mid_mem";
        }
        else if (!strcmp(argv[i], "--full-mem")) {
            mem = "full_mem";
        }
        else if (!strcmp(argv[i], "--lsb")) {
            lsb = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--batch-size")) {
            batch_size = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--gpus-2")) {
            gpus = { 1,1 };
        }
        else if (!strcmp(argv[i], "--gpus-1")) {
            gpus = { 1 };
        }
        else if (!strcmp(argv[i], "--checkpoint")) {
            checkpoint = argv[++i];
        }
    }

    std::mt19937 g(std::random_device{}());

    // Define network
    layer in = Input({ 3, size[0],  size[1] });
    layer out = VGG16(in, num_classes);
    model net = Model({ in }, { out });

    // Build model
    build(net,
        sgd(0.001f, 0.9f), // Optimizer
        { "soft_cross_entropy" }, // Losses
        { "categorical_accuracy" }, // Metrics
        CS_GPU(gpus, lsb, mem) // Computing Service
    );

    if (!checkpoint.empty()) {
        load(net, checkpoint, "bin");
    }

    // View model
    summary(net);
    plot(net, "model.pdf");
    setlogfile(net, "skin_lesion_classification");

    auto training_augs = make_unique<SequentialAugmentationContainer>(
        AugResizeDim(size),
        AugMirror(.5),
        AugFlip(.5),
        AugRotate({ -180, 180 }),
        AugAdditivePoissonNoise({ 0, 10 }),
        AugGammaContrast({ .5,1.5 }),
        AugGaussianBlur({ .0,.8 }),
        AugCoarseDropout({ 0, 0.3 }, { 0.02, 0.05 }, 0.5));

    auto validation_augs = make_unique<SequentialAugmentationContainer>(AugResizeDim(size));

    DatasetAugmentations dataset_augmentations{ {move(training_augs), move(validation_augs), nullptr } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d("D:/dataset/isic_classification/isic_classification.yml", batch_size, move(dataset_augmentations));
    // Create producer thread with 'DLDataset d' and 'std::queue q'
    int num_samples = vsize(d.GetSplit());
    int num_batches = num_samples / batch_size;
    DataGenerator d_generator_t(&d, batch_size, size, { vsize(d.classes_) }, 3);

    d.SetSplit(SplitType::validation);
    int num_samples_validation = vsize(d.GetSplit());
    int num_batches_validation = num_samples_validation / batch_size;
    DataGenerator d_generator_v(&d, batch_size, size, { vsize(d.classes_) }, 2);

    tensor output, target, result, single_image;
    float sum = 0., ca = 0.;

    vector<float> total_metric;
    Metric* m = getMetric("categorical_accuracy");

    bool save_images = true;
    path output_path;
    if (save_images) {
        output_path = "../output_images";
        create_directory(output_path);
    }
    View<DataType::float32> img_t;

    float total_avg;
    ofstream of;

    vector<int> indices(batch_size);
    iota(indices.begin(), indices.end(), 0);
    cv::TickMeter tm;
    cv::TickMeter tm_epoch;

    cout << "Starting training" << endl;
    for (int i = 0; i < epochs; ++i) {

        tm_epoch.reset();
        tm_epoch.start();

        auto current_path{ output_path / path("Epoch_" + to_string(i)) };
        if (save_images) {
            for (int c = 0; c < d.classes_.size(); ++c) {
                create_directories(current_path / path(d.classes_[c]));
            }
        }

        d.SetSplit(SplitType::training);
        // Reset errors
        reset_loss(net);
        total_metric.clear();

        // Shuffle training list
        shuffle(std::begin(d.GetSplit()), std::end(d.GetSplit()), g);
        d.ResetAllBatches();

        d_generator_t.Start();

        // Feed batches to the model
        for (int j = 0; d_generator_t.HasNext() /* j < num_batches */; ++j) {
            tm.reset();
            tm.start();
            cout << "Epoch " << i << "/" << epochs << " (batch " << j << "/" << num_batches << ") - ";
            cout << "|fifo| " << d_generator_t.Size() << " - ";

            tensor x, y;

            // Load a batch
            if (d_generator_t.PopBatch(x, y)) {
                // Preprocessing
                x->div_(255.0);

                // Train batch
                train_batch(net, { x }, { y }, indices);

                // Print errors
                print_loss(net, j);

                delete x;
                delete y;
            }
            tm.stop();

            cout << "Elapsed time: " << tm.getTimeSec() << endl;
        }

        d_generator_t.Stop();
        tm_epoch.stop();
        cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;

        cout << "Saving weights..." << endl;
        save(net, "isic_classification_checkpoint_epoch_" + to_string(i) + ".bin", "bin");

        // Evaluation
        d.SetSplit(SplitType::validation);

        d_generator_v.Start();

        cout << "Evaluate:" << endl;
        for (int j = 0, n = 0; d_generator_v.HasNext(); ++j) {
            cout << "Validation: Epoch " << i << "/" << epochs << " (batch " << j << "/" << num_batches_validation << ") - ";

            tensor x, y;

            // Load a batch
            if (d_generator_v.PopBatch(x, y)) {
                // Preprocessing
                x->div_(255.0);

                // Evaluate batch
                forward(net, { x });
                output = getOutput(out);

                sum = 0.;
                for (int k = 0; k < batch_size; ++k, ++n) {
                    result = output->select({ to_string(k) });
                    target = y->select({ to_string(k) });

                    ca = m->value(target, result);

                    total_metric.push_back(ca);
                    sum += ca;

                    if (save_images) {
                        float max = std::numeric_limits<float>::min();
                        int classe = -1;
                        int gt_class = -1;
                        for (int i = 0; i < result->size; ++i) {
                            if (result->ptr[i] > max) {
                                max = result->ptr[i];
                                classe = i;
                            }

                            if (target->ptr[i] == 1.) {
                                gt_class = i;
                            }
                        }

                        single_image = x->select({ to_string(k) });
                        TensorToView(single_image, img_t);
                        img_t.colortype_ = ColorType::BGR;
                        single_image->mult_(255.);

                        path filename = d.samples_[d.GetSplit()[n]].location_[0].filename();

                        path cur_path = current_path / d.classes_[classe] / filename.replace_extension("_gt_class_" + to_string(gt_class) + ".png");
                        ImWrite(cur_path, img_t);
                    }

                    delete result;
                    delete target;
                    delete single_image;
                }
                cout << " categorical_accuracy: " << static_cast<float>(sum) / batch_size << endl;

                delete x;
                delete y;
            }
        }

        d_generator_v.Stop();

        total_avg = accumulate(total_metric.begin(), total_metric.end(), 0.0f) / total_metric.size();
        cout << "Validation categorical accuracy: " << total_avg << endl;

        of.open("output_evaluate_classification.txt", ios::out | ios::app);
        of << "Epoch " << i << " - Total categorical accuracy: " << total_avg << endl;
        of.close();
    }

    delete output;

    return EXIT_SUCCESS;
}