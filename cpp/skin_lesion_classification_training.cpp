#include "models/models.h"
#include "DataGenerator.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

using namespace ecvl;
using namespace eddl;
using namespace std;
using namespace std::filesystem;


int main( int argc, char *argv[] )
{
    // Settings
    int epochs = 500;
    int batch_size = 8;
    int num_classes = 8;
    std::vector<int> size{ 224,224 }; // Size of images

    vector<int> gpus = {1};
    int lsb = 1;
    string mem = "low_mem";

    string net_filename = "";
    
    for( int i=1; i < argc; i++ ) {
        if ( !strcmp( argv[i], "--low-mem" ) ) {
            mem = "low_mem";
        } else if ( !strcmp( argv[i], "--mid-mem" ) ) {
            mem = "mid_mem";
        } else if ( !strcmp( argv[i], "--full-mem" ) ) {
            mem = "full_mem";
        } else if ( !strcmp( argv[i], "--lsb" ) ) {
            lsb = atoi( argv[++i] );
        } else if ( !strcmp( argv[i], "--batch-size" ) ) {
            batch_size = atoi( argv[++i] );
        } else if ( !strcmp( argv[i], "--gpus-2" ) ) {
            gpus={1,1};
        } else if ( !strcmp( argv[i], "--gpus-1" ) ) {
            gpus={1};
        } else if ( !strcmp( argv[i], "--model" ) ) {
            net_filename = argv[++i];
        }
    }
    
    compserv cs = CS_GPU( gpus, lsb, mem );

    std::mt19937 g(std::random_device{}());

    // Define network
    layer in = Input({ 3, size[0],  size[1] });
    //layer out = VGG16(in, num_classes);
    //layer out = VGG16_inception_2(in, num_classes);
    layer out = ResNet_01(in, num_classes);
    model net = Model({ in }, { out });

    // Build model
    build(net,
        // sgd(0.001f, 0.9f), // Optimizer
        adam(0.0001f), // Optimizer
        { "soft_cross_entropy" }, // Losses
        { "categorical_accuracy" }, // Metrics
        cs // Computing Service
    );

    //toGPU(net);

    if ( net_filename.size() > 0 ) load(net, net_filename, "bin");


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
    //DLDataset d("D:/dataset/isic_classification/isic_classification.yml", batch_size, move(dataset_augmentations));
    DLDataset d("/home/jon/bd/workshop/ISIC/isic_classification/isic_classification.yml", batch_size, move(dataset_augmentations));

    // Prepare tensors which store batch
    // tensor x = new Tensor({ batch_size, d.n_channels_, size[0], size[1] });
    // tensor y = new Tensor({ batch_size, static_cast<int>(d.classes_.size()) });
    tensor output, target, result, single_image;

    int num_samples = vsize(d.GetSplit());
    int num_batches = num_samples / batch_size;

    d.SetSplit(SplitType::validation);
    int num_samples_validation = vsize(d.GetSplit());
    int num_batches_validation = num_samples_validation / batch_size;
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


    // Create producer thread with 'DLDataset d' and 'std::queue q'
    d.SetSplit(SplitType::training);
    // Shuffle training list
    shuffle(std::begin(d.GetSplit()), std::end(d.GetSplit()), g);
    d.ResetAllBatches();
    DataGenerator d_generator_t( &d, batch_size, size, { static_cast<int>(d.classes_.size()) }, 3 );

    d.SetSplit(SplitType::validation);
    DataGenerator d_generator_v( &d, batch_size, size, { static_cast<int>(d.classes_.size()) }, 2 );

    cout << "Starting training" << endl;
    for (int i = 0; i < epochs; ++i) {
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

        tm_epoch.reset();
        tm_epoch.start();

        d_generator_t.start();

        // Feed batches to the model
        for( int j=0; d_generator_t.has_next() /* j < num_batches */; ++j ) {

            tm.reset();
            tm.start();
            cout << "Epoch " << i << "/" << epochs << " (batch " << j << "/" << num_batches << ") - ";
            cout << " |fifo| " << d_generator_t.size() << " ";

            tensor x, y;

            // Load a batch
            // d.LoadBatch(x, y);
            if ( d_generator_t.pop_batch( x, y ) ) {

                // Preprocessing
                x->div_(255.0);

                // Prepare data
                vtensor tx{ x };
                vtensor ty{ y };

                // Train batch
                train_batch(net, tx, ty, indices);

                // Print errors
                print_loss(net, j);

                delete x;
                delete y;
            }
            tm.stop();

            cout << "- Elapsed time: " << tm.getTimeSec() << endl;
        }
        d_generator_t.stop();
        tm_epoch.stop();
        cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;

        cout << "Saving weights..." << endl;
        save(net, "isic_classification_checkpoint_epoch_" + to_string(i) + ".bin", "bin");

        // Evaluation
        d.SetSplit(SplitType::validation);

        d_generator_v.start();

        cout << "Evaluating on validation subset:" << endl;
        for( int j=0, n=0; d_generator_v.has_next(); ++j ) {

            // cout << "Validation: Epoch " << i << "/" << epochs << " (batch " << j << "/" << num_batches_validation << ") - ";

            tensor x, y;

            // Load a batch
            // d.LoadBatch(x, y);
            if ( d_generator_v.pop_batch(x, y) ) {

                // Preprocessing
                x->div_(255.0);

                // Evaluate batch
                forward(net, { x });
                output = getOutput(out);

                sum = 0.;
                for (int k = 0; k < batch_size; ++k, ++n) {
                    result = output->select({to_string(k)});
                    target = y->select({to_string(k)});

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

                        single_image = x->select({to_string(k)});
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
                // cout << " categorical_accuracy: " << static_cast<float>(sum) / batch_size << endl;

                delete x;
                delete y;
            }
        }
        d_generator_v.stop();

        total_avg = accumulate(total_metric.begin(), total_metric.end(), 0.0f) / total_metric.size();
        cout << "Validation categorical accuracy: " << total_avg << endl;

        of.open("output_evaluate_classification.txt", ios::out | ios::app);
        of << "Epoch " << i << " - Total categorical accuracy: " << total_avg << endl;
        of.close();
    }

    // delete x;
    // delete y;
    delete output;

    return EXIT_SUCCESS;
}
