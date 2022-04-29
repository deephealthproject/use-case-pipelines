#include "metrics/metrics.h"
#include "utils/utils.h"

#include <iostream>

#include "ecvl/core/filesystem.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace ecvl;
using namespace ecvl::filesystem;
using namespace eddl;
using namespace std;

void Ensemble(const Settings& s, const string& type, vector<map<path, Image>>& predictions, const path& gt_path, const int& max_size, const vector<path>& name_vector)
{
    int n_models = vsize(predictions);
    vector<Image> images(n_models);
    vector<ContiguousIterator<float>> iterators(n_models);
    vector<double> metric_list, metric_list_postprocessing;
    Image dst, gt, labels;

    path current_path = "Ensemble " + type;
    if (s.save_images) {
        create_directory(s.result_dir / current_path);
    }

    for (int j = 0; j < predictions[0].size(); ++j) {
        auto image = name_vector[j].stem();
        auto image_name = name_vector[j];

        ImRead(gt_path / path(image.string() + "_segmentation.png"), gt, ImReadMode::GRAYSCALE);

        for (int i = 0; i < n_models; ++i) {
            images[i] = predictions[i][image_name];
            ResizeDim(images[i], images[i], { max_size, max_size }, InterpolationType::cubic);
            iterators[i] = images[i].ContiguousBegin<float>();
        }

        ResizeDim(gt, gt, { max_size, max_size }, InterpolationType::nearest);
        dst.Create(gt.dims_, DataType::uint8, "xyc", gt.colortype_);

        auto it_dst = dst.ContiguousBegin<uint8_t>();
        auto it_gt = gt.ContiguousBegin<uint8_t>();
        auto e_dst = dst.ContiguousEnd<uint8_t>();
        double intersection = 0, unions = 0, iou = 0;

        // For each image pixel, calculate its value as the mean of that pixel for all the ensemble models predicted images
        for (; it_dst != e_dst; ++it_dst, ++it_gt) {
            float value = 0;
            for (int i = 0; i < n_models; ++i) {
                value += *iterators[i];
                ++iterators[i];
            }
            value /= n_models;
            *it_dst = value > 0.3 ? 255 : 0;
            intersection += ((*it_gt == 255) && (*it_dst == 255));
            unions += ((*it_gt == 255) || (*it_dst == 255));
        }

        iou = (intersection + 1e-06) / (unions + 1e-06);
        metric_list.push_back(iou);
        cout << "Ensemble IoU " << image << ": " << iou << " - intersection: " << intersection << " - union: " << unions << endl;

        // Find the CC and if they are more than 2 (lesion and background), remove the pixels of the smaller CC
        int n_labels = ConnectedComponentsLabeling(dst, labels);
        ConvertTo(labels, dst, DataType::uint8);

        if (n_labels != 2) {
            vector<int> label_qty(n_labels);
            for (it_dst = dst.ContiguousBegin<uint8_t>(), e_dst = dst.ContiguousEnd<uint8_t>(); it_dst != e_dst; ++it_dst) {
                ++label_qty[*it_dst];
            }

            auto max = distance(label_qty.begin(), max_element(label_qty.begin() + 1, label_qty.end()));

            for (it_dst = dst.ContiguousBegin<uint8_t>(); it_dst != e_dst; ++it_dst) {
                if (*it_dst != max) {
                    *it_dst = 0;
                }
                else {
                    *it_dst = 255;
                }
            }
        }
        else {
            Mul(dst, 255, dst);
        }

        // Calculate again IoU after the post processing
        intersection = 0;
        unions = 0;
        it_dst = dst.ContiguousBegin<uint8_t>(), e_dst = dst.ContiguousEnd<uint8_t>();

        for (it_gt = gt.ContiguousBegin<uint8_t>(); it_dst != e_dst; ++it_dst, ++it_gt) {
            intersection += ((*it_gt == 255) && (*it_dst == 255));
            unions += ((*it_gt == 255) || (*it_dst == 255));
        }

        iou = (intersection + 1e-06) / (unions + 1e-06);
        cout << "Post processing IoU " << image << ": " << iou << " - intersection: " << intersection << " - union: " << unions << endl;
        metric_list_postprocessing.push_back(iou);

        if (s.save_images) {
            ImWrite(s.result_dir / current_path / image_name, dst);
            ImWrite(s.result_dir / current_path / image_name.replace_extension(".png"), gt);
        }
    }
    auto mean_metric = std::accumulate(metric_list.begin(), metric_list.end(), 0.0) / vsize(metric_list);
    auto mean_metric_pp = std::accumulate(metric_list_postprocessing.begin(), metric_list_postprocessing.end(), 0.0) / vsize(metric_list_postprocessing);
    cout << "Ensemble Mean IoU: " << mean_metric << endl;
    cout << "PostProcessing Mean IoU: " << mean_metric_pp << endl;
}

void Inference(const string& type, DLDataset& d, const Settings& s, const int num_batches, const int epoch, const path& current_path, double& best_metric, vector<map<path, Image>>& predictions, vector<path>& name_vector, const int model_index = 0)
{
    double mean_metric;
    vector<vector<Point2i>> contours;
    Image labels, tmp;
    View<DataType::float32> pred_t, target_t, img_t;
    Eval evaluator;
    ofstream of;
    layer out = getOut(s.net)[0];
    name_vector.clear();

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
    for (int j = 0, n = 0; j < num_batches; ++j) {
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
        forward(s.net, { x.get() }); // forward does not require reset_loss
        unique_ptr<Tensor> output(getOutput(out));

        // Compute IoU metric and optionally save the output images
        for (int k = 0; k < current_bs; ++k, ++n) {
            auto name = samples[k].location_[0].filename();
            unique_ptr<Tensor> pred(output->select({ to_string(k) }));
            TensorToView(pred.get(), pred_t);
            unique_ptr<Tensor> target(y->select({ to_string(k) }));
            TensorToView(target.get(), target_t);

            if (s.ensemble) {
                name_vector.push_back(name);

                pred_t.colortype_ = ColorType::GRAY;
                pred_t.channels_ = "xyc";

                predictions[model_index][name] = pred_t;
            }

            cout << " - IoU " << name << ": " << evaluator.BinaryIoU(pred_t, target_t, 0.3f);

            if (s.save_images) {
                unique_ptr<Tensor> single_image(x->select({ to_string(k) }));
                single_image->mult_(255.);
                single_image->normalize_(0.f, 255.f);
                TensorToView(single_image.get(), img_t);
                img_t.colortype_ = ColorType::RGB;
                img_t.channels_ = "xyc";

                pred->mult_(255.);
                pred_t.colortype_ = ColorType::GRAY;
                pred_t.channels_ = "xyc";
                ConvertTo(pred_t, tmp, DataType::uint8);
                ConnectedComponentsLabeling(tmp, labels);
                ConvertTo(labels, tmp, DataType::uint8);
                FindContours(tmp, contours);
                ConvertTo(img_t, tmp, DataType::uint8);

                for (auto& contour : contours) {
                    for (auto c : contour) {
                        *tmp.Ptr({ c[0], c[1], 0 }) = 255;
                        *tmp.Ptr({ c[0], c[1], 1 }) = 0;
                        *tmp.Ptr({ c[0], c[1], 2 }) = 0;
                    }
                }

                ImWrite(current_path / name, tmp);

                if (epoch == 0 || type == "test") {
                    target->mult_(255.);
                    target_t.colortype_ = ColorType::GRAY;
                    target_t.channels_ = "xyc";
                    ImWrite(s.result_dir / (type + " Ground Truth") / samples[k].label_path_.value().filename(), target_t);
                }
            }
        }
        cout << endl;
    }
    d.Stop();

    mean_metric = evaluator.MeanMetric();
    cout << "----------------------------------------" << endl;
    cout << "Epoch " << epoch << " - Mean " << type << " IoU: " << mean_metric << endl;
    cout << "----------------------------------------" << endl;

    if (!s.ensemble) {
        if (type == "validation") {
            if (mean_metric > best_metric) {
                cout << "Saving weights..." << endl;
                save_net_to_onnx_file(s.net, (s.checkpoint_dir / (s.exp_name + "_epoch_" + to_string(epoch) + ".onnx")).string());
                best_metric = mean_metric;
            }
        }

        of.open(s.exp_name + "_stats.txt", ios::out | ios::app);
        of << "Epoch " << epoch << " - Total " << type << " IoU: " << mean_metric << endl;
        of.close();
    }
}

int main(int argc, char* argv[])
{
    time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    cout << "Start at " << ctime(&now) << endl;

    // Default settings, they can be changed from command line
    // num_classes, size, model, loss, lr, exp_name, dataset_path, epochs, batch_size, workers, queue_ratio
    Settings s(1, { 512,512 }, "DeepLabV3Plus", "binary_cross_entropy", 7e-5f, "skin_lesion_segmentation", "", 200, 4, 6, 10);
    if (!TrainingOptions(argc, argv, s)) {
        return EXIT_FAILURE;
    }

    if (!s.ensemble) {
        layer out = getOut(s.net)[0];
        if (typeid(*out) != typeid(LActivation)) {
            out = Sigmoid(out);
            s.net = Model(s.net->lin, { out });
        }

        // Build model
        build(s.net,
            adam(s.lr),      // Optimizer
            { s.loss },      // Loss
            { "dice" },      // Metric
            s.cs,            // Computing Service
            s.random_weights // Randomly initialize network weights
        );

        // 343 is the first layer of deeplab without resnet
        if (s.model == "DeepLabV3Plus" && s.checkpoint_path.empty()) {
            for (int i = 343; i != s.net->layers.size(); i++)
                initializeLayer(s.net, s.net->layers[i]->name);
        }

        // View model
        summary(s.net);
        plot(s.net, s.exp_name + ".pdf");
        setlogfile(s.net, s.exp_name);
    }

    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size, InterpolationType::cubic),
        AugMirror(.5),
        AugFlip(.5),
        AugRotate({ -180, 180 }, {}, 1.0, InterpolationType::cubic),
        OneOfAugmentationContainer(0.3, AugElasticTransform({ 34, 40 }, { 2, 4 }, InterpolationType::nearest, BorderType::BORDER_CONSTANT)),
        OneOfAugmentationContainer(0.5, AugBrightness({ 0.1, 0.2 })),
        OneOfAugmentationContainer(0.5, AugResizeScale({ 1, 1.5 }, InterpolationType::cubic)),
        AugCenterCrop(s.size),
        AugAdditivePoissonNoise({ 0, 10 }),
        AugGammaContrast({ .5, 1.5 }),
        AugGaussianBlur({ .0, .8 }),
        AugCoarseDropout({ 0, 0.03 }, { 0.02, 0.05 }, 0.25),
        AugToFloat32(255, 255),
        AugNormalize({ 0.67501814, 0.5663187, 0.52339128 }, { 0.11092593, 0.10669603, 0.119005 }) // isic stats
        // AugNormalize({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }) // imagenet stats
        );

    auto validation_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim(s.size, InterpolationType::cubic),
        AugToFloat32(255, 255),
        AugNormalize({ 0.67501814, 0.5663187, 0.52339128 }, { 0.11092593, 0.10669603, 0.119005 }) // isic stats
        // AugNormalize({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }) // imagenet stats
        );

    // Replace the random seed with a fixed one to have reproducible experiments
    // AugmentationParam::SetSeed(50);

    DatasetAugmentations dataset_augmentations{ { training_augs, validation_augs, validation_augs } }; // use the same augmentations for validation and test

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d(s.dataset_path, s.batch_size, dataset_augmentations, ColorType::RGB, ColorType::GRAY, s.workers, s.queue_ratio, { {"training", true}, {"validation", false}, {"test", false} });

    // int num_batches_training = d.GetNumBatches("training");  // or
    // int num_batches_training = d.GetNumBatches(0);           // where 0 is the split index, or
    int num_batches_training = d.GetNumBatches(SplitType::training);
    int num_batches_validation = d.GetNumBatches(SplitType::validation);
    int num_batches_test = d.GetNumBatches(SplitType::test);

    double best_metric = 0.;
    float poly_power = 0.9f;
    int iteration = 0, max_iter = num_batches_training * s.epochs;
    cv::TickMeter tm, tm_epoch;
    vector<map<path, Image>> predictions_validation, predictions_test;
    vector<path> name_vector_validation, name_vector_test;

    if (!s.skip_train && !s.ensemble) {
        if (s.save_images) {
            create_directory(s.result_dir / "validation Ground Truth");
        }
        cout << "Starting training" << endl;
        for (int e = s.resume; e < s.epochs; ++e) {
            tm_epoch.reset();
            tm_epoch.start();
            d.SetSplit(SplitType::training);
            auto current_path{ s.result_dir / ("Epoch_" + to_string(e)) };
            if (s.save_images) {
                create_directory(current_path);
            }

            auto new_lr = s.lr * pow((1 - float(iteration) / max_iter), poly_power);
            setlr(s.net, { new_lr });

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
                //    {
                //        unique_ptr<Tensor> tmp(x->select({ to_string(ind), ":", ":", ":" }));
                //        //tmp->normalize_(0.f, 1.f);
                //        tmp->clamp_(0.f, 1.f);
                //        tmp->mult_(255.f);
                //        tmp->save("../images/train_image_" + to_string(j) + "_" + to_string(ind) + ".png");
                //    }
                //    {
                //        unique_ptr<Tensor> tmp(y->select({ to_string(ind), ":", ":", ":" }));
                //        //tmp->normalize_(0.f, 1.f);
                //        tmp->clamp_(0.f, 1.f);
                //        tmp->mult_(255.f);
                //        tmp->save("../images/train_gt_" + to_string(j) + "_" + to_string(ind) + ".png");
                //    }
                //}

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
                cout << "- Elapsed time: " << tm.getTimeSec() << endl;
                ++iteration;
            }
            d.Stop();

            set_mode(s.net, TSMODE);
            Inference("validation", d, s, num_batches_validation, e, current_path, best_metric, predictions_validation, name_vector_validation);

            tm_epoch.stop();
            cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;
        }
    }

    int epoch = s.skip_train ? s.resume : s.epochs;
    if (!s.ensemble) {
        auto current_path{ s.result_dir / ("Test - epoch " + to_string(epoch)) };
        if (s.save_images) {
            create_directory(current_path);
            create_directory(s.result_dir / "test Ground Truth");
        }
        set_mode(s.net, TSMODE);
        Inference("test", d, s, num_batches_test, epoch, current_path, best_metric, predictions_test, name_vector_test);
    }
    else {
        path gt_path = d.samples_[0].label_path_.value().parent_path();
        int model_index = 0, max_size = 0;

        for (auto const& entry : directory_iterator{ s.checkpoint_dir }) {
            auto current_path{ s.result_dir / ("Model " + to_string(model_index)) };
            if (s.save_images) {
                create_directories(current_path / "validation");
                create_directories(current_path / "test");
            }
            s.net = import_net_from_onnx_file(entry.path().string());
            vector<int> size = { s.net->lin[0]->getShape()[2], s.net->lin[0]->getShape()[3] };
            if (size[0] > max_size) {
                max_size = size[0];
            }

            if (entry.path().string().find("deeplabv3plus") != std::string::npos) {
                auto val_augs = make_shared<SequentialAugmentationContainer>(
                    AugResizeDim(size, InterpolationType::cubic),
                    AugToFloat32(255, 255),
                    AugNormalize({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }) // imagenet stats
                    );
                DatasetAugmentations da{ { nullptr, val_augs, val_augs } };
                d.SetAugmentations(da);
            }
            else {
                auto val_augs = make_shared<SequentialAugmentationContainer>(
                    AugResizeDim(size, InterpolationType::cubic),
                    AugToFloat32(255, 255),
                    AugNormalize({ 0.67501814, 0.5663187, 0.52339128 }, { 0.11092593, 0.10669603, 0.119005 }) // isic stats
                    );
                DatasetAugmentations da{ { nullptr, val_augs , val_augs } };
                d.SetAugmentations(da);
            }

            d.resize_dims_ = size;
            d.tensors_shape_.first = { d.batch_size_, d.n_channels_, d.resize_dims_[0], d.resize_dims_[1] };
            d.tensors_shape_.second = { d.batch_size_, d.n_channels_gt_, d.resize_dims_[0], d.resize_dims_[1] };

            cout << "MODEL: " << entry.path().string() << endl;
            cout << "INPUT SHAPE: " << size[0] << " - " << size[1] << endl;
            build(s.net,
                adam(s.lr),      // Optimizer
                { s.loss },      // Loss
                { "dice" },      // Metric
                CS_GPU(s.gpu, "low_mem"),            // Computing Service
                false            // Randomly initialize network weights
            );
            summary(s.net);
            predictions_validation.push_back(map<path, Image>());
            predictions_test.push_back(map<path, Image>());
            set_mode(s.net, TSMODE);
            Inference("validation", d, s, num_batches_validation, epoch, current_path / "validation", best_metric, predictions_validation, name_vector_validation, model_index);
            Inference("test", d, s, num_batches_test, epoch, current_path / "test", best_metric, predictions_test, name_vector_test, model_index);
            ++model_index;

            delete s.net; s.net = nullptr;
        }
        cout << "Starting validation ensemble" << endl;
        Ensemble(s, "validation", predictions_validation, gt_path, max_size, name_vector_validation);

        cout << "Starting test ensemble" << endl;
        Ensemble(s, "test", predictions_test, gt_path, max_size, name_vector_test);
    }

    now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    cout << "End at " << ctime(&now) << endl;

    return EXIT_SUCCESS;
}