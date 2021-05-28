#include "utils.h"
#include "cxxopts.hpp"
#include "eddl/serialization/onnx/eddl_onnx.h"
#include "../models/models.h"

using namespace std;
using namespace eddl;
using namespace ecvl::filesystem;

bool TrainingOptions(int argc, char* argv[], Settings& s)
{
    std::stringstream result;
    std::copy(s.size.begin(), s.size.end(), std::ostream_iterator<int>(result, ","));
    auto size = result.str().substr(0, result.str().size() - 1);
    std::copy(s.gpu.begin(), s.gpu.end(), std::ostream_iterator<int>(result, ","));
    auto gpus = result.str().substr(0, result.str().size() - 1);

    cxxopts::Options options("DeepHealth pipeline", "");
    options.add_options()
        ("exp_name", "Experiment name", cxxopts::value<string>())
        ("d,dataset_path", "Dataset path", cxxopts::value<path>())
        ("e,epochs", "Number of training epochs", cxxopts::value<int>()->default_value(to_string(s.epochs)))
        ("b,batch_size", "Number of images for each batch", cxxopts::value<int>()->default_value(to_string(s.batch_size)))
        ("n,num_classes", "Number of output classes", cxxopts::value<int>()->default_value(to_string(s.num_classes)))
        ("save_images", "Save validation images or not", cxxopts::value<bool>()->default_value(to_string(s.save_images)))
        ("s,size", "Size to which resize the input images", cxxopts::value<vector<int>>()->default_value(size))
        ("loss", "Loss function", cxxopts::value<string>()->default_value(s.loss))
        ("l,learning_rate", "Learning rate", cxxopts::value<float>()->default_value(to_string(s.lr)))
        ("momentum", "Momentum", cxxopts::value<float>()->default_value(to_string(s.momentum)))
        ("model", "Model of the network", cxxopts::value<string>()->default_value(s.model))
        ("g,gpu", "Which GPUs to use", cxxopts::value<vector<int>>())
        ("lsb", "How many batches are processed before synchronizing the model weights",
            cxxopts::value<int>()->default_value(to_string(s.lsb)))
        ("m,mem", "GPU memory usage configuration", cxxopts::value<string>()->default_value(s.mem))
        ("r,result_dir", "Directory where the output images are stored",
            cxxopts::value<path>()->default_value(s.result_dir.generic_string()))
        ("checkpoint_dir", "Directory where the checkpoints are stored",
            cxxopts::value<path>()->default_value(s.checkpoint_dir.generic_string()))
        ("c,checkpoint", "Path to the onnx checkpoint file", cxxopts::value<string>())
        ("i,input_channels", "Number of input channels", cxxopts::value<int>()->default_value(to_string(s.in_channels)))
        ("w,workers", "Number of producer workers", cxxopts::value<int>()->default_value(to_string(s.workers)))
        ("q,queue_ratio", "Queue ratio size (queue size will be: batch_size*workers*queue_ratio)", cxxopts::value<int>()->default_value(to_string(s.queue_ratio)))
        ("resume", "Resume training from this epoch", cxxopts::value<int>()->default_value(to_string(s.resume)))
        ("t,skip_train", "Skip training and perform only test", cxxopts::value<bool>()->default_value(to_string(s.skip_train)))
        ("h,help", "Print usage");

    auto args = options.parse(argc, argv);

    if (args.count("help")) {
        cout << options.help() << endl;
        return false;
    }

    // Settings
    s.epochs = args["epochs"].as<int>();
    s.batch_size = args["batch_size"].as<int>();
    s.num_classes = args["num_classes"].as<int>();
    s.save_images = args["save_images"].as<bool>();
    s.size = args["size"].as<vector<int>>();
    s.model = args["model"].as<string>();
    s.loss = args["loss"].as<string>();
    s.lr = args["learning_rate"].as<float>();
    s.momentum = args["momentum"].as<float>();
    s.random_weights = true;
    s.resume = args["resume"].as<int>();
    s.skip_train = args["skip_train"].as<bool>();

    if (args.count("dataset_path")) {
        s.dataset_path = args["dataset_path"].as<path>();
    }
    else if (s.dataset_path.empty()) {
        cout << ECVL_ERROR_MSG "'d,dataset_path' is a required argument." << endl;
        return false;
    }

    cout << "Options used: \n";
    cout << "epochs: " << s.epochs << "\n";
    cout << "batch_size: " << s.batch_size << "\n";
    cout << "loss: " << s.loss << "\n";
    cout << "learning rate: " << s.lr << "\n";
    cout << "num_classes: " << s.num_classes << "\n";
    cout << "size: (" << s.size[0] << ", " << s.size[1] << ")\n";

    s.checkpoint_dir = args["checkpoint_dir"].as<path>();
    create_directory(s.checkpoint_dir);
    cout << "dataset_path: " << s.dataset_path << "\n";
    cout << "checkpoint_dir: " << s.checkpoint_dir << "\n";

    vector<int> in_shape{ s.in_channels, s.size[0], s.size[1] };
    if (args.count("checkpoint")) {
        s.checkpoint_path = args["checkpoint"].as<string>();
    }
    if (!s.checkpoint_path.empty()) {
        s.random_weights = false;
        s.net = import_net_from_onnx_file(s.checkpoint_path, in_shape, DEV_CPU);
        cout << "pretrained network: " << s.checkpoint_path << "\n";
    }
    else { // Define the network
        layer in, out;
        in = Input(in_shape);

        if (!s.model.compare("SegNet")) {
            out = SegNet(in, s.num_classes);
        }
        else if (!s.model.compare("SegNetBN")) {
            out = SegNetBN(in, s.num_classes);
        }
        else if (!s.model.compare("UNetWithPadding")) {
            out = UNetWithPadding(in, s.num_classes);
        }
        else if (!s.model.compare("UNetWithPaddingBN")) {
            out = UNetWithPaddingBN(in, s.num_classes);
        }
        else if (!s.model.compare("LeNet")) {
            out = LeNet(in, s.num_classes);
        }
        else if (!s.model.compare("VGG16")) {
            out = VGG16(in, s.num_classes);
        }
        else if (!s.model.compare("VGG16_inception_1")) {
            out = VGG16_inception_1(in, s.num_classes);
        }
        else if (!s.model.compare("VGG16_inception_2")) {
            out = VGG16_inception_2(in, s.num_classes);
        }
        else if (!s.model.compare("ResNet_01")) {
            out = ResNet_01(in, s.num_classes);
        }
        else if (!s.model.compare("DeepLabV3Plus")) {
            out = DeepLabV3Plus(s.num_classes).forward(in);
        }
        else {
            cout << ECVL_ERROR_MSG
                << "You must specify one of these models: SegNet, SegNetBN, UNetWithPadding, UNetWithPaddingBN, DeepLabV3Plus for segmentation;"
                "LeNet, VGG16, VGG16_inception_1, VGG16_inception_2, ResNet_01 for classification" << endl;
            return EXIT_FAILURE;
        }

        s.net = Model({ in }, { out });
        cout << "model: " << s.model << "\n";
    }

    s.lsb = args["lsb"].as<int>();
    s.mem = args["mem"].as<string>();
    if (args.count("gpu")) {
        s.gpu = args["gpu"].as<vector<int>>();
    }
    if (!s.gpu.empty()) {
        s.cs = CS_GPU(s.gpu, s.lsb, s.mem);
        cout << "Model running on GPU: {";
        for (auto& x : s.gpu) {
            cout << x << ",";
        }
        cout << "}\n";
    }
    else {
        s.cs = CS_CPU(s.lsb, s.mem);
        cout << "Model running on CPU\n";
    }
    cout << "lsb: " << s.lsb << "\n";
    cout << "mem: " << s.mem << "\n";

    if (s.save_images) {
        s.result_dir = args["result_dir"].as<path>();
        create_directory(s.result_dir);
        cout << "save_images: true\n";
        cout << "result output folder: " << s.result_dir << "\n";
    }
    else {
        cout << "save_images: false\n" << endl;
    }

    if (args.count("exp_name")) {
        s.exp_name = args["exp_name"].as<string>();
    }

    if (s.skip_train) {
        cout << "Skipping train..." << endl;
    }

    return true;
}