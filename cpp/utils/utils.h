#ifndef UTILS_H
#define UTILS_H

#include <string>
#include "ecvl/core/filesystem.h"
#include "ecvl/support_eddl.h"

void Download(const string& url);

struct Settings
{
    int epochs, batch_size, num_classes, in_channels, workers, resume;
    bool save_images;
    std::vector<int> size;
    std::string model;
    std::string loss;
    float lr, momentum;
    std::vector<int> gpu;
    int lsb;
    std::string mem;
    bool random_weights;
    bool skip_train;
    eddl::compserv cs;
    Net* net = nullptr;
    ecvl::filesystem::path result_dir, checkpoint_dir, dataset_path;
    std::string exp_name, checkpoint_path;
    bool last_layer = false;
    double queue_ratio;

    Settings() = delete;
    ~Settings() { if(net) delete net; };
    Settings(int num_classes_,
        const std::vector<int>& size_,
        const std::string& model_,
        const std::string& loss_,
        const float& lr_,
        const string& exp_name_ = "",
        const ecvl::filesystem::path& dataset_path_ = "",
        const int& epochs_ = 100,
        const int& batch_size_ = 12,
        const int& workers_ = 1,
        const double& queue_ratio_ = 1.,
        const vector<int>& gpus_ = {},
        const int& in_channels_ = 3,
        const string& checkpoint_path_ = "",
        const int& resume_ = 0,
        const bool& skip_train_ = false,
        const int& lsb_ = 1,
        const bool& save_images_ = false,
        const string& mem_ = "low_mem",
        const float& momentum_ = 0.9,
        const ecvl::filesystem::path& result_dir_ = "../output_images",
        const ecvl::filesystem::path& checkpoint_dir_ = "../checkpoints"
        ) :
        num_classes(num_classes_), size(size_), model(model_), loss(loss_), lr(lr_), exp_name(exp_name_), dataset_path(dataset_path_),
        epochs(epochs_), batch_size(batch_size_), workers(workers_), queue_ratio(queue_ratio_), lsb(lsb_), gpu(gpus_), 
        in_channels(in_channels_), save_images(save_images_), mem(mem_), momentum(momentum_), result_dir(result_dir_), 
        checkpoint_dir(checkpoint_dir_), checkpoint_path(checkpoint_path_), skip_train(skip_train_), resume(resume_)
    {}
};

bool TrainingOptions(int argc, char* argv[], Settings& s);

#endif //UTILS_H
