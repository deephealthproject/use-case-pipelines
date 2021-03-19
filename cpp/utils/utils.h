#ifndef UTILS_H
#define UTILS_H

#include <string>
#include "ecvl/core/filesystem.h"
#include "ecvl/support_eddl.h"

struct Settings
{
    int epochs;
    int batch_size;
    int num_classes;
    bool save_images;
    std::vector<int> size;
    std::string model;
    std::string loss;
    float lr;
    float momentum;
    std::vector<int> gpu;
    int lsb;
    std::string mem;
    bool random_weights;
    eddl::compserv cs;
    Net* net;
    ecvl::filesystem::path result_dir, checkpoint_dir, dataset_path;
    std::string exp_name;

    Settings() = delete;
    Settings(int num_classes_,
        const std::vector<int>& size_,
        const std::string& model_,
        const std::string& loss_,
        const float& lr_,
        const string& exp_name_ = "") :
        num_classes(num_classes_), size(size_), model(model_), loss(loss_), lr(lr_), exp_name(exp_name_)
    {}
};

bool TrainingOptions(int argc, char* argv[], Settings& s);

#endif //UTILS_H
