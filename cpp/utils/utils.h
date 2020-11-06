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
    int n_channels;
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
    Net *net;
    ecvl::filesystem::path result_dir, checkpoint_dir, dataset_path;
};

bool TrainingOptions(int argc, char *argv[], Settings& s);

#endif //UTILS_H
