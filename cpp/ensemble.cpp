#include "metrics/metrics.h"
#include "models/models.h"
#include "cxxopts.hpp"

#include "eddl/serialization/onnx/eddl_onnx.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

using namespace ecvl;
using namespace eddl;
using namespace std;
using namespace std::filesystem;

void EnsembleWithoutPostProc(Image img_tot, Image gt, path name, Eval& evaluator)
{
    cout << "Dice " << name << ": " << evaluator.DiceCoefficient(img_tot, gt) << endl;
}

void EnsemblePostProc(Image img_tot, Image gt, path name, Eval& evaluator)
{
    int counter = 0, min_area = 300;
    double top_score_thresh = 0.5, bottom_score_thresh = 0.3;
    
    auto i_img_tot = img_tot.ContiguousBegin<float>(), e_img_tot = img_tot.ContiguousEnd<float>();
    
    for (; i_img_tot != e_img_tot; ++i_img_tot) {
        if(*i_img_tot >= top_score_thresh) {
            ++counter;
        }
    }
    
    if(counter >= min_area) {
        cout << "Dice " << name << ": " << evaluator.DiceCoefficient(img_tot, gt, bottom_score_thresh) << endl;
    }
    else {
        memset(img_tot.data_, 0, img_tot.datasize_);
        cout << "Dice " << name << ": " << evaluator.DiceCoefficient(img_tot, gt) << endl;
    }
}

Image CreateTotalImage(vector<Image> v)
{
    Image img_f, img_tot;

    img_tot.Create({512,512,1}, DataType::float32, "xyc", ColorType::GRAY);
    memset(img_tot.data_, 0, img_tot.datasize_);

    auto i_img_tot = img_tot.ContiguousBegin<float>(), e_img_tot = img_tot.ContiguousEnd<float>();

    for (auto &img : v)
    {
        CopyImage(img, img_f, DataType::float32);
        Div(img_f, 255, img_f);
        auto i_img = img_f.ContiguousBegin<float>();
        i_img_tot = img_tot.ContiguousBegin<float>();

        for (; i_img_tot != e_img_tot; ++i_img_tot, ++i_img) {
            *i_img_tot += *i_img;
        }

    }
    Div(img_tot, vsize(v), img_tot);
    
    return img_tot;
}

void Binarization(vector<Image>& v)
{
    for (auto &img : v)
    {
        auto i_img = img.ContiguousBegin<uint8_t>(), e_img = img.ContiguousEnd<uint8_t>();
        for (; i_img != e_img; ++i_img) {
            *i_img = ((*i_img) < 129) ? 0 : 255;
        }
    }
}

int main(int argc, char** argv)
{
    cxxopts::Options options("DeepHealth pipeline pneumothorax validation inference", "");

    options.add_options()
        ("s,segnet_ce", "Folder with output images from the baseline", cxxopts::value<path>())
        ("d,segnet_dice", "Folder with output images from SegNetBN with dice loss", cxxopts::value<path>())
        ("c,segnet_combo", "Folder with output images from SegNetBN with combo loss", cxxopts::value<path>())
        ("u,unet", "Folder with output images from the UNet with BCE", cxxopts::value<path>())
        ("g,ground_truth", "Folder with ground_truth images", cxxopts::value<path>())
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        cout << options.help() << endl;
        return EXIT_SUCCESS;
    }

    Image img_1, img_2, img_3, img_4, gt;
    Image gt_f;
    path segnet_ce, segnet_dice, segnet_combo, unet, ground_truth;
    Eval evaluator;
    Eval evaluator_bin;
    Eval evaluator_post_proc;
    Eval evaluator_post_proc_bin;

    if (result.count("segnet_ce")) {
        segnet_ce = result["segnet_ce"].as<path>();
    }
    else {
        cout << ECVL_WARNING_MSG "'s,segnet_ce' is not provided." << endl;
    }
    if (result.count("segnet_dice")) {
        segnet_dice = result["segnet_dice"].as<path>();
    }
    else {
        cout << ECVL_WARNING_MSG "'d,segnet_dice' is not provided." << endl;
    }
    if (result.count("segnet_combo")) {
        segnet_combo = result["segnet_combo"].as<path>();
    }
    else {
        cout << ECVL_WARNING_MSG "'c,segnet_combo' is not provided." << endl;
    }
    if (result.count("unet")) {
        unet = result["unet"].as<path>();
    }
    else {
        cout << ECVL_WARNING_MSG "'u,unet' is not provided." << endl;
    }
    if (result.count("ground_truth")) {
        ground_truth = result["ground_truth"].as<path>();
    }
    else {
        cout << ECVL_ERROR_MSG "'g,ground_truth' is a required argument." << endl;
        return EXIT_FAILURE;
    }

    for (auto &x: directory_iterator(ground_truth)){
        vector<Image> v;
        path name = x.path().filename();

        if (!segnet_ce.empty()) {
            ImRead(segnet_ce / name, img_1, ImReadMode::GRAYSCALE);
            v.push_back(img_1);
        }
        if (!segnet_dice.empty()) {
            ImRead(segnet_dice / name, img_2, ImReadMode::GRAYSCALE);
            v.push_back(img_2);
        }
        if (!segnet_combo.empty()) {
            ImRead(segnet_combo / name, img_3, ImReadMode::GRAYSCALE);
            v.push_back(img_3);
        }
        if (!unet.empty()) {
            ImRead(unet / name, img_4, ImReadMode::GRAYSCALE);
            v.push_back(img_4);
        }
        ImRead(x, gt, ImReadMode::GRAYSCALE);

        Image img_tot = CreateTotalImage(v);
        CopyImage(gt, gt_f, DataType::float32);
        Div(gt_f, 255, gt_f);

        EnsembleWithoutPostProc(img_tot, gt_f, name, evaluator);

        EnsemblePostProc(img_tot, gt_f, name, evaluator_post_proc);

        Binarization(v);
        img_tot = CreateTotalImage(v);

        EnsembleWithoutPostProc(img_tot, gt_f, name, evaluator_bin);
        EnsemblePostProc(img_tot, gt_f, name, evaluator_post_proc_bin);
    }

    cout << "----------------------" << endl;
    cout << "Mean Dice Coefficient: " << evaluator.MeanMetric() << endl;
    cout << "----------------------" << endl;

    cout << "-------------------------------------------" << endl;
    cout << "Mean Dice Coefficient with Post Processing: " << evaluator_post_proc.MeanMetric() << endl;
    cout << "-------------------------------------------" << endl;

    cout << "--------------------------------------------" << endl;
    cout << "Mean Dice Coefficient with binarized images: " << evaluator_bin.MeanMetric() << endl;
    cout << "--------------------------------------------" << endl;

    cout << "----------------------------------------------------------------" << endl;
    cout << "Mean Dice Coefficient with Post Processing and binarized images: " << evaluator_post_proc_bin.MeanMetric() << endl;
    cout << "----------------------------------------------------------------" << endl;

    return EXIT_SUCCESS;
}