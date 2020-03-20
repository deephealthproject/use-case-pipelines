#include <filesystem>
#include <iostream>

using namespace std;
using namespace std::filesystem;

int main()
{
    path train_images("/path/to/siim-train-test/siim/dicom-images-train");
    path ground_truth("/path/to/pneumothorax_ground_truth");

    try {
        copy(ground_truth / "black.png", train_images / "black.png", copy_options::overwrite_existing);
    }
    catch (filesystem_error & e) {
        cout << "Could not copy black.png: " << e.what() << '\n';
    }

    for (auto& it : recursive_directory_iterator(train_images)) {
        path p = it.path();
        if (p.extension() == ".dcm") {
            path filename = p.filename();
            if (exists(ground_truth / filename.replace_extension(".png"))) {
                try {
                    copy(ground_truth / filename, p.replace_filename(filename), copy_options::overwrite_existing);
                }
                catch (filesystem_error & e) {
                    cout << "Could not copy " << p << ": " << e.what() << '\n';
                }
            }
        }
    }

    return EXIT_SUCCESS;
}