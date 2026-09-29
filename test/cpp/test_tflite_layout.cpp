// Model-directory layout detection for the tflite backend: which directories
// are a text classifier, an image classifier, or neither.

#include "lemon/backends/tflite/tflite_layout.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using lemon::backends::tflite::TfliteLayout;
using lemon::backends::tflite::detect_tflite_layout;
using lemon::backends::tflite::find_complete_model_dirs;

namespace {

int failures = 0;

void check(const std::string& what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

void write(const fs::path& p, const std::string& content) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    out << content;
}

const char* kImageManifest =
    R"({"task":"image-classification","labels_file":"labels.txt"})";

void make_text(const fs::path& dir) {
    write(dir / "model.tflite", "x");
    write(dir / "tokenizer.json", "{}");
    write(dir / "config.json", "{}");
}

void make_image(const fs::path& dir) {
    write(dir / "model.tflite", "x");
    write(dir / "manifest.json", kImageManifest);
    write(dir / "labels.txt", "background\ncat\n");
}

}  // namespace

int main() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root =
        fs::temp_directory_path() / ("lemon_tflite_layout_" + std::to_string(stamp));
    fs::create_directories(root);

    make_text(root / "text");
    check("text layout is Text", detect_tflite_layout(root / "text") == TfliteLayout::Text);

    make_text(root / "text_manifest");
    write(root / "text_manifest" / "manifest.json", R"({"task":"text-classification"})");
    check("text layout with a text manifest is Text",
          detect_tflite_layout(root / "text_manifest") == TfliteLayout::Text);

    make_image(root / "image");
    check("image manifest + labels is Image",
          detect_tflite_layout(root / "image") == TfliteLayout::Image);

    write(root / "image_default_labels" / "model.tflite", "x");
    write(root / "image_default_labels" / "manifest.json", R"({"task":"image-classification"})");
    write(root / "image_default_labels" / "labels.txt", "a\n");
    check("labels_file defaults to labels.txt",
          detect_tflite_layout(root / "image_default_labels") == TfliteLayout::Image);

    write(root / "image_no_labels" / "model.tflite", "x");
    write(root / "image_no_labels" / "manifest.json", kImageManifest);
    check("image manifest with its labels file missing is None",
          detect_tflite_layout(root / "image_no_labels") == TfliteLayout::None);

    write(root / "image_escape" / "model.tflite", "x");
    write(root / "image_escape" / "manifest.json",
          R"({"task":"image-classification","labels_file":"../text/config.json"})");
    check("labels_file outside the model dir is None",
          detect_tflite_layout(root / "image_escape") == TfliteLayout::None);

    write(root / "text_no_tokenizer" / "model.tflite", "x");
    write(root / "text_no_tokenizer" / "config.json", "{}");
    check("text layout without a tokenizer is None",
          detect_tflite_layout(root / "text_no_tokenizer") == TfliteLayout::None);

    write(root / "corrupt" / "model.tflite", "x");
    write(root / "corrupt" / "manifest.json", "{not json");
    write(root / "corrupt" / "labels.txt", "a\n");
    check("corrupt manifest is None without throwing",
          detect_tflite_layout(root / "corrupt") == TfliteLayout::None);

    write(root / "huge" / "model.tflite", "x");
    write(root / "huge" / "labels.txt", "a\n");
    write(root / "huge" / "manifest.json",
          std::string(R"({"task":"image-classification","pad":")") +
              std::string(2 * 1024 * 1024, 'a') + "\"}");
    check("2 MiB manifest is None without throwing",
          detect_tflite_layout(root / "huge") == TfliteLayout::None);

    check("missing directory is None",
          detect_tflite_layout(root / "does_not_exist") == TfliteLayout::None);

    const fs::path two = root / "two";
    make_text(two / "a");
    make_image(two / "b");
    check("two complete dirs are both found", find_complete_model_dirs(two).size() == 2);
    check("one complete dir is found", find_complete_model_dirs(root / "image").size() == 1);

    std::error_code ec;
    fs::remove_all(root, ec);

    if (failures == 0) {
        std::printf("\nAll tflite layout checks passed.\n");
        return 0;
    }
    std::printf("\n%d tflite layout check(s) failed.\n", failures);
    return 1;
}
