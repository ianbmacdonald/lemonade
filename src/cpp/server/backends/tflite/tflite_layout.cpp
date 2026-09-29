#include "lemon/backends/tflite/tflite_layout.h"

#include "lemon/backends/hf_cache_util.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace lemon {
namespace backends {
namespace tflite {

namespace {

constexpr std::uintmax_t kMaxManifestBytes = 1024 * 1024;

// True only for a readable manifest declaring the image-classification task. A
// text model's optional manifest carries no task, so it falls through to Text.
bool read_image_manifest(const fs::path& dir, std::string& labels_file) {
    std::error_code ec;
    const fs::path manifest = dir / "manifest.json";
    if (!fs::is_regular_file(manifest, ec) || ec) return false;
    const auto size = fs::file_size(manifest, ec);
    if (ec || size > kMaxManifestBytes) return false;

    std::ifstream in(manifest, std::ios::binary);
    if (!in) return false;
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const nlohmann::json doc = nlohmann::json::parse(text, nullptr, false);
    if (!doc.is_object()) return false;
    const auto task = doc.find("task");
    if (task == doc.end() || !task->is_string() ||
        task->get<std::string>() != "image-classification") {
        return false;
    }
    labels_file = "labels.txt";
    const auto labels = doc.find("labels_file");
    if (labels != doc.end()) {
        if (!labels->is_string()) return false;
        labels_file = labels->get<std::string>();
    }
    // The labels file must sit in the model dir; a path that climbs out of it
    // would let a manifest point the backend anywhere on disk.
    const fs::path rel(labels_file);
    if (labels_file.empty() || rel.is_absolute() || rel.has_root_path()) return false;
    for (const auto& part : rel) {
        if (part == "..") return false;
    }
    return true;
}

}  // namespace

TfliteLayout detect_tflite_layout(const fs::path& dir) noexcept {
    try {
        std::error_code ec;
        if (!fs::exists(dir / "model.tflite", ec) || ec) return TfliteLayout::None;

        std::string labels_file;
        if (read_image_manifest(dir, labels_file)) {
            return fs::is_regular_file(dir / fs::path(labels_file), ec) && !ec
                       ? TfliteLayout::Image
                       : TfliteLayout::None;
        }
        // The config is mandatory even when a manifest is present: the manifest
        // describes the output contract only, and the backend checks the
        // architecture in config.json against its supported input convention.
        if (fs::exists(dir / "tokenizer.json", ec) && fs::exists(dir / "config.json", ec)) {
            return TfliteLayout::Text;
        }
    } catch (...) {
    }
    return TfliteLayout::None;
}

const char* tflite_layout_name(TfliteLayout layout) noexcept {
    switch (layout) {
        case TfliteLayout::Text: return "text-classifier";
        case TfliteLayout::Image: return "image-classifier";
        default: return "unrecognized";
    }
}

std::vector<fs::path> find_complete_model_dirs(const fs::path& root) {
    std::vector<fs::path> dirs;
    std::error_code ec;
    fs::recursive_directory_iterator it(root, hf_cache::dir_options(), ec);
    if (ec) return dirs;
    for (auto end = fs::recursive_directory_iterator(); it != end; it.increment(ec)) {
        if (ec) break;
        if (it->is_regular_file(ec) && !ec && it->path().filename() == "model.tflite" &&
            detect_tflite_layout(it->path().parent_path()) != TfliteLayout::None) {
            dirs.push_back(it->path().parent_path());
        }
    }
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

}  // namespace tflite
}  // namespace backends
}  // namespace lemon
