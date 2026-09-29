#pragma once

#include <filesystem>
#include <vector>

namespace lemon {
namespace backends {
namespace tflite {

enum class TfliteLayout { None, Text, Image };

// Text:  model.tflite + tokenizer.json + config.json (manifest.json optional).
// Image: model.tflite + manifest.json {"task": "image-classification"} + the
//        labels file the manifest names.
// Runs during bulk model listing, so it never throws.
TfliteLayout detect_tflite_layout(const std::filesystem::path& dir) noexcept;

const char* tflite_layout_name(TfliteLayout layout) noexcept;

// Every directory under root that holds a servable model, sorted.
std::vector<std::filesystem::path> find_complete_model_dirs(const std::filesystem::path& root);

}  // namespace tflite
}  // namespace backends
}  // namespace lemon
