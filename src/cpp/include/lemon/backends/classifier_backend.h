#pragma once

#include "lemon/model_types.h"
#include "lemon/utils/http_client.h"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace lemon {
namespace backends {
namespace classifier {

// The model directory layouts shared by the classification engines
// (tflite-server, et-server, ort-server). `graphs` names the graph files an
// engine loads (model.tflite; model.pte; model.onnx or model.ort); any one of
// them counts.
//
// Text:  a graph + tokenizer.json + config.json (manifest.json optional).
// Image: a graph + manifest.json {"task": "image-classification"} + the
//        labels file the manifest names.
enum class Layout { None, Text, Image };

// Runs during bulk model listing, so it never throws.
Layout detect_layout(const std::filesystem::path& dir,
                     const std::vector<std::string>& graphs) noexcept;

const char* layout_name(Layout layout) noexcept;

// Every directory under root that holds a servable model, sorted; a directory
// holding several of `graphs` is listed once.
std::vector<std::filesystem::path> find_model_dirs(const std::filesystem::path& root,
                                                   const std::vector<std::string>& graphs);

// The layout `type` needs. Throws NonRetryableLoadException naming both
// layouts when dir holds the other one.
Layout require_layout(const std::string& model_name, ModelType type,
                      const std::filesystem::path& dir,
                      const std::vector<std::string>& graphs);

// The 400 for a request on the route that the loaded layout does not serve.
nlohmann::json wrong_route_error(Layout loaded);

// The multipart body of the engines' POST /classify/image: the image and top_k
// (lemond's default 5 when params carry none).
std::vector<utils::MultipartField> image_classify_fields(const nlohmann::json& params,
                                                         std::string image_bytes);

// forward_multipart_request nests the engine's status and body under
// error.details. Lifts the engine's own message, code and status to the top so
// the client sees "image too large" with its 4xx, not a generic 500.
nlohmann::json normalize_backend_error(nlohmann::json response);

// Longer than the backend watchdog takes to kill a hung engine (about 105 s
// with the defaults), so a hang ends in a reset rather than a network error.
constexpr long kImageForwardTimeoutSeconds = 120;

}  // namespace classifier
}  // namespace backends
}  // namespace lemon
