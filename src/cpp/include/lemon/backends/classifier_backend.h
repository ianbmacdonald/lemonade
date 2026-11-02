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
// Text:  a graph + tokenizer.json + config.json (manifest.json optional, and
//        not declaring the image task).
// Image: a graph + manifest.json {"task": "image-classification"} + the
//        labels file the manifest names. The manifest and the labels file
//        must resolve inside the model dir, or inside its registry repository
//        for a snapshot. A manifest that declares the image task but fails a
//        check makes the dir None, never Text.
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
// layouts when dir holds the other one or none.
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

// An engine older than its first image release answers 404 on /classify/image,
// which would reach the client as a bare 404 on every request. Turns that 404
// (in a normalized response) into a 502 naming the release image models need,
// such as "et-server 0.2.0".
nlohmann::json require_image_route(nlohmann::json response, const std::string& engine_release);

// Longer than the backend watchdog takes to kill a hung engine (about 105 s
// with the defaults), so a hang ends in a reset rather than a network error.
constexpr long kImageForwardTimeoutSeconds = 120;

}  // namespace classifier
}  // namespace backends
}  // namespace lemon
