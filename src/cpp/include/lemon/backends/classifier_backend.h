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

// The layout, and for a dir that holds a graph but is not a model, why not.
struct Inspection {
    Layout layout;
    std::string problem;
    // False when the only problem is missing layout files: no manifest.json
    // declaring the image task, and not both tokenizer.json and config.json (a
    // stray graph, or a text model still downloading). That is not logged or
    // listed as a refusal; manifest and labels failures are.
    bool from_files = true;
};

// Runs during bulk model listing, so neither throws.
Inspection inspect_layout(const std::filesystem::path& dir,
                          const std::vector<std::string>& graphs) noexcept;
Layout detect_layout(const std::filesystem::path& dir,
                     const std::vector<std::string>& graphs) noexcept;

const char* layout_name(Layout layout) noexcept;

// Every directory under root that holds a servable model, sorted; a directory
// holding several of `graphs` is listed once. A directory that holds a graph but
// is not a model is logged once, with the reason, as a WARNING.
std::vector<std::filesystem::path> find_model_dirs(const std::filesystem::path& root,
                                                   const std::vector<std::string>& graphs);

// One "\n  <dir>: <reason>" line per directory under root that holds a graph
// but is not a model; empty when there is none. For load errors.
std::string refused_model_dirs(const std::filesystem::path& root,
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

// An engine without a POST /classify/image route answers it with cpp-httplib's
// default 404, whose body is empty; an engine's own 404 carries one. True for
// that response, as forward_multipart_request returns it. For requests after
// the load only; the load probe uses classify_route_probe.
bool is_missing_image_route(const nlohmann::json& response);

// What the load probe (an image request without an image) found. Only a 4xx
// other than 404 shows the route exists; any 404 means it does not; everything
// else (2xx, 5xx, a timeout, a crash) is Failed and `outcome` says what happened.
enum class RouteProbe { Present, Missing, Failed };
RouteProbe classify_route_probe(const nlohmann::json& response, std::string& outcome);

// The one wording for an engine without the image route, used both when a load
// probe finds it and when a request does. `requirement` says which engine
// release has the route (the descriptor's kImageRouteRequirement).
std::string missing_image_route_message(const std::string& binary,
                                        const std::string& requirement);

// The 502 image_classification_unsupported carrying that message.
nlohmann::json missing_image_route_error(const std::string& binary,
                                         const std::string& requirement);

// Longer than the backend watchdog takes to kill a hung engine (about 105 s
// with the defaults), so a hang ends in a reset rather than a network error.
constexpr long kImageForwardTimeoutSeconds = 120;

}  // namespace classifier
}  // namespace backends
}  // namespace lemon
