#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <lemon/utils/aixlog.hpp>

#include "lemon/image_classify_request.h"
#include "lemon/model_types.h"
#include "lemon/models/model_info.h"
#include "lemon/router.h"
#include "lemon/server/api_route.h"
#include "lemon/server/model_loader.h"

namespace lemon {
namespace {

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

const RouteError kBadContentType{400, "", "`Content-Type` is neither `multipart/form-data` nor `application/json`"};
const RouteError kBadImageRequest{
    400, "",
    "Malformed request: zero or several image parts, invalid base64, a remote URL, a non-JPEG/PNG "
    "image, a bad `top_k`, or an image the backend cannot decode"};
const RouteError kImageTooLarge{413, "payload_too_large", "The decoded image exceeds 16 MiB, or the body exceeds 23 MiB"};
const RouteError kLengthRequired{
    411, "length_required",
    "No `Content-Length`, or any `Transfer-Encoding` header, even alongside a `Content-Length`; "
    "refused before the body is read"};
const RouteError kContentEncoding{
    415, "unsupported_content_encoding",
    "Any `Content-Encoding` header, such as a gzip-compressed body; refused before the body is read"};
const RouteError kServerBusy{
    503, "server_busy",
    "Two image-classification requests are already in progress (`Retry-After: 1`)"};
const RouteError kCrashCooldown{
    503, "backend_crash_cooldown",
    "The model's backend crashed on 3 inputs within 60 seconds, which pauses image classification "
    "for that model for 60 seconds (with `Retry-After`)"};
const RouteError kNotImageClassifier{400, "model_not_applicable", "`model` is not an image-classification model"};
const RouteError kNoSoleImageClassifier{
    400, "", "`model` is omitted and not exactly one image-classification model is loaded"};
const RouteError kCrashedOnInput{
    502, "backend_crashed_on_input",
    "The backend crashed on this input; the request is not retried, and the next request restarts the backend"};

const RouteArg kModel{"model", ArgIn::Form, {{"type", "string"}}, false, Support::Available,
                      "The image-classification model to use. Optional when exactly one is loaded."};
const RouteArg kImage{"image", ArgIn::Form, {{"type", "string"}, {"format", "binary"}}, true, Support::Available,
                      "Multipart: exactly one file part named `image` (or `file`). JSON: a base64 string, "
                      "or a `data:image/jpeg;base64,...` / `data:image/png;base64,...` URL. Remote "
                      "`http(s)` URLs are not fetched. At most 16 MiB decoded.", {"file"}};
const RouteArg kTopK{"top_k", ArgIn::Form, {{"type", "integer"}, {"minimum", 1}, {"maximum", 1000000}}, false, Support::Available,
                     "Number of labels to return. Defaults to 5; the backend caps it at the model's label count."};

int status_for_backend_error(const json& err) {
    if (err.contains("status_code") && err["status_code"].is_number_integer()) {
        return err["status_code"].get<int>();
    }
    // forward_multipart_request nests the backend's status under error.details.
    if (err.contains("details") && err["details"].is_object() &&
        err["details"].contains("status_code") &&
        err["details"]["status_code"].is_number_integer()) {
        return err["details"]["status_code"].get<int>();
    }
    if (err.contains("code") && err["code"].is_string()) {
        return ModelLoader::load_error(err["code"].get<std::string>()).status;
    }
    if (err.contains("type") && err["type"].is_string()) {
        const std::string type = err["type"].get<std::string>();
        if (type == "invalid_request" || type == "invalid_request_error" ||
            type == "unsupported_operation") {
            return 400;
        }
        if (type == "model_not_loaded") {
            return kModelUnloaded.status;
        }
    }
    return kBackendError.status;
}

class ImagesClassifyRoute : public ApiRoute {
public:
    using ApiRoute::ApiRoute;

    RouteSpec spec() const override {
        RouteSpec s;
        s.id = "lemonade.images_classify";
        s.methods = {"POST"};
        s.paths = {"images/classify"};
        s.summary = "Classify an image";
        s.experimental = true;
        s.description =
            "Classifies one image and returns the top-scoring labels. The model must be labeled "
            "`image-classification` and use the `tflite` recipe, which runs it in a `tflite-server` "
            "(0.2.0 or newer) subprocess on the CPU. The request is either `multipart/form-data` "
            "(the same shape as `/v1/images/edits`) or `application/json`. Only JPEG and PNG are "
            "accepted, and EXIF orientation is not applied.";
        s.notes = {
            "`data` is sorted by descending score and is authoritative. `labels` has the same shape "
            "as `/v1/classify`, but some label sets repeat names (ImageNet has two `crane` "
            "classes), so a repeated name keeps only its highest score there; use `data[].index` "
            "to tell classes apart.",
            "The body is capped before it is read: a request without `Content-Length`, or with any "
            "`Transfer-Encoding`, gets `411`; any `Content-Encoding` gets `415`; and a "
            "`Content-Length` over 23 MiB gets `413`. At most two requests run at once.",
            "An image that crashes the backend is never replayed, because replaying it would crash "
            "the restarted backend too: this request fails with `502` `backend_crashed_on_input` "
            "and the next request restarts the backend.",
            "### Model Directory\n\n"
            "An image model directory holds `model.tflite` (float32 input `[1, H, W, 3]`, float32 "
            "output `[1, N]`), a labels file with `N` lines, and a `manifest.json` that declares the "
            "task and preprocessing:\n\n"
            "```json\n"
            "{\n"
            "  \"task\": \"image-classification\",\n"
            "  \"labels_file\": \"labels.txt\",\n"
            "  \"score_normalization\": \"none\",\n"
            "  \"preprocess\": {\n"
            "    \"resize\": \"stretch\",\n"
            "    \"mean\": [127.5, 127.5, 127.5],\n"
            "    \"std\": [127.5, 127.5, 127.5],\n"
            "    \"channel_order\": \"RGB\",\n"
            "    \"layout\": \"auto\"\n"
            "  },\n"
            "  \"top_k_default\": 5\n"
            "}\n"
            "```\n\n"
            "A text model directory (`model.tflite` + `tokenizer.json` + `config.json`) keeps "
            "serving `/v1/classify`. Loading fails with a message naming both layouts when a "
            "model's label and its directory disagree, and when the installed `tflite-server` is "
            "too old to report the image-classification task. The per-model `tflite_args` option "
            "passes flags such as `--max-image-pixels` through to `tflite-server`.",
        };
        s.args = {kModel, kImage, kTopK};

        RouteResponse response;
        response.format = ResponseFormat::Json;
        response.schema = ordered_json::parse(R"({
            "type": "object",
            "required": ["object", "model", "data", "labels"],
            "properties": {
                "object": {"const": "image_classification", "description": "Always `image_classification`."},
                "model": {"type": "string", "description": "The model that classified the image."},
                "data": {
                    "type": "array",
                    "description": "The highest-scoring classes, best first.",
                    "items": {
                        "type": "object",
                        "required": ["index", "label", "score"],
                        "properties": {
                            "index": {"type": "integer", "description": "The class index in the model's output."},
                            "label": {"type": "string", "description": "The class name, or the index as a string when the model has no label file."},
                            "score": {"type": "number", "description": "The class score."}
                        }
                    }
                },
                "labels": {
                    "type": "object",
                    "description": "Each class's score keyed by label; a repeated label keeps its highest score, and `data` is authoritative.",
                    "additionalProperties": {"type": "number"}
                },
                "timings": {"type": "object", "description": "Backend timings, when the backend reports them."}
            }
        })");
        s.responses = {response};
        s.errors = join_errors({{kBadContentType, kBadImageRequest, kImageTooLarge, kLengthRequired, kContentEncoding,
                                 kServerBusy, kCrashCooldown,
                                 kNotImageClassifier, kNoSoleImageClassifier, kCrashedOnInput,
                                 kModelUnloaded, kBackendError, kMalformedBackendResponse,
                                 kUnexpectedException},
                                ModelLoader::errors()});
        s.error_format = ErrorFormat::OpenAi;
        s.request_format = RequestFormat::Raw;
        return s;
    }

    void handle(RouteRequest& req, httplib::Response& res) override {
        const httplib::Request& http = req.http;
        auto send_if_paused = [this, &res](const std::string& model) {
            auto paused = ctx_.router->image_classify_cooldown_error(model);
            if (!paused) return false;
            res.status = 503;
            res.set_header("Retry-After",
                           std::to_string((*paused)["error"]["retry_after"].get<long long>()));
            res.set_content(paused->dump(), "application/json");
            return true;
        };
        try {
            std::uint64_t received = http.body.size();
            for (const auto& [name, field] : http.form.fields) received += name.size() + field.content.size();
            for (const auto& [name, file] : http.form.files) received += name.size() + file.content.size();
            if (received > image_classify::kMaxImageClassifyRequestBytes) {
                write_openai_error(res, kImageTooLarge,
                                   "Request body exceeds " +
                                       std::to_string(image_classify::kMaxImageClassifyRequestBytes /
                                                      (1024 * 1024)) + " MiB");
                return;
            }

            image_classify::ParsedRequest parsed;
            if (http.is_multipart_form_data()) {
                std::optional<std::string> model;
                std::optional<std::string> top_k;
                if (http.form.has_field("model")) model = http.form.get_field("model");
                if (http.form.has_field("top_k")) top_k = http.form.get_field("top_k");
                std::vector<std::string_view> parts;
                for (const auto& [name, file] : http.form.files) {
                    if (name == "image" || name == "file") parts.emplace_back(file.content);
                }
                parsed = image_classify::from_form(model, top_k, parts);
            } else {
                const std::string content_type = http.get_header_value("Content-Type");
                if (!content_type.empty() &&
                    content_type.find("application/json") == std::string::npos) {
                    write_openai_error(res, kBadContentType,
                                       "Content-Type must be multipart/form-data or application/json");
                    return;
                }
                parsed = image_classify::parse_json(http.body);
            }
            if (!parsed.ok()) {
                const RouteError& error = parsed.status == 413 ? kImageTooLarge : kBadImageRequest;
                RouteError sent = error;
                sent.status = parsed.status;
                write_openai_error(res, sent, parsed.error);
                return;
            }

            json params = std::move(parsed.params);
            std::string model;
            if (params.contains("model") && params["model"].is_string()) {
                model = canonical_model_name(ctx_, params["model"].get<std::string>());
                params["model"] = model;
            }

            if (!model.empty()) {
                if (auto info = ctx_.router->try_get_model_info(model);
                    info && info->type != ModelType::IMAGE_CLASSIFICATION) {
                    write_openai_error(res, kNotImageClassifier,
                                       "model '" + model + "' is not an image-classification model");
                    return;
                }
                if (send_if_paused(model)) return;
                try {
                    ctx_.model_loader->ensure_loaded(model);
                } catch (const std::exception& e) {
                    LOG(ERROR, "Server") << "Failed to load image-classification model: "
                                         << e.what() << std::endl;
                    ctx_.model_loader->write_load_error(res, model, e.what());
                    return;
                }
            } else {
                model = ctx_.router->get_sole_loaded_model_of_type(
                    ModelType::IMAGE_CLASSIFICATION, /*include_dead_backends=*/true);
                if (model.empty()) {
                    write_openai_error(res, kNoSoleImageClassifier,
                        "No 'model' specified and no single image-classification model "
                        "is loaded (load one, or name it in the request)");
                    return;
                }
                if (send_if_paused(model)) return;
                params["model"] = model;
            }

            auto response = ctx_.router->classify_image(params, std::move(parsed.bytes));
            if (response.contains("error") && response["error"].is_object()) {
                res.status = status_for_backend_error(response["error"]);
                if (response["error"].contains("retry_after") &&
                    response["error"]["retry_after"].is_number_integer()) {
                    res.set_header("Retry-After",
                                   std::to_string(response["error"]["retry_after"].get<long long>()));
                }
                res.set_content(response.dump(), "application/json");
                return;
            }

            // The envelope is the public contract, not the backend's raw body.
            auto malformed = [&res]() {
                write_openai_error(res, kMalformedBackendResponse,
                                   "backend returned malformed image classification",
                                   "classification_error");
            };
            if (!response.contains("predictions") || !response["predictions"].is_array()) {
                malformed();
                return;
            }
            json data = json::array();
            json labels = json::object();
            for (const auto& p : response["predictions"]) {
                if (!p.is_object() || !p.contains("index") || !p["index"].is_number_integer() ||
                    !p.contains("score") || !p["score"].is_number()) {
                    malformed();
                    return;
                }
                const std::string label = p.contains("label") && p["label"].is_string()
                                              ? p["label"].get<std::string>()
                                              : std::to_string(p["index"].get<long long>());
                data.push_back({{"index", p["index"]}, {"label", label}, {"score", p["score"]}});
                if (!labels.contains(label) ||
                    labels[label].get<double>() < p["score"].get<double>()) {
                    labels[label] = p["score"];
                }
            }
            // The documented contract is best-first; do not rely on the backend's order.
            std::stable_sort(data.begin(), data.end(), [](const json& a, const json& b) {
                return a["score"].get<double>() > b["score"].get<double>();
            });
            json enveloped = {
                {"object", "image_classification"},
                {"model", model},
                {"data", std::move(data)},
                {"labels", std::move(labels)},
            };
            if (response.contains("timings") && response["timings"].is_object()) {
                enveloped["timings"] = response["timings"];
            }
            res.set_content(enveloped.dump(), "application/json");
        } catch (const std::exception& e) {
            LOG(ERROR, "Server") << "ERROR in images/classify: " << e.what() << std::endl;
            write_openai_error(res, kUnexpectedException, e.what(), "internal_error");
        }
    }
};

} // namespace

std::unique_ptr<ApiRoute> make_images_classify_route(ServerContext& ctx) {
    return std::make_unique<ImagesClassifyRoute>(ctx);
}

} // namespace lemon
