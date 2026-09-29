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
const RouteError kBadImageRequest{400, "", "The image is missing, not strict base64, not a JPEG or PNG, or `top_k` is invalid"};
const RouteError kImageTooLarge{413, "payload_too_large", "The decoded image exceeds 16 MiB, or the body exceeds 23 MiB"};
const RouteError kLengthRequired{411, "length_required", "The request carries no `Content-Length` (a chunked upload)"};
const RouteError kNotImageClassifier{400, "model_not_applicable", "`model` is not an image-classification model"};
const RouteError kNoSoleImageClassifier{
    400, "", "`model` is omitted and not exactly one image-classification model is loaded"};
const RouteError kCrashedOnInput{
    502, "backend_crashed_on_input",
    "The backend crashed on this input; it is reloaded and the request is not retried"};

const RouteArg kModel{"model", ArgIn::Form, {{"type", "string"}}, false, Support::Available,
                      "The image-classification model to use. Optional when exactly one is loaded."};
const RouteArg kImage{"image", ArgIn::Form, {{"type", "string"}, {"format", "binary"}}, true, Support::Available,
                      "The JPEG or PNG image, at most 16 MiB: a multipart file part named `image` (or "
                      "`file`), or in a JSON body a base64 string or `data:` URL.", {"file"}};
const RouteArg kTopK{"top_k", ArgIn::Form, {{"type", "integer"}, {"minimum", 1}, {"maximum", 1000000}}, false, Support::Available,
                     "Return only the highest-scoring `k` classes. Defaults to 5."};

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
        s.id = "openai.images_classify";
        s.methods = {"POST"};
        s.paths = {"images/classify"};
        s.summary = "Classify an image";
        s.modality = "image -> class scores";
        s.experimental = true;
        s.description =
            "Runs an image-classification model, such as `MobileNetV2-1.0-224-TFLite` served by "
            "`tflite-server`, on one JPEG or PNG image and returns the highest-scoring classes. The "
            "image is sent as a multipart file part, or as base64 in a JSON body.";
        s.notes = {
            "The body is capped before it is read: a request without `Content-Length` gets `411`, "
            "and one over 23 MiB gets `413`.",
            "An image that crashes the backend is never replayed: the backend is reloaded and the "
            "request fails with `502` `backend_crashed_on_input`.",
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
        s.errors = join_errors({{kBadContentType, kBadImageRequest, kImageTooLarge, kLengthRequired,
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
        try {
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
                try {
                    ctx_.model_loader->ensure_loaded(model);
                } catch (const std::exception& e) {
                    LOG(ERROR, "Server") << "Failed to load image-classification model: "
                                         << e.what() << std::endl;
                    ctx_.model_loader->write_load_error(res, model, e.what());
                    return;
                }
            } else {
                model = ctx_.router->get_sole_loaded_model_of_type(ModelType::IMAGE_CLASSIFICATION);
                if (model.empty()) {
                    write_openai_error(res, kNoSoleImageClassifier,
                        "No 'model' specified and no single image-classification model "
                        "is loaded (load one, or name it in the request)");
                    return;
                }
                params["model"] = model;
            }

            auto response = ctx_.router->classify_image(params, std::move(parsed.bytes));
            if (response.contains("error") && response["error"].is_object()) {
                res.status = status_for_backend_error(response["error"]);
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
                if (!labels.contains(label)) labels[label] = p["score"];
            }
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
