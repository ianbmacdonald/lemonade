#include "lemon/backends/classifier_backend.h"

#include "lemon/backends/hf_cache_util.h"
#include "lemon/error_types.h"
#include "lemon/image_classify_request.h"
#include "lemon/utils/path_utils.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <set>
#include <system_error>
#include <utility>

#include <lemon/utils/aixlog.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace lemon {
namespace backends {
namespace classifier {

namespace {

constexpr std::uintmax_t kMaxManifestBytes = 1024 * 1024;

// The registry cache names a repository <prefix><org>--<name>, the prefix being
// "models--" (Hugging Face) or "modelscope--models--" (ModelScope).
bool is_registry_repo_dir(const fs::path& dir) {
    const std::string name = utils::path_to_utf8(dir.filename());
    return name.rfind("models--", 0) == 0 || name.rfind("modelscope--models--", 0) == 0;
}

// A registry snapshot links each file into its repository's blobs/, so a model
// file may resolve anywhere in that repository; anywhere else it must resolve
// inside the model dir. A directory that is merely named "snapshots" does not
// widen the root.
fs::path confinement_root(const fs::path& abs_dir) {
    for (fs::path p = abs_dir; p.has_relative_path(); p = p.parent_path()) {
        if (p.filename() == "snapshots" && is_registry_repo_dir(p.parent_path())) {
            return p.parent_path();
        }
    }
    return abs_dir;
}

enum class Resolution { Inside, Outside, Unresolvable };

// Where dir/rel resolves relative to dir's confinement root, with the target
// when it is inside. Checked on the resolved target, since a lexical check
// does not see through symlinks. An escape and a path that cannot be resolved
// at all are told apart, so the reason a model is dropped can say which.
Resolution resolve_confined(const fs::path& dir, const fs::path& rel, fs::path& target,
                            std::string& error) {
    std::error_code ec;
    const fs::path abs_dir = fs::absolute(dir, ec);
    if (!ec) {
        const fs::path base = fs::canonical(confinement_root(abs_dir), ec);
        if (!ec) {
            target = fs::canonical(abs_dir / rel, ec);
            if (!ec) {
                return std::mismatch(base.begin(), base.end(), target.begin(), target.end())
                                   .first == base.end()
                           ? Resolution::Inside
                           : Resolution::Outside;
            }
        }
    }
    error = ec.message();
    return Resolution::Unresolvable;
}

enum class ImageDeclaration { Absent, Valid, Invalid };

// Valid only for a confined, readable manifest declaring the image-classification
// task with a lexically in-dir labels file. A manifest that declares the task but
// fails any check is Invalid, never Absent: falling through to the text layout
// would load an explicit image model as a text classifier. A manifest that cannot
// be read safely (escaping or unresolvable link, not a regular file, oversized,
// not a JSON object) is Invalid too, since what it declares is unknown. Only a
// manifest without the image task is Absent. `problem` says why it is Invalid.
ImageDeclaration read_image_manifest(const fs::path& dir, std::string& labels_file,
                                     std::string& problem) {
    std::error_code ec;
    const fs::path manifest = dir / "manifest.json";
    const fs::file_type type = fs::symlink_status(manifest, ec).type();
    if (type == fs::file_type::not_found) return ImageDeclaration::Absent;
    if (ec) {
        problem = "manifest.json cannot be inspected (" + ec.message() + ")";
        return ImageDeclaration::Invalid;
    }

    fs::path target;
    std::string error;
    switch (resolve_confined(dir, "manifest.json", target, error)) {
        case Resolution::Outside:
            problem = "manifest.json resolves outside the model directory";
            return ImageDeclaration::Invalid;
        case Resolution::Unresolvable:
            problem = "manifest.json cannot be resolved (" + error + ")";
            return ImageDeclaration::Invalid;
        case Resolution::Inside:
            break;
    }
    if (!fs::is_regular_file(target, ec) || ec) {
        problem = "manifest.json is not a regular file";
        return ImageDeclaration::Invalid;
    }

    std::ifstream in(target, std::ios::binary);
    if (!in) {
        problem = "manifest.json cannot be read";
        return ImageDeclaration::Invalid;
    }
    std::string text(kMaxManifestBytes + 1, '\0');
    in.read(&text[0], static_cast<std::streamsize>(text.size()));
    if (static_cast<std::uintmax_t>(in.gcount()) > kMaxManifestBytes) {
        problem = "manifest.json is larger than 1 MiB";
        return ImageDeclaration::Invalid;
    }
    text.resize(static_cast<size_t>(in.gcount()));

    const json doc = json::parse(text, nullptr, false);
    if (!doc.is_object()) {
        problem = "manifest.json is not a JSON object";
        return ImageDeclaration::Invalid;
    }
    const auto task = doc.find("task");
    if (task == doc.end() || !task->is_string() ||
        task->get<std::string>() != "image-classification") {
        return ImageDeclaration::Absent;
    }
    labels_file = "labels.txt";
    const auto labels = doc.find("labels_file");
    if (labels != doc.end()) {
        if (!labels->is_string()) {
            problem = "manifest.json declares image-classification, but its labels_file is not "
                      "a string";
            return ImageDeclaration::Invalid;
        }
        labels_file = labels->get<std::string>();
    }
    // The labels file must sit in the model dir; a path that climbs out of it
    // would let a manifest point the backend anywhere on disk.
    const fs::path rel(labels_file);
    bool lexical_ok = !labels_file.empty() && !rel.is_absolute() && !rel.has_root_path();
    for (const auto& part : rel) {
        if (part == "..") lexical_ok = false;
    }
    if (!lexical_ok) {
        problem = "manifest.json declares image-classification, but its labels_file '" +
                  labels_file + "' is empty, absolute or climbs out of the model directory";
        return ImageDeclaration::Invalid;
    }
    return ImageDeclaration::Valid;
}

bool has_graph(const fs::path& dir, const std::vector<std::string>& graphs) {
    for (const auto& graph : graphs) {
        std::error_code ec;
        if (fs::exists(dir / graph, ec) && !ec) return true;
    }
    return false;
}

json invalid_request(const std::string& message) {
    return json{
        {"error", {
            {"message", message},
            {"type", "invalid_request_error"},
            {"status_code", 400},
        }}
    };
}

}  // namespace

Inspection inspect_layout(const fs::path& dir, const std::vector<std::string>& graphs) noexcept {
    try {
        std::error_code ec;
        if (!has_graph(dir, graphs)) return {Layout::None, ""};

        std::string labels_file;
        std::string problem;
        switch (read_image_manifest(dir, labels_file, problem)) {
            case ImageDeclaration::Valid: {
                fs::path labels;
                std::string error;
                switch (resolve_confined(dir, fs::path(labels_file), labels, error)) {
                    case Resolution::Outside:
                        return {Layout::None, "the labels file '" + labels_file +
                                                  "' resolves outside the model directory"};
                    case Resolution::Unresolvable:
                        return {Layout::None, "the labels file '" + labels_file +
                                                  "' cannot be resolved (" + error + ")"};
                    case Resolution::Inside:
                        break;
                }
                if (!fs::is_regular_file(labels, ec) || ec) {
                    return {Layout::None,
                            "the labels file '" + labels_file + "' is not a regular file"};
                }
                return {Layout::Image, ""};
            }
            case ImageDeclaration::Invalid:
                return {Layout::None, problem};
            case ImageDeclaration::Absent:
                break;
        }
        // The config is mandatory even when a manifest is present: the manifest
        // describes the output contract only, and the engine checks the
        // architecture in config.json against its supported input convention.
        if (fs::exists(dir / "tokenizer.json", ec) && fs::exists(dir / "config.json", ec)) {
            return {Layout::Text, ""};
        }
        return {Layout::None,
                "it has neither an image-classification manifest.json nor tokenizer.json + "
                "config.json"};
    } catch (const std::exception& e) {
        return {Layout::None, std::string("it cannot be inspected (") + e.what() + ")"};
    } catch (...) {
        return {Layout::None, "it cannot be inspected"};
    }
}

Layout detect_layout(const fs::path& dir, const std::vector<std::string>& graphs) noexcept {
    return inspect_layout(dir, graphs).layout;
}

const char* layout_name(Layout layout) noexcept {
    switch (layout) {
        case Layout::Text: return "text-classifier";
        case Layout::Image: return "image-classifier";
        default: return "unrecognized";
    }
}

namespace {

// Listing runs on every model-list build, so each dropped dir is logged once
// per reason rather than on every pass.
void warn_once(const fs::path& dir, const std::string& problem) {
    static std::mutex mutex;
    static std::set<std::string> logged;
    const std::string key = utils::path_to_utf8(dir) + "\n" + problem;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!logged.insert(key).second) return;
    }
    LOG(WARNING, "Classifier") << "Not a model: " << utils::path_to_utf8(dir) << ": " << problem
                               << std::endl;
}

void scan_model_dirs(const fs::path& root, const std::vector<std::string>& graphs,
                     std::vector<fs::path>* servable,
                     std::vector<std::pair<fs::path, std::string>>* refused) {
    std::error_code ec;
    fs::recursive_directory_iterator it(root, hf_cache::dir_options(), ec);
    if (ec) return;
    for (auto end = fs::recursive_directory_iterator(); it != end; it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file(ec) || ec) continue;
        const fs::path name = it->path().filename();
        const bool is_graph = std::any_of(graphs.begin(), graphs.end(),
                                          [&name](const std::string& g) { return name == g; });
        if (!is_graph) continue;
        const fs::path dir = it->path().parent_path();
        const Inspection found = inspect_layout(dir, graphs);
        if (found.layout != Layout::None) {
            if (servable) servable->push_back(dir);
        } else if (!found.problem.empty()) {
            warn_once(dir, found.problem);
            if (refused) refused->emplace_back(dir, found.problem);
        }
    }
}

}  // namespace

std::vector<fs::path> find_model_dirs(const fs::path& root, const std::vector<std::string>& graphs) {
    std::vector<fs::path> dirs;
    scan_model_dirs(root, graphs, &dirs, nullptr);
    std::sort(dirs.begin(), dirs.end());
    dirs.erase(std::unique(dirs.begin(), dirs.end()), dirs.end());
    return dirs;
}

std::string refused_model_dirs(const fs::path& root, const std::vector<std::string>& graphs) {
    std::vector<std::pair<fs::path, std::string>> refused;
    scan_model_dirs(root, graphs, nullptr, &refused);
    std::sort(refused.begin(), refused.end());
    refused.erase(std::unique(refused.begin(), refused.end()), refused.end());
    std::string out;
    for (const auto& [dir, problem] : refused) {
        out += "\n  " + utils::path_to_utf8(dir) + ": " + problem;
    }
    return out;
}

Layout require_layout(const std::string& model_name, ModelType type, const fs::path& dir,
                      const std::vector<std::string>& graphs) {
    const Layout wanted = type == ModelType::IMAGE_CLASSIFICATION ? Layout::Image : Layout::Text;
    const Inspection inspected = inspect_layout(dir, graphs);
    const Layout layout = inspected.layout;
    if (layout != wanted) {
        const std::string found =
            layout == Layout::None
                ? "has no servable classifier layout" +
                      (inspected.problem.empty() ? std::string() : ": " + inspected.problem)
                : std::string("has the ") + layout_name(layout) + " layout";
        throw NonRetryableLoadException(
            "Model '" + model_name + "' is labeled " + model_type_to_string(type) +
            " and needs the " + layout_name(wanted) + " layout, but '" +
            utils::path_to_utf8(dir) + "' " + found);
    }
    return layout;
}

json wrong_route_error(Layout loaded) {
    switch (loaded) {
        case Layout::Image:
            return invalid_request("This is an image-classification model; POST /v1/images/classify");
        case Layout::Text:
            return invalid_request("This is a text-classification model; POST /v1/classify");
        default:
            return invalid_request("No classification model is loaded on this backend");
    }
}

bool is_missing_image_route(const json& response) {
    if (!response.contains("error") || !response["error"].is_object()) return false;
    const json& err = response["error"];
    const json* details = err.contains("details") && err["details"].is_object()
                              ? &err["details"]
                              : nullptr;
    int status = 0;
    if (err.contains("status_code") && err["status_code"].is_number_integer()) {
        status = err["status_code"].get<int>();
    } else if (details && details->contains("status_code") &&
               (*details)["status_code"].is_number_integer()) {
        status = (*details)["status_code"].get<int>();
    }
    if (status != 404) return false;
    if (!details || !details->contains("response")) return true;
    const json& body = (*details)["response"];
    return body.is_null() || (body.is_string() && body.get<std::string>().empty());
}

std::string missing_image_route_message(const std::string& binary,
                                        const std::string& requirement) {
    return binary + " has no POST /classify/image route, so it cannot serve "
           "image-classification models; " + requirement;
}

json missing_image_route_error(const std::string& binary, const std::string& requirement) {
    return json{
        {"error", {
            {"message", missing_image_route_message(binary, requirement)},
            {"type", ErrorType::BACKEND_ERROR},
            {"code", "image_classification_unsupported"},
            {"status_code", 502},
        }}
    };
}

std::vector<utils::MultipartField> image_classify_fields(const json& params,
                                                         std::string image_bytes) {
    int top_k = 5;
    if (params.contains("top_k") && params["top_k"].is_number_integer()) {
        top_k = params["top_k"].get<int>();
    }
    std::vector<utils::MultipartField> fields;
    std::string mime = image_classify::sniff_image_mime(image_bytes);
    if (mime.empty()) mime = "application/octet-stream";
    fields.push_back({"image", std::move(image_bytes), "image", std::move(mime)});
    fields.push_back({"top_k", std::to_string(top_k), "", ""});
    return fields;
}

json normalize_backend_error(json response) {
    if (!response.contains("error") || !response["error"].is_object()) return response;
    json& err = response["error"];
    if (!err.contains("details") || !err["details"].is_object()) return response;
    const json& details = err["details"];
    if (!details.contains("status_code") || !details["status_code"].is_number_integer()) {
        return response;
    }
    const int status = details["status_code"].get<int>();
    err["status_code"] = status;
    if (status >= 400 && status < 500) err["type"] = "invalid_request_error";
    if (details.contains("response")) {
        const json& body = details["response"];
        if (body.is_object() && body.contains("error")) {
            const json& backend_error = body["error"];
            if (backend_error.is_string()) {
                err["message"] = backend_error;
            } else if (backend_error.is_object()) {
                if (backend_error.contains("message") && backend_error["message"].is_string()) {
                    err["message"] = backend_error["message"];
                }
                if (backend_error.contains("code") && backend_error["code"].is_string()) {
                    err["code"] = backend_error["code"];
                }
            }
        } else if (body.is_string() && !body.get<std::string>().empty()) {
            err["message"] = body;
        }
    }
    return response;
}

}  // namespace classifier
}  // namespace backends
}  // namespace lemon
