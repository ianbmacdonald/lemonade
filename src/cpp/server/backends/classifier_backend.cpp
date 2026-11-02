#include "lemon/backends/classifier_backend.h"

#include "lemon/backends/hf_cache_util.h"
#include "lemon/error_types.h"
#include "lemon/image_classify_request.h"
#include "lemon/utils/path_utils.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <system_error>
#include <utility>

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

// The canonical target of dir/rel when it resolves inside dir's confinement
// root, else empty. Checked on the resolved target, since a lexical check does
// not see through symlinks.
fs::path confined_target(const fs::path& dir, const fs::path& rel) {
    std::error_code ec;
    const fs::path abs_dir = fs::absolute(dir, ec);
    if (ec) return {};
    const fs::path base = fs::canonical(confinement_root(abs_dir), ec);
    if (ec) return {};
    fs::path target = fs::canonical(abs_dir / rel, ec);
    if (ec) return {};
    if (std::mismatch(base.begin(), base.end(), target.begin(), target.end()).first != base.end()) {
        return {};
    }
    return target;
}

enum class ImageDeclaration { Absent, Valid, Invalid };

// Valid only for a confined, readable manifest declaring the image-classification
// task with a lexically in-dir labels file. A manifest that declares the task but
// fails any check is Invalid, never Absent: falling through to the text layout
// would load an explicit image model as a text classifier. A manifest that cannot
// be read safely (escaping link, oversized, not JSON) is Invalid too, since what
// it declares is unknown. Only a manifest without the image task is Absent.
ImageDeclaration read_image_manifest(const fs::path& dir, std::string& labels_file) {
    std::error_code ec;
    const fs::path manifest = dir / "manifest.json";
    const fs::file_type type = fs::symlink_status(manifest, ec).type();
    if (type == fs::file_type::not_found) return ImageDeclaration::Absent;
    if (ec) return ImageDeclaration::Invalid;

    const fs::path target = confined_target(dir, "manifest.json");
    if (target.empty()) return ImageDeclaration::Invalid;
    if (!fs::is_regular_file(target, ec) || ec) return ImageDeclaration::Absent;

    std::ifstream in(target, std::ios::binary);
    if (!in) return ImageDeclaration::Invalid;
    std::string text(kMaxManifestBytes + 1, '\0');
    in.read(&text[0], static_cast<std::streamsize>(text.size()));
    if (static_cast<std::uintmax_t>(in.gcount()) > kMaxManifestBytes) {
        return ImageDeclaration::Invalid;
    }
    text.resize(static_cast<size_t>(in.gcount()));

    const json doc = json::parse(text, nullptr, false);
    if (!doc.is_object()) return ImageDeclaration::Invalid;
    const auto task = doc.find("task");
    if (task == doc.end() || !task->is_string() ||
        task->get<std::string>() != "image-classification") {
        return ImageDeclaration::Absent;
    }
    labels_file = "labels.txt";
    const auto labels = doc.find("labels_file");
    if (labels != doc.end()) {
        if (!labels->is_string()) return ImageDeclaration::Invalid;
        labels_file = labels->get<std::string>();
    }
    // The labels file must sit in the model dir; a path that climbs out of it
    // would let a manifest point the backend anywhere on disk.
    const fs::path rel(labels_file);
    if (labels_file.empty() || rel.is_absolute() || rel.has_root_path()) {
        return ImageDeclaration::Invalid;
    }
    for (const auto& part : rel) {
        if (part == "..") return ImageDeclaration::Invalid;
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

Layout detect_layout(const fs::path& dir, const std::vector<std::string>& graphs) noexcept {
    try {
        std::error_code ec;
        if (!has_graph(dir, graphs)) return Layout::None;

        std::string labels_file;
        switch (read_image_manifest(dir, labels_file)) {
            case ImageDeclaration::Valid: {
                const fs::path labels = confined_target(dir, fs::path(labels_file));
                return !labels.empty() && fs::is_regular_file(labels, ec) && !ec
                           ? Layout::Image
                           : Layout::None;
            }
            case ImageDeclaration::Invalid:
                return Layout::None;
            case ImageDeclaration::Absent:
                break;
        }
        // The config is mandatory even when a manifest is present: the manifest
        // describes the output contract only, and the engine checks the
        // architecture in config.json against its supported input convention.
        if (fs::exists(dir / "tokenizer.json", ec) && fs::exists(dir / "config.json", ec)) {
            return Layout::Text;
        }
    } catch (...) {
    }
    return Layout::None;
}

const char* layout_name(Layout layout) noexcept {
    switch (layout) {
        case Layout::Text: return "text-classifier";
        case Layout::Image: return "image-classifier";
        default: return "unrecognized";
    }
}

std::vector<fs::path> find_model_dirs(const fs::path& root, const std::vector<std::string>& graphs) {
    std::vector<fs::path> dirs;
    std::error_code ec;
    fs::recursive_directory_iterator it(root, hf_cache::dir_options(), ec);
    if (ec) return dirs;
    for (auto end = fs::recursive_directory_iterator(); it != end; it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file(ec) || ec) continue;
        const fs::path name = it->path().filename();
        const bool is_graph = std::any_of(graphs.begin(), graphs.end(),
                                          [&name](const std::string& g) { return name == g; });
        if (is_graph && detect_layout(it->path().parent_path(), graphs) != Layout::None) {
            dirs.push_back(it->path().parent_path());
        }
    }
    std::sort(dirs.begin(), dirs.end());
    dirs.erase(std::unique(dirs.begin(), dirs.end()), dirs.end());
    return dirs;
}

Layout require_layout(const std::string& model_name, ModelType type, const fs::path& dir,
                      const std::vector<std::string>& graphs) {
    const Layout wanted = type == ModelType::IMAGE_CLASSIFICATION ? Layout::Image : Layout::Text;
    const Layout layout = detect_layout(dir, graphs);
    if (layout != wanted) {
        const std::string found = layout == Layout::None
                                      ? "has no servable classifier layout"
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

json require_image_route(json response, const std::string& engine_release) {
    if (!response.contains("error") || !response["error"].is_object()) return response;
    const json& err = response["error"];
    if (!err.contains("status_code") || err["status_code"] != 404) return response;
    response["error"] = {
        {"message", engine_release + " or newer is required to serve image-classification "
                    "models: the engine answered 404 for POST /classify/image"},
        {"type", ErrorType::BACKEND_ERROR},
        {"code", "image_classification_unsupported"},
        {"status_code", 502},
    };
    return response;
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
