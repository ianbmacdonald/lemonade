#include "lemon/backends/classifier_backend.h"

#include "lemon/backends/hf_cache_util.h"
#include "lemon/error_types.h"
#include "lemon/image_classify_request.h"
#include "lemon/utils/path_utils.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace lemon {
namespace backends {
namespace classifier {

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
    const json doc = json::parse(text, nullptr, false);
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

// A Hugging Face snapshot links each file into its repository's blobs/, so a
// labels file may resolve anywhere in that repository; outside a snapshot it
// must resolve inside the model dir. Checked on the resolved target, since the
// lexical check in read_image_manifest does not see through symlinks.
bool labels_file_confined(const fs::path& dir, const std::string& labels_file) {
    std::error_code ec;
    const fs::path abs_dir = fs::absolute(dir, ec);
    if (ec) return false;
    fs::path root = abs_dir;
    for (fs::path p = abs_dir; p.has_relative_path(); p = p.parent_path()) {
        if (p.filename() == "snapshots") {
            root = p.parent_path();
            break;
        }
    }
    const fs::path base = fs::canonical(root, ec);
    if (ec) return false;
    const fs::path target = fs::canonical(abs_dir / fs::path(labels_file), ec);
    if (ec) return false;
    return std::mismatch(base.begin(), base.end(), target.begin(), target.end()).first == base.end();
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
        if (read_image_manifest(dir, labels_file)) {
            return fs::is_regular_file(dir / fs::path(labels_file), ec) && !ec &&
                           labels_file_confined(dir, labels_file)
                       ? Layout::Image
                       : Layout::None;
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
        throw NonRetryableLoadException(
            "Model '" + model_name + "' is labeled " + model_type_to_string(type) +
            " and needs a " + layout_name(wanted) + " layout, but '" +
            utils::path_to_utf8(dir) + "' is a " + layout_name(layout) + " layout");
    }
    return layout;
}

json wrong_route_error(Layout loaded) {
    return loaded == Layout::Image
               ? invalid_request("This is an image-classification model; POST /v1/images/classify")
               : invalid_request("This is a text-classification model; POST /v1/classify");
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
