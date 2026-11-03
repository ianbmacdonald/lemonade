// Model-directory layout detection for the classification backends: which
// directories are a text classifier, an image classifier, or neither.

#include "lemon/backends/classifier_backend.h"

#include "lemon/backends/backend_descriptor_registry.h"
#include "lemon/backends/backend_ops.h"
#include "lemon/backends/executorch/executorch_server.h"
#include "lemon/backends/onnxruntime/onnxruntime_server.h"
#include "lemon/backends/tflite/tflite_server.h"
#include "lemon/error_types.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace classifier = lemon::backends::classifier;
using classifier::Layout;

namespace {

const std::vector<std::string> kTflite = {"model.tflite"};
const std::vector<std::string> kExecutorch = {"model.pte"};
const std::vector<std::string> kOnnxRuntime = {"model.onnx", "model.ort"};

int failures = 0;

Layout tflite_layout(const fs::path& dir) {
    return classifier::detect_layout(dir, kTflite);
}

void check(const std::string& what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

void write(const fs::path& p, const std::string& content) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    out << content;
}

const char* kImageManifest =
    R"({"task":"image-classification","labels_file":"labels.txt"})";

void make_text(const fs::path& dir) {
    write(dir / "model.tflite", "x");
    write(dir / "tokenizer.json", "{}");
    write(dir / "config.json", "{}");
}

void make_image(const fs::path& dir, const std::string& graph = "model.tflite") {
    write(dir / graph, "x");
    write(dir / "manifest.json", kImageManifest);
    write(dir / "labels.txt", "background\ncat\n");
}

void make_text_with(const fs::path& dir, const std::string& graph) {
    write(dir / graph, "x");
    write(dir / "tokenizer.json", "{}");
    write(dir / "config.json", "{}");
}

// What the recipe's resolver picks for a local import, "" when nothing or
// several models are found: the same search load() runs.
fs::path resolved(const lemon::backends::BackendOps* ops, const fs::path& dir) {
    return fs::path(ops->find_imported_checkpoint(dir.string()));
}

bool layout_refused(const std::string& model, lemon::ModelType type, const fs::path& dir,
                    const std::vector<std::string>& graphs) {
    try {
        classifier::require_layout(model, type, dir, graphs);
    } catch (const lemon::NonRetryableLoadException&) {
        return true;
    }
    return false;
}

std::string load_error(const std::string& model, lemon::ModelType type, const fs::path& dir,
                       const std::vector<std::string>& graphs) {
    try {
        classifier::require_layout(model, type, dir, graphs);
    } catch (const lemon::NonRetryableLoadException& e) {
        return e.what();
    }
    return "";
}

void make_text_with_manifest(const fs::path& dir, const std::string& graph,
                             const std::string& manifest) {
    make_text_with(dir, graph);
    write(dir / "manifest.json", manifest);
}

// Not a layout: the dir must be None, resolve to nothing, and load under
// neither label, so an explicit image declaration never becomes a text model.
void check_unservable(const std::string& what, const fs::path& dir,
                      const std::vector<std::string>& graphs,
                      const lemon::backends::BackendOps* ops) {
    using lemon::ModelType;
    check(what + " is None", classifier::detect_layout(dir, graphs) == Layout::None);
    check(what + " does not resolve", resolved(ops, dir).empty());
    check(what + " loads under neither label",
          layout_refused("m", ModelType::CLASSIFICATION, dir, graphs) &&
              layout_refused("m", ModelType::IMAGE_CLASSIFICATION, dir, graphs));
}

// Every way a manifest makes a dir not a model gives its own reason in the load
// error, so a model that disappears from the list can be diagnosed.
void check_reasons(const fs::path& base, const std::string& recipe, const std::string& graph,
                   const std::vector<std::string>& graphs,
                   const lemon::backends::BackendOps* ops) {
    using lemon::ModelType;
    auto reason = [&](const std::string& what, const fs::path& dir, const std::string& text) {
        const std::string error = load_error("m", ModelType::CLASSIFICATION, dir, graphs);
        check(recipe + ": " + what + " gives the reason '" + text + "'",
              error.find(text) != std::string::npos);
    };
    reason("a climbing labels_file", base / "bad_labels_0", "climbs out of the model directory");
    reason("a non-string labels_file", base / "bad_labels_3", "labels_file is not a string");
    reason("a missing labels file", base / "bad_labels_4", "the labels file 'absent.txt'");
    reason("an oversized manifest", base / "huge_text", "manifest.json is larger than 1 MiB");

    make_text_with_manifest(base / "array_manifest", graph, "[1, 2]");
    check_unservable(recipe + ": a manifest that is a JSON array, with text files",
                     base / "array_manifest", graphs, ops);
    reason("a JSON array manifest", base / "array_manifest", "manifest.json is not a JSON object");

    make_text_with(base / "manifest_dir", graph);
    fs::create_directories(base / "manifest_dir" / "manifest.json");
    check_unservable(recipe + ": a manifest.json directory, with text files",
                     base / "manifest_dir", graphs, ops);
    reason("a manifest.json directory", base / "manifest_dir",
           "manifest.json is not a regular file");

#ifndef _WIN32
    reason("a manifest linked outside", base / "manifest_link_escape",
           "manifest.json resolves outside the model directory");

    make_text_with(base / "manifest_link_dir", graph);
    fs::create_directories(base / "manifest_link_dir" / "sub");
    fs::create_symlink("sub", base / "manifest_link_dir" / "manifest.json");
    check_unservable(recipe + ": a manifest.json linked to a directory, with text files",
                     base / "manifest_link_dir", graphs, ops);
    reason("a manifest.json linked to a directory", base / "manifest_link_dir",
           "manifest.json is not a regular file");

    make_text_with(base / "manifest_dangling", graph);
    fs::create_symlink("gone.json", base / "manifest_dangling" / "manifest.json");
    check_unservable(recipe + ": a dangling manifest.json link, with text files",
                     base / "manifest_dangling", graphs, ops);
    reason("a dangling manifest link", base / "manifest_dangling",
           "manifest.json cannot be resolved");
#endif
}

// A refusal reaches the log and the client's error body, so it is bounded and
// carries no raw control characters; a graph with no layout files is not one.
void check_refusal_bounds(const fs::path& root, const std::string& recipe,
                          const std::string& graph, const std::vector<std::string>& graphs) {
    const fs::path base = root / ("bounds-" + recipe);

    const std::string forged = "../\n[Warn] forged " + std::string(500, 'a');
    make_text_with_manifest(base / "one" / "forged", graph,
                            nlohmann::json{{"task", "image-classification"},
                                           {"labels_file", forged}}
                                .dump());
    const std::string one = classifier::refused_model_dirs(base / "one", graphs);
    check(recipe + ": a refusal escapes control characters",
          one.find('\n', 1) == std::string::npos && one.find("\\x0a") != std::string::npos);
    check(recipe + ": a refusal cuts an echoed labels_file", one.size() < 400);

    for (int i = 0; i < 7; ++i) {
        make_text_with_manifest(base / "many" / ("m" + std::to_string(i)), graph, "[]");
    }
    const std::string many = classifier::refused_model_dirs(base / "many", graphs);
    size_t lines = 0;
    for (const char c : many) lines += c == '\n';
    check(recipe + ": a refusal lists at most 5 dirs and says how many more",
          lines == 6 && many.find("and 2 more") != std::string::npos);

    // A cut must not split a character, and bytes that are not UTF-8 are
    // escaped: the reason goes into a JSON error body, whose dump() throws on
    // invalid UTF-8.
    auto dumps = [](const std::string& text) {
        try {
            (void)nlohmann::json(text).dump();
            return true;
        } catch (const std::exception&) {
            return false;
        }
    };
    std::string accented = "../";
    for (int i = 0; i < 100; ++i) accented += "\xc3\xa9";
    make_text_with_manifest(base / "utf8" / "cut", graph,
                            nlohmann::json{{"task", "image-classification"},
                                           {"labels_file", accented}}
                                .dump());
    const std::string cut = classifier::refused_model_dirs(base / "utf8", graphs);
    check(recipe + ": a cut labels_file stays valid UTF-8",
          dumps(cut) && cut.find("climbs out") != std::string::npos);
    check(recipe + ": a load error with a cut labels_file stays valid UTF-8",
          dumps(load_error("m", lemon::ModelType::CLASSIFICATION, base / "utf8" / "cut", graphs)));

#ifndef _WIN32
    const fs::path raw = base / "raw" / std::string("bad\xff\xfe");
    make_text_with_manifest(raw, graph, "[]");
    const std::string escaped = classifier::refused_model_dirs(base / "raw", graphs);
    check(recipe + ": a directory name that is not UTF-8 is escaped",
          dumps(escaped) && escaped.find("bad\\xff\\xfe") != std::string::npos);
#endif

    fs::path deep = base / "deep";
    for (int i = 0; i < 12; ++i) deep /= std::string(200, static_cast<char>('a' + i));
    make_text_with_manifest(deep, graph, "[]");
    const std::string long_one = classifier::refused_model_dirs(base / "deep", graphs);
    check(recipe + ": a refusal over 2 KiB still names the dir and its reason",
          long_one.find("not a JSON object") != std::string::npos &&
              long_one.find("more") == std::string::npos && long_one.size() <= 2048);

    const std::string named =
        load_error("m\n[Warn] forged", lemon::ModelType::IMAGE_CLASSIFICATION, base / "one" / "forged",
                   graphs);
    check(recipe + ": the load error escapes the model name",
          named.find('\n') == std::string::npos && named.find("m\\x0a") != std::string::npos);

    write(base / "stray" / "partial" / graph, "x");
    check(recipe + ": a graph with no layout files is not listed as a refusal",
          classifier::refused_model_dirs(base / "stray", graphs).empty());
}

// Manifest and labels confinement, and image declarations that fail a check
// with tokenizer.json and config.json present, for one engine's graph names.
void check_declarations(const fs::path& root, const std::string& recipe, const std::string& graph,
                        const std::vector<std::string>& graphs,
                        const lemon::backends::BackendOps* ops) {
    const fs::path base = root / ("decl-" + recipe);
    write(base / "outside.txt", "background\ncat\n");

    const std::pair<const char*, std::string> bad_labels[] = {
        {"a parent-traversing labels_file", R"({"task":"image-classification","labels_file":"../outside.txt"})"},
        {"an absolute labels_file",
         R"({"task":"image-classification","labels_file":")" + (base / "outside.txt").generic_string() + R"("})"},
        {"an empty labels_file", R"({"task":"image-classification","labels_file":""})"},
        {"a non-string labels_file", R"({"task":"image-classification","labels_file":5})"},
        {"a missing labels file", R"({"task":"image-classification","labels_file":"absent.txt"})"},
    };
    int n = 0;
    for (const auto& [name, manifest] : bad_labels) {
        const fs::path dir = base / ("bad_labels_" + std::to_string(n++));
        make_text_with_manifest(dir, graph, manifest);
        check_unservable(recipe + ": image manifest with " + name + " and text files", dir,
                         graphs, ops);
    }

    make_text_with_manifest(base / "huge_text", graph,
                            std::string(R"({"task":"image-classification","pad":")") +
                                std::string(2 * 1024 * 1024, 'a') + "\"}");
    check_unservable(recipe + ": 2 MiB image manifest with text files", base / "huge_text",
                     graphs, ops);

    make_text_with_manifest(base / "text_manifest", graph, R"({"task":"text-classification"})");
    check(recipe + ": a text manifest with text files is still Text",
          classifier::detect_layout(base / "text_manifest", graphs) == Layout::Text);

#ifndef _WIN32
    write(base / "elsewhere" / "manifest.json", kImageManifest);
    const fs::path linked = base / "manifest_link_escape";
    write(linked / graph, "x");
    write(linked / "labels.txt", "background\ncat\n");
    fs::create_symlink(base / "elsewhere" / "manifest.json", linked / "manifest.json");
    check_unservable(recipe + ": manifest linked outside a plain model dir", linked, graphs, ops);

    const fs::path linked_text = base / "manifest_link_escape_text";
    make_text_with(linked_text, graph);
    fs::create_symlink(base / "elsewhere" / "manifest.json", linked_text / "manifest.json");
    check_unservable(recipe + ": manifest linked outside a text model dir", linked_text, graphs,
                     ops);

    const fs::path repo = base / "hub" / "models--org--img";
    write(repo / "blobs" / "m1", kImageManifest);
    write(repo / "blobs" / "l1", "background\ncat\n");
    const fs::path snap = repo / "snapshots" / "6e091d820cbe8f22eeb604d136403eca290b8c1e";
    write(snap / graph, "x");
    fs::create_symlink("../../blobs/m1", snap / "manifest.json");
    fs::create_symlink("../../blobs/l1", snap / "labels.txt");
    check(recipe + ": manifest and labels linked to their repository's blobs is Image",
          classifier::detect_layout(snap, graphs) == Layout::Image);
    check(recipe + ": a snapshot with blob links resolves", resolved(ops, snap) == snap);

    const fs::path snap_out = repo / "snapshots" / "aaaa";
    write(snap_out / graph, "x");
    fs::create_symlink("../../blobs/l1", snap_out / "labels.txt");
    fs::create_symlink(base / "elsewhere" / "manifest.json", snap_out / "manifest.json");
    check_unservable(recipe + ": manifest linked outside its repository", snap_out, graphs, ops);

    const fs::path ms = base / "hub" / "modelscope--models--org--img" / "snapshots" / "bbbb";
    write(ms.parent_path().parent_path() / "blobs" / "l1", "a\n");
    make_image(ms, graph);
    fs::remove(ms / "labels.txt");
    fs::create_symlink("../../blobs/l1", ms / "labels.txt");
    check(recipe + ": a ModelScope snapshot linking its labels to a blob is Image",
          classifier::detect_layout(ms, graphs) == Layout::Image);

    const fs::path fake_snap = base / "plain" / "snapshots" / "m";
    write(base / "plain" / "other.txt", "a\n");
    make_image(fake_snap, graph);
    fs::remove(fake_snap / "labels.txt");
    fs::create_symlink(base / "plain" / "other.txt", fake_snap / "labels.txt");
    check_unservable(recipe + ": labels linked beside a dir that is only named snapshots",
                     fake_snap, graphs, ops);
#endif

    using lemon::ModelType;
    make_text_with(base / "text", graph);
    make_image(base / "image", graph);
    const std::string to_image = load_error("m", ModelType::IMAGE_CLASSIFICATION, base / "text", graphs);
    check(recipe + ": the mismatch error reads 'needs the image-classifier layout'",
          to_image.find("needs the image-classifier layout") != std::string::npos &&
              to_image.find("has the text-classifier layout") != std::string::npos);
    const std::string to_text = load_error("m", ModelType::CLASSIFICATION, base / "image", graphs);
    check(recipe + ": the mismatch error reads 'needs the text-classifier layout'",
          to_text.find("needs the text-classifier layout") != std::string::npos &&
              to_text.find("has the image-classifier layout") != std::string::npos);
    const std::string to_none =
        load_error("m", ModelType::CLASSIFICATION, base / "bad_labels_0", graphs);
    check(recipe + ": a dir with no layout is named as such",
          to_none.find("has no servable classifier layout") != std::string::npos);

    check_reasons(base, recipe, graph, graphs, ops);
    check_refusal_bounds(root, recipe, graph, graphs);
}

void check_engine(const fs::path& root, const std::string& recipe, const std::string& graph,
                  const std::vector<std::string>& graphs, const lemon::backends::BackendOps* ops) {
    const fs::path base = root / recipe;
    using lemon::ModelType;

    make_text_with(base / "text", graph);
    check(recipe + ": text layout is Text",
          classifier::detect_layout(base / "text", graphs) == Layout::Text);
    check(recipe + ": a text model resolves", resolved(ops, base / "text") == base / "text");

    make_image(base / "image", graph);
    check(recipe + ": image manifest + labels is Image",
          classifier::detect_layout(base / "image", graphs) == Layout::Image);
    check(recipe + ": an image model resolves", resolved(ops, base / "image") == base / "image");

    write(base / "image_no_labels" / graph, "x");
    write(base / "image_no_labels" / "manifest.json", kImageManifest);
    check(recipe + ": image manifest with its labels file missing is None",
          classifier::detect_layout(base / "image_no_labels", graphs) == Layout::None);
    check(recipe + ": an image model without labels does not resolve",
          resolved(ops, base / "image_no_labels").empty());

    write(base / "image_escape" / graph, "x");
    write(base / "image_escape" / "manifest.json",
          R"({"task":"image-classification","labels_file":"../text/config.json"})");
    check(recipe + ": labels_file outside the model dir is None",
          classifier::detect_layout(base / "image_escape", graphs) == Layout::None);
    check(recipe + ": an escaping labels file does not resolve",
          resolved(ops, base / "image_escape").empty());

#ifndef _WIN32
    const fs::path linked = base / "image_link_escape";
    make_image(linked, graph);
    fs::remove(linked / "labels.txt");
    fs::create_symlink(root / "text" / "config.json", linked / "labels.txt");
    check(recipe + ": labels file linked outside a plain model dir is None",
          classifier::detect_layout(linked, graphs) == Layout::None);
#endif

    make_text_with(base / "text_image_manifest", graph);
    write(base / "text_image_manifest" / "manifest.json", kImageManifest);
    write(base / "text_image_manifest" / "labels.txt", "a\n");
    check(recipe + ": an image manifest makes a dir with a tokenizer Image",
          classifier::detect_layout(base / "text_image_manifest", graphs) == Layout::Image);

    make_image(base / "other_engine", "model.tflite");
    check(recipe + ": another engine's graph is None",
          classifier::detect_layout(base / "other_engine", graphs) == Layout::None);
    write(base / "none" / "labels.txt", "a\n");
    check(recipe + ": a dir without a graph is None",
          classifier::detect_layout(base / "none", graphs) == Layout::None);

    check(recipe + ": an image label on a text dir is refused",
          layout_refused("m", ModelType::IMAGE_CLASSIFICATION, base / "text", graphs));
    check(recipe + ": a text label on an image dir is refused",
          layout_refused("m", ModelType::CLASSIFICATION, base / "image", graphs));
    check(recipe + ": a matching label is accepted",
          !layout_refused("m", ModelType::IMAGE_CLASSIFICATION, base / "image", graphs) &&
              !layout_refused("m", ModelType::CLASSIFICATION, base / "text", graphs));

    const fs::path two = base / "two";
    make_text_with(two / "a", graph);
    make_image(two / "b", graph);
    check(recipe + ": a text and an image model under one root do not resolve",
          resolved(ops, two).empty());

    check(recipe + " serves image-classification",
          lemon::backends::backend_serves_mode(recipe, ModelType::IMAGE_CLASSIFICATION));
    check(recipe + " accepts the image-classification label",
          lemon::backends::illegal_deployment_labels({"image-classification"}, recipe).empty());
    check(recipe + " still defaults to classification",
          lemon::backends::default_mode_for(recipe) == "classification");
}

}  // namespace

int main() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root =
        fs::temp_directory_path() / ("lemon_classifier_layout_" + std::to_string(stamp));
    fs::create_directories(root);

    make_text(root / "text");
    check("text layout is Text", tflite_layout(root / "text") == Layout::Text);

    make_text(root / "text_manifest");
    write(root / "text_manifest" / "manifest.json", R"({"task":"text-classification"})");
    check("text layout with a text manifest is Text",
          tflite_layout(root / "text_manifest") == Layout::Text);

    make_image(root / "image");
    check("image manifest + labels is Image",
          tflite_layout(root / "image") == Layout::Image);

    write(root / "image_default_labels" / "model.tflite", "x");
    write(root / "image_default_labels" / "manifest.json", R"({"task":"image-classification"})");
    write(root / "image_default_labels" / "labels.txt", "a\n");
    check("labels_file defaults to labels.txt",
          tflite_layout(root / "image_default_labels") == Layout::Image);

    write(root / "image_no_labels" / "model.tflite", "x");
    write(root / "image_no_labels" / "manifest.json", kImageManifest);
    check("image manifest with its labels file missing is None",
          tflite_layout(root / "image_no_labels") == Layout::None);

    write(root / "image_escape" / "model.tflite", "x");
    write(root / "image_escape" / "manifest.json",
          R"({"task":"image-classification","labels_file":"../text/config.json"})");
    check("labels_file outside the model dir is None",
          tflite_layout(root / "image_escape") == Layout::None);

    write(root / "text_no_tokenizer" / "model.tflite", "x");
    write(root / "text_no_tokenizer" / "config.json", "{}");
    check("text layout without a tokenizer is None",
          tflite_layout(root / "text_no_tokenizer") == Layout::None);

    write(root / "corrupt" / "model.tflite", "x");
    write(root / "corrupt" / "manifest.json", "{not json");
    write(root / "corrupt" / "labels.txt", "a\n");
    check("corrupt manifest is None without throwing",
          tflite_layout(root / "corrupt") == Layout::None);

    write(root / "huge" / "model.tflite", "x");
    write(root / "huge" / "labels.txt", "a\n");
    write(root / "huge" / "manifest.json",
          std::string(R"({"task":"image-classification","pad":")") +
              std::string(2 * 1024 * 1024, 'a') + "\"}");
    check("2 MiB manifest is None without throwing",
          tflite_layout(root / "huge") == Layout::None);

    check("missing directory is None",
          tflite_layout(root / "does_not_exist") == Layout::None);

#ifndef _WIN32
    {
        const fs::path repo = root / "hub" / "models--org--img";
        write(repo / "blobs" / "4f1a", "background\ncat\n");
        const fs::path snap = repo / "snapshots" / "6e091d820cbe8f22eeb604d136403eca290b8c1e";
        write(snap / "model.tflite", "x");
        write(snap / "manifest.json", kImageManifest);
        fs::create_symlink("../../blobs/4f1a", snap / "labels.txt");
        check("labels file linked to its repository's blob is Image",
              tflite_layout(snap) == Layout::Image);

        const fs::path escape = repo / "snapshots" / "aaaa";
        write(escape / "model.tflite", "x");
        write(escape / "manifest.json", kImageManifest);
        fs::create_symlink(root / "text" / "config.json", escape / "labels.txt");
        check("labels file linked outside its repository is None",
              tflite_layout(escape) == Layout::None);

        const fs::path plain = root / "image_link_escape";
        write(plain / "model.tflite", "x");
        write(plain / "manifest.json", kImageManifest);
        fs::create_symlink(root / "text" / "config.json", plain / "labels.txt");
        check("labels file linked outside a plain model dir is None",
              tflite_layout(plain) == Layout::None);
    }
#endif

    const fs::path two = root / "two";
    make_text(two / "a");
    make_image(two / "b");
    check("two complete dirs are both found", classifier::find_model_dirs(two, kTflite).size() == 2);
    check("one complete dir is found", classifier::find_model_dirs(root / "image", kTflite).size() == 1);

    check_declarations(root, "tflite", "model.tflite", kTflite, lemon::backends::tflite::ops());
    check_declarations(root, "executorch", "model.pte", kExecutorch,
                       lemon::backends::executorch::ops());
    check_declarations(root, "onnxruntime", "model.ort", kOnnxRuntime,
                       lemon::backends::onnxruntime::ops());

    {
        const std::string none = classifier::wrong_route_error(Layout::None)["error"]["message"];
        check("wrong_route_error(None) says nothing is loaded",
              none.find("No classification model is loaded") != std::string::npos);

        auto refuses_both_routes = [](lemon::IClassificationServer& text,
                                      lemon::IImageClassificationServer& image) {
            auto said = [](const nlohmann::json& r) {
                return r.contains("error") && r["error"].value("message", "").find(
                                                  "No classification model is loaded") !=
                                                  std::string::npos;
            };
            return said(text.classify({{"input", "hello"}})) &&
                   said(image.classify_image(nlohmann::json::object(), "x"));
        };
        lemon::backends::TfliteServer tflite("info", nullptr, nullptr);
        lemon::backends::ExecutorchServer executorch("info", nullptr, nullptr);
        lemon::backends::OnnxRuntimeServer onnxruntime("info", nullptr, nullptr);
        check("tflite with nothing loaded refuses both routes",
              refuses_both_routes(tflite, tflite));
        check("executorch with nothing loaded refuses both routes",
              refuses_both_routes(executorch, executorch));
        check("onnxruntime with nothing loaded refuses both routes",
              refuses_both_routes(onnxruntime, onnxruntime));
    }

    check_engine(root, "executorch", "model.pte", kExecutorch, lemon::backends::executorch::ops());
    check_engine(root, "onnxruntime", "model.ort", kOnnxRuntime, lemon::backends::onnxruntime::ops());

    const fs::path both = root / "onnxruntime" / "image_both_graphs";
    make_image(both, "model.onnx");
    write(both / "model.ort", "x");
    check("onnxruntime: an image dir with model.onnx and model.ort is one model",
          classifier::find_model_dirs(both, kOnnxRuntime).size() == 1 &&
              resolved(lemon::backends::onnxruntime::ops(), both) == both);

    std::error_code ec;
    fs::remove_all(root, ec);

    if (failures == 0) {
        std::printf("\nAll classifier layout checks passed.\n");
        return 0;
    }
    std::printf("\n%d classifier layout check(s) failed.\n", failures);
    return 1;
}
