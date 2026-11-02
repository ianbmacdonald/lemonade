// Model-directory layout detection for the classification backends: which
// directories are a text classifier, an image classifier, or neither.

#include "lemon/backends/classifier_backend.h"

#include "lemon/backends/backend_descriptor_registry.h"
#include "lemon/backends/backend_ops.h"
#include "lemon/backends/executorch/executorch_server.h"
#include "lemon/backends/onnxruntime/onnxruntime_server.h"
#include "lemon/error_types.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
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
