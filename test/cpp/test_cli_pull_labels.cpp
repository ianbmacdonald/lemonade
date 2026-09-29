// A tflite model's default mode is text classification, so an image classifier
// registered with `lemonade pull` needs `--label image-classification`; if the
// CLI rejects that label the model can only be registered as a text classifier.
// The same holds for every mode a backend serves beyond its default.

#include "lemon/backends/backend_descriptor_registry.h"
#include "lemon/model_types.h"
#include "lemon_cli/pull_labels.h"

#include <algorithm>
#include <cstdio>
#include <string>

using lemon::ModelType;

namespace {

int failures = 0;

void check(const std::string& what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

bool cli_accepts(const std::string& label) {
    const auto& labels = lemon_cli::pull_labels();
    return std::find(labels.begin(), labels.end(), label) != labels.end();
}

} // namespace

int main() {
    ModelType mode = ModelType::LLM;
    check("image-classification names the IMAGE_CLASSIFICATION mode",
          lemon::deployment_mode_of("image-classification", mode) &&
              mode == ModelType::IMAGE_CLASSIFICATION);
    check("tflite serves image-classification",
          lemon::backends::backend_serves_mode("tflite", ModelType::IMAGE_CLASSIFICATION));
    check("tflite defaults to a different mode, so the label is required",
          lemon::backends::default_mode_for("tflite") != "image-classification");
    check("lemonade pull --label accepts image-classification",
          cli_accepts("image-classification"));

    for (const auto* descriptor : lemon::backends::all_descriptors()) {
        const auto& modes = descriptor->supported_modes;
        for (std::size_t i = 1; i < modes.size(); ++i) {
            check("lemonade pull --label accepts " + descriptor->recipe +
                      "'s non-default mode " + modes[i],
                  cli_accepts(modes[i]));
        }
    }

    const auto& labels = lemon_cli::pull_labels();
    check("pull labels are sorted and unique",
          std::is_sorted(labels.begin(), labels.end()) &&
              std::adjacent_find(labels.begin(), labels.end()) == labels.end());

    std::printf("%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
