#pragma once

#include "lemon/backends/backend_registry.h"

#include "lemon/wrapped_server.h"
#include "lemon/server_capabilities.h"
#include "lemon/backends/backend_utils.h"
#include <string>

namespace lemon {
namespace backends {

// Runs an exported TFLite model as a tflite-server subprocess on the CPU. A text
// model serves /v1/classify (text -> {label: score}); an image model serves
// /v1/images/classify (image -> ranked labels), which needs tflite-server >= 0.2.0.
class TfliteServer : public WrappedServer,
                     public IClassificationServer,
                     public IImageClassificationServer {
public:
    static InstallParams get_install_params(const std::string& backend, const std::string& version);

    explicit TfliteServer(const std::string& log_level,
                               ModelManager* model_manager,
                               BackendManager* backend_manager);

    ~TfliteServer() override;

    void load(const std::string& model_name,
              const ModelInfo& model_info,
              const RecipeOptions& options,
              bool do_not_upgrade = false) override;

    void unload() override;

    // IClassificationServer
    json classify(const json& request) override;

    // IImageClassificationServer
    json classify_image(const json& params, std::string image_bytes) override;

private:
    // The /health "task" the loaded subprocess must report.
    std::string expected_task_;

    // Forward a classify request to the subprocess and normalize its response.
    json forward_classify(const std::string& text, const json& params);
};

namespace tflite {
// Factory for the tflite backend (constructs the server class — lemond only).
std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
constexpr uint32_t capabilities() { return capability_mask_of<TfliteServer>(); }
}  // namespace tflite
}  // namespace backends
}  // namespace lemon
