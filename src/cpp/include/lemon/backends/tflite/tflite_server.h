#pragma once

#include "lemon/backends/backend_registry.h"

#include "lemon/wrapped_server.h"
#include "lemon/server_capabilities.h"
#include "lemon/backends/backend_utils.h"
#include <string>

namespace lemon {
namespace backends {

// Runs an exported TFLite model as a tflite-server subprocess. v1 serves text
// classification (/v1/classify): input text -> {label: score}, CPU EP. The
// server is generic; embeddings/reranking are future capabilities on the same
// backend (issue #2592).
class TfliteServer : public WrappedServer, public IClassificationServer {
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

private:
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
