#pragma once

#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/server_capabilities.h"
#include "lemon/wrapped_server.h"
#include <string>

namespace lemon {
namespace backends {

class ExecutorchServer : public WrappedServer, public IClassificationServer {
public:
    static InstallParams get_install_params(const std::string& backend, const std::string& version);

    explicit ExecutorchServer(const std::string& log_level,
                              ModelManager* model_manager,
                              BackendManager* backend_manager);

    ~ExecutorchServer() override;

    void load(const std::string& model_name,
              const ModelInfo& model_info,
              const RecipeOptions& options,
              bool do_not_upgrade = false) override;

    void unload() override;

    json classify(const json& request) override;

private:
    json forward_classify(const std::string& text, const json& params);
};

namespace executorch {
std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
constexpr uint32_t capabilities() { return capability_mask_of<ExecutorchServer>(); }
}  // namespace executorch
}  // namespace backends
}  // namespace lemon
