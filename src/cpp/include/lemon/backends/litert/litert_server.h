#pragma once

#include "lemon/backends/backend_registry.h"

#include "lemon/wrapped_server.h"
#include "lemon/server_capabilities.h"
#include "lemon/backends/backend_utils.h"
#include <string>

namespace lemon {
namespace backends {

// Runs a `.litertlm` model as a litert-lm-server subprocess. The subprocess is
// an OpenAI-compatible chat/completions server (Google LiteRT-LM runtime, CPU
// backend), so serving the existing completion capability needs nothing beyond
// forwarding requests to it. WrappedServer already inherits ICompletionServer;
// only chat_completion/completion are overridden here.
class LiteRtServer : public WrappedServer {
public:
    static InstallParams get_install_params(const std::string& backend, const std::string& version);

    explicit LiteRtServer(const std::string& log_level,
                          ModelManager* model_manager,
                          BackendManager* backend_manager);

    ~LiteRtServer() override;

    void load(const std::string& model_name,
              const ModelInfo& model_info,
              const RecipeOptions& options,
              bool do_not_upgrade = false) override;

    void unload() override;

    // ICompletionServer
    json chat_completion(const json& request) override;
    json completion(const json& request) override;
};

namespace litert {
// Factory for the litert backend (constructs the server class — lemond only).
std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
}  // namespace litert
}  // namespace backends
}  // namespace lemon
