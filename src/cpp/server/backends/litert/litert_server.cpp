#include "lemon/backends/litert/litert_server.h"
#include "lemon/backends/litert/litert.h"
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_ops.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/backend_manager.h"
#include "lemon/utils/custom_args.h"
#include "lemon/utils/http_client.h"
#include "lemon/utils/process_manager.h"
#include <algorithm>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include <lemon/utils/aixlog.hpp>

namespace fs = std::filesystem;
using namespace lemon::utils;

namespace lemon {
namespace backends {

namespace {
bool ends_with_ci(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin(),
                      [](char a, char b) {
                          return std::tolower(static_cast<unsigned char>(a)) ==
                                 std::tolower(static_cast<unsigned char>(b));
                      });
}
}  // namespace

// litert-lm-server is a pre-built binary on PATH (the "system" backend), so it
// has no downloadable release. install_backend() short-circuits for "system"
// and never calls this; it exists only to satisfy make_spec<T>.
InstallParams LiteRtServer::get_install_params(const std::string& backend,
                                               const std::string& version) {
    (void)backend;
    (void)version;
    throw std::runtime_error(
        "The litert backend uses a pre-installed litert-lm-server binary on PATH "
        "and has no downloadable release");
}

LiteRtServer::LiteRtServer(const std::string& log_level, ModelManager* model_manager,
                           BackendManager* backend_manager)
    : WrappedServer("litert-lm-server", log_level, model_manager, backend_manager) {
}

LiteRtServer::~LiteRtServer() {
    unload();
}

void LiteRtServer::load(const std::string& model_name,
                        const ModelInfo& model_info,
                        const RecipeOptions& options,
                        bool do_not_upgrade) {
    (void)do_not_upgrade;
    LOG(INFO, "LiteRtServer") << "Loading model: " << model_name << std::endl;
    LOG(INFO, "LiteRtServer") << "Per-model settings: " << options.to_log_string() << std::endl;

    std::string extra_args = options.get_option("litert_args");
    device_type_ = DEVICE_CPU;

    // "system" backend: nothing to install (PATH binary), returns immediately.
    backend_manager_->install_backend(litert::spec()->recipe, "system");

    std::string model_path = model_info.resolved_path();
    if (model_path.empty() || !fs::exists(path_from_utf8(model_path))) {
        throw std::runtime_error("Model file not found for checkpoint: " + model_info.checkpoint());
    }
    LOG(INFO, "LiteRtServer") << "Using model: " << model_path << std::endl;

    std::string executable = BackendUtils::get_backend_binary_path(*litert::spec(), "system");
    LOG(INFO, "LiteRtServer") << "Using executable: " << executable << std::endl;

    port_ = utils::ProcessManager::find_free_port(8001);
    if (port_ == 0) {
        throw std::runtime_error("Failed to find an available port for litert-lm-server");
    }

    std::vector<std::string> args = {
        "--model", model_path,
        "--port", std::to_string(port_),
    };

    std::set<std::string> reserved_flags = {"--model", "--port"};
    if (!extra_args.empty()) {
        std::string validation_error = validate_custom_args(extra_args, reserved_flags);
        if (!validation_error.empty()) {
            throw std::invalid_argument("Invalid custom litert-lm-server arguments:\n" + validation_error);
        }
        std::vector<std::string> custom_args_vec = parse_custom_args(extra_args);
        args.insert(args.end(), custom_args_vec.begin(), custom_args_vec.end());
    }

    bool inherit_output = (log_level_ == "info") || is_debug();
    ProcessHandle started_handle = utils::ProcessManager::start_process(
        executable, args, "", inherit_output, false, {});
    set_process_handle(started_handle);

    if (!has_process_handle(started_handle)) {
        throw std::runtime_error("Failed to start litert-lm-server process");
    }
    LOG(INFO, "LiteRtServer") << "Process started with PID: " << started_handle.pid << std::endl;

    if (!wait_for_ready("/health")) {
        unload();
        throw std::runtime_error("litert-lm-server failed to start or become ready");
    }
    start_backend_watchdog("/health");
    LOG(INFO, "LiteRtServer") << "Server is ready!" << std::endl;
}

void LiteRtServer::unload() {
    stop_backend_watchdog();
    const ProcessHandle handle = consume_process_handle_for_cleanup();
    if (has_process_handle(handle)) {
        LOG(INFO, "LiteRtServer") << "Stopping server (PID: " << handle.pid << ")" << std::endl;
        utils::ProcessManager::stop_process(handle);
    }
}

json LiteRtServer::chat_completion(const json& request) {
    return forward_request("/v1/chat/completions", request);
}

json LiteRtServer::completion(const json& request) {
    return forward_request("/v1/completions", request);
}

}  // namespace backends
}  // namespace lemon

namespace lemon {
namespace backends {
namespace litert {

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<LiteRtServer>(ctx);
}

namespace {
// A litert checkpoint names a single `.litertlm` file (CHECKPOINT:VARIANT). The
// default resolver locates that file in the active snapshot; here we scope the
// download to just it (plus companion config files, added generically upstream)
// so the whole multi-gigabyte repo isn't pulled.
class LiteRtOps : public BackendOps {
public:
    std::optional<std::vector<std::string>> select_checkpoint_files(
        const std::string& main_variant, const std::vector<std::string>& repo_files) const override {
        if (!main_variant.empty()) {
            for (const auto& f : repo_files) {
                if (f == main_variant) {
                    return std::vector<std::string>{f};
                }
            }
            throw std::runtime_error("LiteRT model file not found in repository: " + main_variant);
        }
        for (const auto& f : repo_files) {
            if (ends_with_ci(f, ".litertlm")) {
                return std::vector<std::string>{f};
            }
        }
        throw std::runtime_error("No .litertlm model file found in repository");
    }
};
}  // namespace

const BackendSpec* spec() { return make_spec<LiteRtServer>(descriptor); }
const BackendOps* ops() { return single_ops<LiteRtOps>(); }

}  // namespace litert
}  // namespace backends
}  // namespace lemon
