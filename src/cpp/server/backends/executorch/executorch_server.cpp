#include "lemon/backends/executorch/executorch_server.h"

#include "lemon/backend_manager.h"
#include "lemon/backends/backend_ops.h"
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/backends/executorch/executorch.h"
#include "lemon/backends/hf_cache_util.h"
#include "lemon/error_types.h"
#include "lemon/utils/custom_args.h"
#include "lemon/utils/http_client.h"
#include "lemon/utils/path_utils.h"
#include "lemon/utils/process_manager.h"
#include <algorithm>
#include <filesystem>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <lemon/utils/aixlog.hpp>

namespace fs = std::filesystem;
using namespace lemon::utils;

namespace lemon {
namespace backends {

namespace {
// et-server rejects larger bodies itself; refusing here keeps an oversized
// request from being copied into a second body just to be refused.
constexpr size_t kMaxClassifyTextBytes = 1024 * 1024;

// The config is mandatory even when a manifest is present: the manifest
// describes the output contract only, and et-server still needs the config to
// check the architecture against its supported input convention.
bool is_complete_model_dir(const fs::path& dir) {
    std::error_code ec;
    return fs::exists(dir / "model.pte", ec) && fs::exists(dir / "tokenizer.json", ec) &&
           fs::exists(dir / "config.json", ec);
}

// The subprocess's stderr is invisible when lemond runs windowless (CI, tray),
// so relaunch et-server just long enough to read the one "et-server: ..." line
// it prints when it refuses a model.
std::string capture_startup_error(const std::string& executable,
                                  const std::vector<std::string>& args) {
    std::string captured;
    int exit_code = -1;
    try {
        exit_code = utils::ProcessManager::run_process_with_output(
            executable, args,
            [&captured](const std::string& line) {
                if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
                    return true;  // skip blank lines
                }
                // Keep everything, not only "et-server: ..." lines: a crash it
                // cannot report itself (a tokenizer panic, a missing library)
                // prints something else.
                if (!captured.empty()) captured += " | ";
                captured += line;
                return captured.size() < 400;  // enough to diagnose; stop there
            },
            "", /*timeout_seconds=*/20);
    } catch (const std::exception&) {
    }

    // A process that dies silently can only be described by how it died and
    // what was installed next to it.
    if (captured.empty()) {
        std::ostringstream detail;
        detail << "exited with code " << exit_code;
        std::error_code ec;
        fs::path bin_dir = path_from_utf8(executable).parent_path();
        std::string listing;
        for (const auto& entry : fs::directory_iterator(bin_dir, ec)) {
            if (ec) break;
            if (!listing.empty()) listing += ", ";
            listing += path_to_utf8(entry.path().filename());
        }
        if (!listing.empty()) detail << "; installed: " << listing;
        captured = detail.str();
    }
    // Trim, and keep the message bounded — it is going into an HTTP error body.
    const size_t start = captured.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    captured = captured.substr(start);
    captured.erase(captured.find_last_not_of(" \t\r\n") + 1);
    if (captured.size() > 400) captured = captured.substr(0, 400) + "...";
    return captured;
}

std::vector<fs::path> find_complete_model_dirs(const fs::path& root) {
    std::vector<fs::path> dirs;
    std::error_code ec;
    fs::recursive_directory_iterator it(root, hf_cache::dir_options(), ec);
    if (ec) return dirs;
    for (auto end = fs::recursive_directory_iterator(); it != end; it.increment(ec)) {
        if (ec) break;
        if (it->is_regular_file(ec) && !ec && it->path().filename() == "model.pte" &&
            is_complete_model_dir(it->path().parent_path())) {
            dirs.push_back(it->path().parent_path());
        }
    }
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}
}  // namespace

// Subprocess contract (ort-server's):
//   GET  /health             -> 200 once every method in the .pte is loaded
//   POST /classify {text}    -> 200 {"labels": {"<label>": <score in [0,1]>, ...}}
//   refusing a model         -> one "et-server: ..." stderr line, then exit(1)
// Exit statuses above 128 are read as signal deaths, so a launcher wrapper must
// exec et-server rather than propagate its status from a shell.
InstallParams ExecutorchServer::get_install_params(const std::string& backend,
                                                   const std::string& version) {
    (void)backend;
    (void)version;
    throw std::runtime_error(
        "The executorch backend uses a pre-installed et-server binary on PATH "
        "and has no downloadable release");
}

ExecutorchServer::ExecutorchServer(const std::string& log_level, ModelManager* model_manager,
                                   BackendManager* backend_manager)
    : WrappedServer("et-server", log_level, model_manager, backend_manager) {
}

ExecutorchServer::~ExecutorchServer() {
    unload();
}

void ExecutorchServer::load(const std::string& model_name,
                            const ModelInfo& model_info,
                            const RecipeOptions& options,
                            bool do_not_upgrade) {
    (void)do_not_upgrade;
    LOG(INFO, "ExecutorchServer") << "Loading model: " << model_name << std::endl;
    LOG(INFO, "ExecutorchServer") << "Per-model settings: " << options.to_log_string() << std::endl;

    std::string extra_args = options.get_option("executorch_args");
    device_type_ = DEVICE_CPU;

    // "system" backend: nothing to install (PATH binary), returns immediately.
    backend_manager_->install_backend(executorch::spec()->recipe, "system");

    std::string model_path = model_info.resolved_path();
    if (model_path.empty() || !fs::exists(model_path)) {
        throw std::runtime_error("Model directory not found for checkpoint: " + model_info.checkpoint());
    }

    // load() is the sole arbiter of ambiguity: resolve_checkpoint_path falls back
    // to the cache root when resolution is ambiguous, so a complete root that also
    // contains a nested complete model must still be rejected here — never assume
    // the resolved path is the only candidate.
    auto candidates = find_complete_model_dirs(path_from_utf8(model_path));
    if (candidates.empty()) {
        throw NonRetryableLoadException(
            "No servable model directory under '" + model_path +
            "': need model.pte + tokenizer.json + config.json "
            "(manifest.json is optional and overrides the output contract)");
    }
    if (candidates.size() > 1) {
        std::string listing;
        for (const auto& c : candidates) listing += "\n  " + path_to_utf8(c);
        throw NonRetryableLoadException(
            "Ambiguous model layout under '" + model_path + "': " +
            std::to_string(candidates.size()) +
            " complete model directories found — keep exactly one:" + listing);
    }
    model_path = path_to_utf8(candidates.front());
    LOG(INFO, "ExecutorchServer") << "Using model: " << model_path << std::endl;

    std::string executable = BackendUtils::get_backend_binary_path(*executorch::spec(), "system");
    LOG(INFO, "ExecutorchServer") << "Using executable: " << executable << std::endl;

    const int port = choose_port();

    std::vector<std::string> args = {
        "--model-path", model_path,
        "--port", std::to_string(port),
    };

    std::set<std::string> reserved_flags = {"--model-path", "--port"};
    if (!extra_args.empty()) {
        std::string validation_error = validate_custom_args(extra_args, reserved_flags);
        if (!validation_error.empty()) {
            throw std::invalid_argument("Invalid custom et-server arguments:\n" + validation_error);
        }
        std::vector<std::string> custom_args_vec = parse_custom_args(extra_args);
        args.insert(args.end(), custom_args_vec.begin(), custom_args_vec.end());
    }

    bool inherit_output = (log_level_ == "info") || is_debug();
    ProcessHandle started_handle = utils::ProcessManager::start_process(
        executable, args, "", inherit_output, false, {});
    set_process_handle(started_handle, executable, args);

    if (!has_process_handle(started_handle)) {
        throw std::runtime_error("Failed to start et-server process");
    }
    LOG(INFO, "ExecutorchServer") << "Process started with PID: " << started_handle.pid << std::endl;

    constexpr long kReadyTimeoutSeconds = 600;
    if (!wait_for_ready("/health", kReadyTimeoutSeconds)) {
        unload();
        // Every failure below is non-retryable: et-server is CPU-only, so the
        // Router's evict-everything-and-retry would only cost the resident LLM
        // and repeat a load that failed for reasons eviction does not change.
        const int exit_code = startup_exit_code();
        if (load_cancel_ && load_cancel_->load()) {
            throw NonRetryableLoadException("et-server load cancelled");
        }
        if (exit_code < 0) {
            // Still running at the deadline: on a small box that is memory
            // thrash, and a capture relaunch would load the model a second time.
            throw NonRetryableLoadException("et-server did not become ready within " +
                                            std::to_string(kReadyTimeoutSeconds) + " s");
        }
        if (exit_code == 128 + 9) {
            // SIGKILL during load is the OOM killer's signature; relaunching to
            // capture stderr would repeat the full load under the same pressure.
            throw NonRetryableLoadException(
                "et-server was killed by signal 9 (likely out of memory)");
        }
        // A fast refusal or crash (abort, segfault) is cheap to repeat, and the
        // relaunch is the only way to show the diagnostic line it printed.
        std::string details = capture_startup_error(executable, args);
        std::string status = exit_code > 128
            ? "killed by signal " + std::to_string(exit_code - 128)
            : "exited with code " + std::to_string(exit_code);
        throw NonRetryableLoadException("et-server " + status + " during startup" +
                                        (details.empty() ? "" : ": " + details));
    }
    start_backend_watchdog("/health");
    LOG(INFO, "ExecutorchServer") << "Server is ready!" << std::endl;
}

void ExecutorchServer::unload() {
    stop_backend_watchdog();
    const ProcessHandle handle = consume_process_handle_for_cleanup();
    if (has_process_handle(handle)) {
        LOG(INFO, "ExecutorchServer") << "Stopping server (PID: " << handle.pid << ")" << std::endl;
        utils::ProcessManager::stop_process(handle);
    }
}

json ExecutorchServer::forward_classify(const std::string& text, const json& params) {
    json body = {{"text", text}};
    if (params.contains("top_k")) body["top_k"] = params["top_k"];
    return forward_request("/classify", body, 120);
}

json ExecutorchServer::classify(const json& request) {
    // Accept either OpenAI-style "input" or plain "text".
    std::string text;
    if (request.contains("text") && request["text"].is_string()) {
        text = request["text"].get<std::string>();
    } else if (request.contains("input") && request["input"].is_string()) {
        text = request["input"].get<std::string>();
    } else {
        return json{
            {"error", {
                {"message", "Missing 'input' (or 'text') string in classify request"},
                {"type", "invalid_request_error"},
                {"status_code", 400},
            }}
        };
    }
    if (text.size() > kMaxClassifyTextBytes) {
        return json{
            {"error", {
                {"message", "Classify text exceeds " + std::to_string(kMaxClassifyTextBytes) +
                            " bytes"},
                {"type", "invalid_request_error"},
                {"status_code", 413},
            }}
        };
    }
    return forward_classify(text, request);
}

}  // namespace backends
}  // namespace lemon

namespace lemon {
namespace backends {
namespace executorch {

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<ExecutorchServer>(ctx);
}

namespace {
// Resolve to the one directory holding model.pte + tokenizer.json + config.json,
// which is what et-server takes as --model-path.
class ExecutorchOps : public BackendOps {
public:
    // For a Hugging Face cache, `refs/main` names the active revision — an older
    // snapshot lying beside it is normal, not an ambiguity, so scope the search
    // to that snapshot. Only a local import (no refs/main) is searched whole,
    // where exactly-one really is the right rule.
    std::string resolve_checkpoint_path(const ModelInfo&,
                                        const CheckpointResolveContext& ctx) const override {
        fs::path cache = path_from_utf8(ctx.model_cache_path);
        fs::path active = hf_cache::active_snapshot_path(cache);
        if (!active.empty()) {
            std::string found = find_imported_checkpoint(path_to_utf8(active));
            return found.empty() ? path_to_utf8(active) : found;
        }
        std::string found = find_imported_checkpoint(ctx.model_cache_path);
        return found.empty() ? ctx.model_cache_path : found;
    }

    // Resolve only when the layout is unambiguous: exactly one complete model
    // directory. Anything else returns "" and load() reports a precise error;
    // resolution runs during bulk model listing, so it must never throw.
    std::string find_imported_checkpoint(const std::string& import_dir) const override {
        fs::path dir = path_from_utf8(import_dir);
        if (!hf_cache::exists(dir)) {
            return "";
        }
        auto candidates = find_complete_model_dirs(dir);
        if (candidates.size() != 1) {
            if (candidates.size() > 1) {
                LOG(WARNING, "ExecutorchServer")
                    << candidates.size() << " complete model directories under "
                    << import_dir << "; refusing to pick one" << std::endl;
            }
            return "";
        }
        return path_to_utf8(candidates.front());
    }
};
}  // namespace

const BackendSpec* spec() { return make_spec<ExecutorchServer>(descriptor); }
const BackendOps* ops() { return single_ops<ExecutorchOps>(); }

}  // namespace executorch
}  // namespace backends
}  // namespace lemon
