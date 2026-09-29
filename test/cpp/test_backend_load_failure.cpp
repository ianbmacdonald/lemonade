// A backend that dies while loading must be reported with how it died, and the
// Router must not answer an unfixable failure by evicting every other model and
// loading the failed one a second time. The exit status is only observable
// inside wait_for_ready(), which reaps the child, so the Linux cases start real
// /bin/sh children rather than faking ProcessManager.

#include "lemon/error_types.h"
#include "lemon/router.h"
#include "lemon/utils/process_manager.h"
#include "lemon/wrapped_server.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(const std::string& what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

}  // namespace

namespace lemon {

class ShellWrappedServer : public WrappedServer {
public:
    ShellWrappedServer() : WrappedServer("shell", "error", nullptr, nullptr) {}

    void load(const std::string&, const ModelInfo&, const RecipeOptions&, bool) override {}

    void unload() override {
        const ProcessHandle handle = consume_process_handle_for_cleanup();
        if (has_process_handle(handle)) utils::ProcessManager::stop_process(handle);
    }

    // Returns wait_for_ready's verdict and leaves startup_exit_code() readable.
    bool start_and_wait(const std::string& script, long timeout_seconds) {
        choose_port();
        const std::vector<std::string> args = {"-c", script};
        ProcessHandle handle = utils::ProcessManager::start_process("/bin/sh", args);
        set_process_handle(handle, "/bin/sh", args);
        return wait_for_ready("/health", timeout_seconds, 50);
    }

    using WrappedServer::startup_exit_code;
};

}  // namespace lemon

using lemon::NonRetryableLoadException;
using lemon::Router;
using lemon::ShellWrappedServer;

static void test_nuclear_retry_classification() {
    check("NonRetryableLoadException skips the nuclear retry",
          Router::load_error_skips_nuclear_retry(
              NonRetryableLoadException("et-server was killed by signal 9 (likely out of memory)")));
    check("the same text as a plain runtime_error does not",
          !Router::load_error_skips_nuclear_retry(
              std::runtime_error("et-server was killed by signal 9 (likely out of memory)")));
    check("a missing file still skips it",
          Router::load_error_skips_nuclear_retry(
              std::runtime_error("Model directory not found for checkpoint: x")));
    check("a generic failure keeps it",
          !Router::load_error_skips_nuclear_retry(
              std::runtime_error("llama-server failed to allocate VRAM")));
}

#ifdef __linux__
static void test_startup_exit_code() {
    {
        ShellWrappedServer server;
        check("startup_exit_code is -1 before any wait", server.startup_exit_code() == -1);
    }
    {
        ShellWrappedServer server;
        const bool ready = server.start_and_wait("kill -9 $$", 10);
        check("SIGKILL child is not ready", !ready);
        check("SIGKILL child reports 137, got " + std::to_string(server.startup_exit_code()),
              server.startup_exit_code() == 137);
    }
    {
        ShellWrappedServer server;
        const bool ready = server.start_and_wait("exit 3", 10);
        check("exit 3 child is not ready", !ready);
        check("exit 3 child reports 3, got " + std::to_string(server.startup_exit_code()),
              server.startup_exit_code() == 3);
    }
    {
        ShellWrappedServer server;
        std::atomic<bool> cancel{false};
        server.set_load_cancel_flag(&cancel);
        std::thread canceller([&cancel] {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            cancel.store(true);
        });
        const auto start = std::chrono::steady_clock::now();
        const bool ready = server.start_and_wait("exec sleep 30", 20);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        canceller.join();
        server.set_load_cancel_flag(nullptr);
        check("cancelled child is not ready", !ready);
        check("cancelled child reports -1, got " + std::to_string(server.startup_exit_code()),
              server.startup_exit_code() == -1);
        check("cancel returns promptly",
              elapsed < std::chrono::seconds(5));
    }
    {
        ShellWrappedServer server;
        const bool ready = server.start_and_wait("exec sleep 30", 1);
        check("timed-out child is not ready", !ready);
        check("timed-out child reports -1, got " + std::to_string(server.startup_exit_code()),
              server.startup_exit_code() == -1);
        server.unload();
    }
    {
        ShellWrappedServer server;
        server.start_and_wait("exit 3", 10);
        server.start_and_wait("exec sleep 30", 1);
        check("a later timeout resets the previous exit status",
              server.startup_exit_code() == -1);
        server.unload();
    }
}
#endif

int main() {
    test_nuclear_retry_classification();
#ifdef __linux__
    test_startup_exit_code();
#endif
    if (failures > 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
