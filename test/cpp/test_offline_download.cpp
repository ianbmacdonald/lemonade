// offline=true must refuse every model download, including the implicit one
// that /load, chat auto-load, Ollama auto-load and collection loads trigger for
// a registered model that is not on disk.

#include "lemon/model_manager.h"
#include "lemon/runtime_config.h"
#include "lemon/utils/path_utils.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

using lemon::json;

static int failures = 0;

static void check(const char* name, bool ok) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
}

static void set_env_var(const std::string& name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}

static size_t count_files(const std::filesystem::path& root) {
    size_t n = 0;
    std::error_code ec;
    for (auto it = std::filesystem::recursive_directory_iterator(root, ec);
         !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        if (it->is_regular_file(ec)) ++n;
    }
    return n;
}

int main() {
    std::cout << "=== Offline Download Tests ===" << std::endl;

    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path temp_cache =
        std::filesystem::temp_directory_path() / ("lemonade_offline_test_" + std::to_string(tick));
    std::error_code ec;
    std::filesystem::create_directories(temp_cache, ec);
    lemon::utils::set_cache_dir(temp_cache.string());
    lemon::utils::set_config_dir(temp_cache.string());
    set_env_var("LEMONADE_CACHE_DIR", temp_cache.string());
    set_env_var("HF_HOME", (temp_cache / "huggingface").string());
    // Keep the test off the network even if the guard regresses.
    set_env_var("HF_ENDPOINT", "http://127.0.0.1:9");

    lemon::RuntimeConfig cfg(json{
        {"offline", true},
        {"disable_model_filtering", true},
        {"enable_dgpu_gtt", false},
        {"no_fetch_executables", true},
        {"auto_check_model_updates", false},
        {"auto_update_models", false},
    });
    lemon::RuntimeConfig::set_global(&cfg);

    lemon::ModelManager mm;
    const std::string model = "Tiny-Test-Model-GGUF";
    check("registry model exists", mm.model_exists(model));
    check("registry model is not downloaded", !mm.is_model_downloaded(model));

    const size_t files_before = count_files(temp_cache);
    bool threw_offline = false;
    try {
        mm.download_registered_model(mm.get_model_info(model), true);
    } catch (const lemon::OfflineDownloadError&) {
        threw_offline = true;
    } catch (const std::exception& e) {
        std::printf("unexpected exception: %s\n", e.what());
    }
    check("download_registered_model throws OfflineDownloadError", threw_offline);
    check("no files written", count_files(temp_cache) == files_before);
    check("model still not downloaded", !mm.is_model_downloaded(model));

    lemon::RuntimeConfig::set_global(nullptr);
    std::filesystem::remove_all(temp_cache, ec);

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
