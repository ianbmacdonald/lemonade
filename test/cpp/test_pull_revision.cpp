#include "lemon/model_manager.h"
#include "lemon/model_registry.h"
#include "lemon/registry_files.h"
#include "lemon/utils/path_utils.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using lemon::ModelInfo;
using lemon::ModelManager;
using lemon::json;
using lemon::utils::path_from_utf8;
using lemon::utils::path_to_utf8;

static int g_failures = 0;

static void check(const std::string& name, bool ok) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (!ok) ++g_failures;
}

static fs::path make_temp_dir() {
    fs::path dir = fs::temp_directory_path();
    dir /= "pull_revision_" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    fs::create_directories(dir);
    return dir;
}

static void write_file(const fs::path& path, const std::string& contents) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

static void test_revision_validation() {
    const std::string sha = "6e091d820cbe8f22eeb604d136403eca290b8c1e";
    for (const std::string& valid :
         {std::string(), std::string("main"), sha, std::string("v1.0"),
          std::string("refs/pr/1"), std::string("release_2024-01"),
          std::string(128, 'a')}) {
        check("revision accepted: '" + valid.substr(0, 40) + "'",
              lemon::registry_revision_error(valid).empty());
    }
    for (const std::string& invalid :
         {std::string(129, 'a'), std::string("../main"), std::string("a..b"),
          std::string("/main"), std::string("-main"), std::string("bad revision"),
          std::string("rev~1"), std::string("re?v"), std::string("main@{1}"),
          std::string("a\\b"), std::string("tag:v1")}) {
        check("revision rejected: '" + invalid.substr(0, 40) + "'",
              !lemon::registry_revision_error(invalid).empty());
    }
}

static void test_revision_api_url() {
    check("a commit sha is used verbatim in the revision API URL",
          lemon::huggingface_repository_api_url(
              "https://huggingface.co", "org/model", "abc123") ==
              "https://huggingface.co/api/models/org/model/revision/abc123");
    check("a slash in a branch name is encoded so the Hub resolves the ref",
          lemon::huggingface_repository_api_url(
              "https://huggingface.co", "org/model", "refs/pr/1") ==
              "https://huggingface.co/api/models/org/model/revision/refs%2Fpr%2F1");
}

static void test_pinned_snapshot_id(const fs::path& root) {
    const fs::path cache = root / "provenance" / "models--org--repo";
    namespace rf = lemon::registry_files;

    check("no provenance file means no pin", rf::pinned_snapshot_id(cache, "m").empty());

    write_file(cache / ".lemonade_registry.json",
               R"({"processed_models":{
                   "pinned":{"selection":"s","snapshot_id":"bbbb","pinned_revision":"bbbb"},
                   "follow":{"selection":"s","snapshot_id":"aaaa"},
                   "escape":{"selection":"s","snapshot_id":"x","pinned_revision":"../aaaa"}}})");
    check("a pinned entry yields its snapshot", rf::pinned_snapshot_id(cache, "pinned") == "bbbb");
    check("an unpinned entry yields no pin", rf::pinned_snapshot_id(cache, "follow").empty());
    check("an unknown model yields no pin", rf::pinned_snapshot_id(cache, "other").empty());
    check("a pin that is not a single path component is ignored",
          rf::pinned_snapshot_id(cache, "escape").empty());

    write_file(cache / ".lemonade_registry.json", "{not json");
    check("malformed provenance yields no pin", rf::pinned_snapshot_id(cache, "pinned").empty());
}

static void test_pinned_model_resolution(const fs::path& root, const fs::path& hf_root) {
    const fs::path repo = hf_root / "models--org--pin";
    const fs::path main_snapshot = repo / "snapshots" / "aaaa";
    const fs::path pinned_snapshot = repo / "snapshots" / "bbbb";
    write_file(main_snapshot / "model.gguf", "GGUF-main");
    write_file(pinned_snapshot / "model.gguf", "GGUF-pinned");
    write_file(repo / "refs" / "main", "aaaa");
    write_file(repo / ".lemonade_registry.json",
               R"({"processed_models":{
                   "user.pinned":{"selection":"s","snapshot_id":"bbbb","pinned_revision":"bbbb"},
                   "user.follow":{"selection":"s","snapshot_id":"aaaa"}}})");

    const json user_models = {
        {"pinned", json{{"checkpoint", "org/pin:model.gguf"}, {"recipe", "llamacpp"}}},
        {"follow", json{{"checkpoint", "org/pin:model.gguf"}, {"recipe", "llamacpp"}}},
    };
    write_file(root / "user_models.json", user_models.dump(2));

    ModelManager manager;

    const ModelInfo pinned = manager.get_model_info("user.pinned");
    check("pinned model reports its pinned revision", pinned.pinned_revision == "bbbb");
    check("pinned model resolves inside the pinned snapshot",
          pinned.downloaded &&
              path_from_utf8(pinned.resolved_path()) == pinned_snapshot / "model.gguf");

    const ModelInfo follow = manager.get_model_info("user.follow");
    check("unpinned model sharing the repo has no pin", follow.pinned_revision.empty());
    check("unpinned model sharing the repo follows refs/main",
          follow.downloaded &&
              path_from_utf8(follow.resolved_path()) == main_snapshot / "model.gguf");

    fs::remove(pinned_snapshot / "model.gguf");
    manager.invalidate_models_cache();
    const ModelInfo missing = manager.get_model_info("user.pinned");
    check("a pinned model whose snapshot lacks the file is not downloaded",
          !missing.downloaded);
    check("a pinned model never falls back to another snapshot",
          path_from_utf8(missing.resolved_path()).string().find(
              path_to_utf8(main_snapshot)) == std::string::npos);

    fs::remove_all(pinned_snapshot);
    manager.invalidate_models_cache();
    const ModelInfo gone = manager.get_model_info("user.pinned");
    check("a pinned model whose snapshot is gone is not downloaded",
          !gone.downloaded && gone.resolved_path().empty());
}

static void test_pinnable_snapshot_id() {
    namespace rf = lemon::registry_files;
    const std::string sha = "6e091d820cbe8f22eeb604d136403eca290b8c1e";
    check("a branch resolved to a commit can be pinned", rf::is_pinnable_snapshot_id("main", sha));
    check("a commit sha echoed back can be pinned", rf::is_pinnable_snapshot_id(sha, sha));
    check("a slashed ref echoed back for lack of a commit is refused",
          !rf::is_pinnable_snapshot_id("refs/pr/1", "refs/pr/1"));
    check("a tag echoed back for lack of a commit is refused",
          !rf::is_pinnable_snapshot_id("v1.0", "v1.0"));
    check("an empty snapshot id is refused", !rf::is_pinnable_snapshot_id("main", ""));
    check("a snapshot id that escapes snapshots/ is refused",
          !rf::is_pinnable_snapshot_id("main", "../x"));
}

int main() {
    fs::path temp = make_temp_dir();
    fs::path hf_root = temp / "hf";
    fs::create_directories(hf_root);

    lemon::utils::set_cache_dir(path_to_utf8(temp));
    lemon::utils::set_config_dir(path_to_utf8(temp));
    lemon::utils::set_models_dir(path_to_utf8(hf_root));

    test_revision_validation();
    test_revision_api_url();
    test_pinned_snapshot_id(temp);
    test_pinned_model_resolution(temp, hf_root);
    test_pinnable_snapshot_id();

    std::error_code ec;
    fs::remove_all(temp, ec);

    if (g_failures == 0) {
        std::printf("All pull revision tests passed.\n");
    } else {
        std::printf("%d pull revision test(s) failed.\n", g_failures);
    }
    return g_failures == 0 ? 0 : 1;
}
