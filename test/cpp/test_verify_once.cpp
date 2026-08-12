// Unit tests for the ".verified" sidecar fast path in HttpClient::download_file.
// A sidecar recording {algorithm, digest, size, mtime} lets a repeat pull of an
// already-complete file skip the full content re-hash; any divergence (size,
// mtime, digest, or a missing/garbled sidecar) falls back to the full hash.
//
// No network: the URL points at a closed loopback port, so any code path that
// reaches the transport fails fast and observably. That makes the skip provable:
// a file whose content does NOT match the expected digest, but whose sidecar
// attests it does, can only return success if the sidecar short-circuited.
//
// Checks use an explicit pass/fail counter (not assert()) so the test stays
// effective under the Release build the CI `default` preset uses, where
// -DNDEBUG would compile assert() to a no-op.

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include <lemon/utils/http_client.h>

namespace fs = std::filesystem;
using lemon::utils::DownloadOptions;
using lemon::utils::HttpClient;

namespace {

// sha256 of "lemonade verify-once test content\n"
const std::string kContent = "lemonade verify-once test content\n";
const std::string kContentSha256 =
    "40b8c9cb744d4edc60c7172bf0f3b3c67b58a3431ac9921c062dc0fd843bfd0f";
// sha256 of "LEMONADE VERIFY-ONCE TEST CONTENT\n" (same length as kContent)
const std::string kOtherContent = "LEMONADE VERIFY-ONCE TEST CONTENT\n";

const std::string kDeadUrl = "http://127.0.0.1:9/never-served.bin";

struct TestResult {
    int passed = 0;
    int failed = 0;

    void check(bool cond, const std::string& name) {
        if (cond) {
            printf("[PASS] %s\n", name.c_str());
            ++passed;
        } else {
            printf("[FAIL] %s\n", name.c_str());
            ++failed;
        }
    }
};

void write_file(const fs::path& p, const std::string& content) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out << content;
}

void write_sidecar(const fs::path& file, const std::string& algorithm,
                   const std::string& digest, uintmax_t size, long long mtime) {
    std::ofstream out(fs::path(file) += ".verified", std::ios::trunc);
    out << algorithm << "\n" << digest << "\n" << size << "\n" << mtime << "\n";
}

long long mtime_count(const fs::path& p) {
    return static_cast<long long>(fs::last_write_time(p).time_since_epoch().count());
}

DownloadOptions fast_fail_options(const std::string& digest) {
    DownloadOptions options;
    options.expected_hash_algorithm = "sha256";
    options.expected_hash = digest;
    options.max_retries = 0;
    options.initial_retry_delay_ms = 1;
    options.max_retry_delay_ms = 1;
    options.connect_timeout = 2;
    return options;
}

}  // namespace

int main() {
    TestResult r;
    const fs::path dir = fs::temp_directory_path() / "lemonade-verify-once-test";
    fs::remove_all(dir);
    fs::create_directories(dir);

    {
        // A matching sidecar short-circuits the hash: the file content does NOT
        // match the expected digest, so only the sidecar path can return success.
        const fs::path f = dir / "skip.bin";
        write_file(f, kOtherContent);
        write_sidecar(f, "sha256", kContentSha256, fs::file_size(f), mtime_count(f));
        auto result = HttpClient::download_file(kDeadUrl, f.string(), nullptr, {},
                                                fast_fail_options(kContentSha256));
        r.check(result.success, "matching sidecar skips the re-hash (attested success)");
        r.check(fs::exists(f), "attested file is left in place");
    }

    {
        // First full verification records a sidecar.
        const fs::path f = dir / "record.bin";
        write_file(f, kContent);
        auto result = HttpClient::download_file(kDeadUrl, f.string(), nullptr, {},
                                                fast_fail_options(kContentSha256));
        r.check(result.success, "full verification of a complete file succeeds");
        const fs::path sidecar = fs::path(f) += ".verified";
        r.check(fs::exists(sidecar), "successful verification writes the sidecar");
        std::ifstream in(sidecar);
        std::string algorithm, digest;
        std::getline(in, algorithm);
        std::getline(in, digest);
        r.check(algorithm == "sha256" && digest == kContentSha256,
                "sidecar records the verified algorithm and digest");
    }

    {
        // A sidecar whose recorded size disagrees with the file is stale: fall
        // back to the full hash (which passes here) and rewrite the sidecar.
        const fs::path f = dir / "stale-size.bin";
        write_file(f, kContent);
        write_sidecar(f, "sha256", kContentSha256, fs::file_size(f) + 1, mtime_count(f));
        auto result = HttpClient::download_file(kDeadUrl, f.string(), nullptr, {},
                                                fast_fail_options(kContentSha256));
        r.check(result.success, "stale-size sidecar falls back to a passing full hash");
    }

    {
        // Changed content with a stale sidecar (old mtime) is NOT attested: the
        // full hash runs, fails, the file and sidecar are removed, and the
        // forced re-download hits the dead URL.
        const fs::path f = dir / "tampered.bin";
        write_file(f, kContent);
        write_sidecar(f, "sha256", kContentSha256, fs::file_size(f), mtime_count(f) - 1);
        auto result = HttpClient::download_file(kDeadUrl, f.string(), nullptr, {},
                                                fast_fail_options(kContentSha256));
        // Content still matches the digest, so this passes via the full hash.
        r.check(result.success, "mtime-mismatched sidecar falls back to the full hash");

        const fs::path g = dir / "tampered2.bin";
        write_file(g, kOtherContent);
        write_sidecar(g, "sha256", kContentSha256, fs::file_size(g), mtime_count(g) - 1);
        auto bad = HttpClient::download_file(kDeadUrl, g.string(), nullptr, {},
                                             fast_fail_options(kContentSha256));
        r.check(!bad.success, "unattested content mismatch is caught and re-download fails on the dead URL");
        r.check(!fs::exists(g), "failed verification removes the file");
        r.check(!fs::exists(fs::path(g) += ".verified"),
                "failed verification removes the sidecar");
    }

    {
        // A garbled sidecar never attests.
        const fs::path f = dir / "garbled.bin";
        write_file(f, kOtherContent);
        std::ofstream out(fs::path(f) += ".verified", std::ios::trunc);
        out << "not-a-sidecar\n";
        out.close();
        auto result = HttpClient::download_file(kDeadUrl, f.string(), nullptr, {},
                                                fast_fail_options(kContentSha256));
        r.check(!result.success, "garbled sidecar falls back to the full hash (which fails)");
    }

    fs::remove_all(dir);
    printf("\n%d passed, %d failed\n", r.passed, r.failed);
    return r.failed == 0 ? 0 : 1;
}
