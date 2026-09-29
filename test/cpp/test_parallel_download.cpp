// Unit tests for parallel ranged downloads in HttpClient::download_file.
//
// A loopback cpp-httplib server serves a deterministic payload. cpp-httplib
// answers Range requests itself (206 + Content-Range) unless a handler sets
// status 200, which is how the "server ignores Range" case is modelled.
//
// Checks use an explicit pass/fail counter (not assert()) so the test stays
// effective under the Release build the CI `default` preset uses.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <lemon/runtime_config.h>
#include <lemon/utils/http_client.h>

namespace fs = std::filesystem;
using lemon::utils::DownloadOptions;
using lemon::utils::DownloadResult;
using lemon::utils::HttpClient;
using lemon::utils::HttpSecurityPolicy;

namespace {

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

std::string make_payload(size_t size) {
    std::string data(size, '\0');
    uint32_t x = 2463534242u;
    for (size_t i = 0; i < size; ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        data[i] = static_cast<char>(x & 0xff);
    }
    return data;
}

std::string read_all(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_all(const fs::path& p, const std::string& data) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out << data;
}

// Start offset of a "bytes=S-E" header, or -1 when absent or unparsable.
long long range_start(const std::string& header) {
    const std::string prefix = "bytes=";
    if (header.rfind(prefix, 0) != 0) {
        return -1;
    }
    try {
        return std::stoll(header.substr(prefix.size()));
    } catch (...) {
        return -1;
    }
}

struct RequestLog {
    std::mutex mutex;
    std::vector<std::string> ranges;  // "" = no Range header

    void add(const httplib::Request& req) {
        std::lock_guard<std::mutex> lock(mutex);
        ranges.push_back(req.get_header_value("Range"));
    }
    std::vector<std::string> take() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<std::string> out;
        out.swap(ranges);
        return out;
    }
};

size_t count_body_ranges(const std::vector<std::string>& ranges) {
    size_t n = 0;
    for (const auto& r : ranges) {
        if (!r.empty() && r != "bytes=0-0") {
            ++n;
        }
    }
    return n;
}

constexpr size_t kMinBytes = 64 * 1024;

DownloadOptions parallel_options() {
    DownloadOptions options;
    options.allow_parallel = true;
    options.parallel_min_bytes = kMinBytes;
    options.max_retries = 0;
    options.initial_retry_delay_ms = 1;
    options.max_retry_delay_ms = 1;
    options.connect_timeout = 5;
    return options;
}

fs::path partial_of(const fs::path& f) {
    return fs::path(f) += ".partial";
}

fs::path journal_of(const fs::path& f) {
    return fs::path(f) += ".partial.ranges";
}

}  // namespace

int main() {
    TestResult r;
    const fs::path dir = fs::temp_directory_path() / "lemonade-parallel-download-test";
    fs::remove_all(dir);
    fs::create_directories(dir);

    const std::string payload = make_payload(1024 * 1024 + 12345);
    const std::string small = make_payload(32 * 1024);
    const long long half = static_cast<long long>(payload.size() / 2);

    RequestLog log;
    std::atomic<bool> flaky_fails{true};

    httplib::Server server;
    server.Get("/ranged.bin", [&](const httplib::Request& req, httplib::Response& res) {
        log.add(req);
        res.set_content(payload, "application/octet-stream");
    });
    server.Get("/norange.bin", [&](const httplib::Request& req, httplib::Response& res) {
        log.add(req);
        res.status = 200;
        res.set_content(payload, "application/octet-stream");
    });
    server.Get("/small.bin", [&](const httplib::Request& req, httplib::Response& res) {
        log.add(req);
        res.set_content(small, "application/octet-stream");
    });
    server.Get("/fail503.bin", [&](const httplib::Request& req, httplib::Response& res) {
        log.add(req);
        if (range_start(req.get_header_value("Range")) >= half) {
            res.status = 503;
            res.set_content("unavailable", "text/plain");
            return;
        }
        res.set_content(payload, "application/octet-stream");
    });
    // Ranges in the upper half are cut off mid-body while flaky_fails is set.
    server.Get("/truncated.bin", [&](const httplib::Request& req, httplib::Response& res) {
        log.add(req);
        const bool cut = flaky_fails.load() && range_start(req.get_header_value("Range")) >= half;
        res.set_content_provider(
            payload.size(), "application/octet-stream",
            [&payload, cut](size_t offset, size_t length, httplib::DataSink& sink) {
                if (cut) {
                    sink.write(payload.data() + offset, (std::min)(length, size_t(1000)));
                    return false;
                }
                const size_t n = (std::min)(length, size_t(64 * 1024));
                sink.write(payload.data() + offset, n);
                return true;
            });
    });
    // Ranges in the upper half never send a byte.
    server.Get("/stall.bin", [&](const httplib::Request& req, httplib::Response& res) {
        log.add(req);
        if (range_start(req.get_header_value("Range")) >= half) {
            std::this_thread::sleep_for(std::chrono::seconds(8));
        }
        res.set_content(payload, "application/octet-stream");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    std::thread server_thread([&server]() { server.listen_after_bind(); });
    server.wait_until_ready();
    const std::string base = "http://127.0.0.1:" + std::to_string(port);
    const auto policy = HttpSecurityPolicy::TrustedLoopback;

    HttpClient::set_download_connections(4);

    {
        const fs::path f = dir / "ranged.bin";
        log.take();
        size_t last_progress = 0;
        bool monotonic = true;
        auto progress = [&](size_t done, size_t total) {
            if (done < last_progress || (total != 0 && total != payload.size())) {
                monotonic = false;
            }
            last_progress = done;
            return true;
        };
        auto result = HttpClient::download_file(base + "/ranged.bin", f.string(), progress, {},
                                                parallel_options(), policy);
        const auto ranges = log.take();
        r.check(result.success, "ranged download succeeds");
        r.check(read_all(f) == payload, "ranged download is byte-identical");
        r.check(count_body_ranges(ranges) >= 4, "ranged download issues several Range requests");
        r.check(!fs::exists(partial_of(f)) && !fs::exists(journal_of(f)),
                "ranged download leaves no partial or journal");
        r.check(monotonic && last_progress == payload.size(),
                "progress is monotonic and ends at the file size");
    }

    {
        const fs::path f = dir / "norange.bin";
        log.take();
        auto result = HttpClient::download_file(base + "/norange.bin", f.string(), nullptr, {},
                                                parallel_options(), policy);
        const auto ranges = log.take();
        r.check(result.success, "download from a server that ignores Range succeeds");
        r.check(read_all(f) == payload, "single-stream fallback is byte-identical");
        r.check(count_body_ranges(ranges) == 0 && ranges.size() == 2,
                "Range-ignoring server gets one probe and one plain GET");
    }

    {
        const fs::path f = dir / "small-known.bin";
        log.take();
        auto options = parallel_options();
        options.expected_size = small.size();
        auto result = HttpClient::download_file(base + "/small.bin", f.string(), nullptr, {},
                                                options, policy);
        const auto ranges = log.take();
        r.check(result.success && read_all(f) == small, "small file downloads intact");
        r.check(ranges.size() == 1 && ranges[0].empty(),
                "small file with a known size is one plain GET (no probe)");

        const fs::path g = dir / "small-unknown.bin";
        auto unknown = HttpClient::download_file(base + "/small.bin", g.string(), nullptr, {},
                                                 parallel_options(), policy);
        const auto probed = log.take();
        r.check(unknown.success && read_all(g) == small, "small file of unknown size downloads intact");
        r.check(count_body_ranges(probed) == 0, "small file below the threshold is not split");
    }

    {
        const fs::path f = dir / "optout.bin";
        log.take();
        DownloadOptions options = parallel_options();
        options.allow_parallel = false;
        auto result = HttpClient::download_file(base + "/ranged.bin", f.string(), nullptr, {},
                                                options, policy);
        const auto ranges = log.take();
        r.check(result.success && read_all(f) == payload, "opted-out download is intact");
        r.check(ranges.size() == 1 && ranges[0].empty(), "opted-out download is one plain GET");

        HttpClient::set_download_connections(1);
        const fs::path g = dir / "one-connection.bin";
        auto single = HttpClient::download_file(base + "/ranged.bin", g.string(), nullptr, {},
                                                parallel_options(), policy);
        const auto single_ranges = log.take();
        r.check(single.success && read_all(g) == payload, "download_connections=1 download is intact");
        r.check(single_ranges.size() == 1 && single_ranges[0].empty(),
                "download_connections=1 is one plain GET");
        HttpClient::set_download_connections(4);

        HttpClient::set_download_rate_limit(1024LL * 1024 * 1024);
        const fs::path h = dir / "rate-limited.bin";
        auto limited = HttpClient::download_file(base + "/ranged.bin", h.string(), nullptr, {},
                                                 parallel_options(), policy);
        const auto limited_ranges = log.take();
        HttpClient::set_download_rate_limit(0);
        r.check(limited.success && read_all(h) == payload, "rate-limited download is intact");
        r.check(limited_ranges.size() == 1 && limited_ranges[0].empty(),
                "an active rate limit keeps a single stream");
    }

    {
        const fs::path f = dir / "fail503.bin";
        log.take();
        auto result = HttpClient::download_file(base + "/fail503.bin", f.string(), nullptr, {},
                                                parallel_options(), policy);
        log.take();
        r.check(!result.success, "a failed range fails the whole download");
        r.check(result.http_code == 503, "the failing range's HTTP status is reported");
        r.check(!fs::exists(f), "no final file after a failed range");
        const std::string partial = read_all(partial_of(f));
        r.check(partial.size() <= static_cast<size_t>(half) &&
                    partial == payload.substr(0, partial.size()),
                "partial is cut back to a byte-exact contiguous prefix");
        r.check(!fs::exists(journal_of(f)), "no journal after a cleanly failed attempt");
    }

    {
        const fs::path f = dir / "truncated.bin";
        log.take();
        flaky_fails = true;
        auto result = HttpClient::download_file(base + "/truncated.bin", f.string(), nullptr, {},
                                                parallel_options(), policy);
        r.check(!result.success, "a truncated range fails the whole download");
        r.check(!fs::exists(f), "no final file after a truncated range");
        const std::string partial = read_all(partial_of(f));
        r.check(partial == payload.substr(0, partial.size()) && partial.size() < payload.size(),
                "truncated attempt keeps only a byte-exact prefix");

        flaky_fails = false;
        const size_t kept = partial.size();
        auto resumed = HttpClient::download_file(base + "/truncated.bin", f.string(), nullptr, {},
                                                 parallel_options(), policy);
        const auto ranges = log.take();
        bool resumed_from_prefix = kept == 0;
        for (const auto& range : ranges) {
            if (range_start(range) == static_cast<long long>(kept) && kept > 0) {
                resumed_from_prefix = true;
            }
        }
        r.check(resumed.success && read_all(f) == payload, "resumed download completes byte-identical");
        r.check(resumed_from_prefix, "resume restarts at the kept prefix");
    }

    {
        // An interrupted process leaves out-of-order bytes past the prefix the
        // journal records; they must never be trusted.
        const fs::path f = dir / "crashed.bin";
        const size_t prefix = 200000;
        std::string holey = payload.substr(0, prefix) + std::string(payload.size() - prefix, 'X');
        write_all(partial_of(f), holey);
        write_all(journal_of(f), std::to_string(prefix) + "\n");
        r.check(HttpClient::resumable_partial_bytes(f.string()) == prefix,
                "resumable_partial_bytes reports the journal prefix, not the file size");
        log.take();
        auto result = HttpClient::download_file(base + "/ranged.bin", f.string(), nullptr, {},
                                                parallel_options(), policy);
        log.take();
        r.check(result.success && read_all(f) == payload,
                "resume after an interrupted parallel download is byte-identical");
        r.check(!fs::exists(journal_of(f)), "journal is removed after completion");

        // The same leftovers must not fool a single-stream download either.
        const fs::path g = dir / "crashed-single.bin";
        write_all(partial_of(g), holey);
        write_all(journal_of(g), std::to_string(prefix) + "\n");
        DownloadOptions options = parallel_options();
        options.allow_parallel = false;
        auto single = HttpClient::download_file(base + "/ranged.bin", g.string(), nullptr, {},
                                                options, policy);
        log.take();
        r.check(single.success && read_all(g) == payload,
                "single-stream resume after an interrupted parallel download is byte-identical");
    }

    {
        const fs::path f = dir / "cancelled.bin";
        log.take();
        auto cancel = [](size_t, size_t) { return false; };
        auto result = HttpClient::download_file(base + "/ranged.bin", f.string(), cancel, {},
                                                parallel_options(), policy);
        log.take();
        r.check(result.cancelled && !result.success, "progress callback cancels a parallel download");
        const std::string partial = read_all(partial_of(f));
        r.check(!fs::exists(f) && partial == payload.substr(0, partial.size()),
                "cancelled download keeps only a byte-exact prefix");
        r.check(!fs::exists(journal_of(f)), "no journal after cancellation");
    }

    {
        const fs::path f = dir / "stall.bin";
        log.take();
        auto options = parallel_options();
        options.no_progress_timeout = 1;
        const auto started = std::chrono::steady_clock::now();
        auto result = HttpClient::download_file(base + "/stall.bin", f.string(), nullptr, {},
                                                options, policy);
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - started).count();
        log.take();
        r.check(!result.success && !result.cancelled &&
                    result.error_message.find("stalled") != std::string::npos,
                "a silent range trips the per-transfer stall timeout");
        r.check(elapsed < 8, "stall is detected before the silent range answers");
        const std::string partial = read_all(partial_of(f));
        r.check(!fs::exists(f) && partial == payload.substr(0, partial.size()),
                "stalled download keeps only a byte-exact prefix");
    }

    server.stop();
    server_thread.join();

    {
        using nlohmann::json;
        auto connections = [](const json& cfg) {
            return lemon::RuntimeConfig(cfg).download_connections();
        };
        auto set_throws = [](const json& change) {
            try {
                lemon::RuntimeConfig(json::object()).set(change);
                return false;
            } catch (const std::invalid_argument&) {
                return true;
            }
        };
        r.check(connections(json::object()) == HttpClient::kDefaultDownloadConnections,
                "download_connections defaults when absent");
        r.check(connections(json{{"download_connections", 3}}) == 3, "download_connections is read");
        r.check(connections(json{{"download_connections", 0}}) ==
                    HttpClient::kDefaultDownloadConnections,
                "out-of-range download_connections in config falls back to the default");
        r.check(!set_throws(json{{"download_connections", 1}}), "set accepts 1");
        r.check(!set_throws(json{{"download_connections", 32}}), "set accepts 32");
        r.check(set_throws(json{{"download_connections", 0}}), "set rejects 0");
        r.check(set_throws(json{{"download_connections", 33}}), "set rejects 33");
        r.check(set_throws(json{{"download_connections", "8"}}), "set rejects a string");
    }

    fs::remove_all(dir);
    printf("\n%d passed, %d failed\n", r.passed, r.failed);
    return r.failed == 0 ? 0 : 1;
}
