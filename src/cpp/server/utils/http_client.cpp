#include <lemon/utils/http_client.h>
#include <lemon/utils/path_utils.h>
#include <lemon/utils/aixlog.hpp>
#include <curl/curl.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>
#include <mbedtls/md.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace lemon {
namespace utils {

std::atomic<bool> g_download_cancelled{false};

std::atomic<long> HttpClient::default_timeout_seconds_{300};

std::atomic<int64_t> HttpClient::download_rate_limit_bytes_per_second_{0};

std::atomic<int> HttpClient::download_connections_{HttpClient::kDefaultDownloadConnections};

// Serializes transfers so concurrent downloads cannot exceed the cap in aggregate.
static std::mutex g_download_gate;

namespace {

// Bounds the TCP handshake so an unroutable host cannot hold a worker for the
// full request timeout.
constexpr long kConnectTimeoutSeconds = 30;

// Resolves the 0-means-default convention shared by every request method.
// Without this, curl reads 0 as "no timeout" and a silent upstream parks the
// calling httplib worker permanently.
long effective_timeout(long timeout_seconds) {
    if (timeout_seconds == HttpClient::kNoTimeout) {
        return 0;
    }
    return timeout_seconds > 0 ? timeout_seconds
                               : HttpClient::get_default_timeout();
}

static std::string trim_copy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n\"'");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n\"'");
    return value.substr(first, last - first + 1);
}

static std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static size_t curl_off_to_size(curl_off_t value) {
    return value > 0 ? static_cast<size_t>(value) : 0;
}

static bool is_hex_digest(const std::string& value, size_t expected_len) {
    if (value.size() != expected_len) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isxdigit(c) != 0;
    });
}

struct ExpectedHash {
    std::string algorithm;
    std::string value;

    bool present() const { return !algorithm.empty() && !value.empty(); }
};

struct HashCheckResult {
    bool ok = false;
    std::string actual;
    std::string error;
};

static ExpectedHash parse_expected_hash(const DownloadOptions& options) {
    std::string algorithm = lower_copy(trim_copy(options.expected_hash_algorithm));
    std::string value = lower_copy(trim_copy(options.expected_hash));

    const auto colon = value.find(':');
    if (colon != std::string::npos) {
        const std::string prefix = lower_copy(trim_copy(value.substr(0, colon)));
        if (!prefix.empty() && algorithm.empty()) {
            algorithm = prefix;
        }
        value = trim_copy(value.substr(colon + 1));
    }

    if (algorithm == "sha-256") algorithm = "sha256";
    if (algorithm == "sha-1") algorithm = "sha1";
    if (algorithm == "gitsha1" || algorithm == "git-sha") algorithm = "git-sha1";

    if (!algorithm.empty() &&
        algorithm != "sha256" &&
        algorithm != "sha1" &&
        algorithm != "git-sha1") {
        return {};
    }

    if (algorithm.empty()) {
        if (is_hex_digest(value, 64)) {
            algorithm = "sha256";
        } else if (is_hex_digest(value, 40)) {
            algorithm = "sha1";
        }
    }

    if ((algorithm == "sha256" && !is_hex_digest(value, 64)) ||
        ((algorithm == "sha1" || algorithm == "git-sha1") && !is_hex_digest(value, 40))) {
        return {};
    }

    return {algorithm, value};
}


static std::string bytes_to_hex(const unsigned char* bytes, size_t len) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        oss << std::setw(2) << static_cast<unsigned int>(bytes[i]);
    }
    return oss.str();
}

static HashCheckResult digest_file_with_library(const fs::path& path,
                                                const ExpectedHash& expected,
                                                const std::string& git_blob_prefix) {
    HashCheckResult result;

    const mbedtls_md_type_t md_type =
        (expected.algorithm == "sha256") ? MBEDTLS_MD_SHA256 :
        (expected.algorithm == "sha1" || expected.algorithm == "git-sha1") ? MBEDTLS_MD_SHA1 :
        MBEDTLS_MD_NONE;

    if (md_type == MBEDTLS_MD_NONE) {
        result.error = "unsupported hash algorithm: " + expected.algorithm;
        return result;
    }

    const mbedtls_md_info_t* md_info = mbedtls_md_info_from_type(md_type);
    if (!md_info) {
        result.error = "mbedTLS digest algorithm is unavailable: " + expected.algorithm;
        return result;
    }

    mbedtls_md_context_t ctx;
    mbedtls_md_init(&ctx);

    auto cleanup = [&ctx]() {
        mbedtls_md_free(&ctx);
    };

    if (mbedtls_md_setup(&ctx, md_info, 0) != 0 ||
        mbedtls_md_starts(&ctx) != 0) {
        cleanup();
        result.error = "failed to initialize mbedTLS digest context";
        return result;
    }

    if (!git_blob_prefix.empty() &&
        mbedtls_md_update(&ctx,
                          reinterpret_cast<const unsigned char*>(git_blob_prefix.data()),
                          git_blob_prefix.size()) != 0) {
        cleanup();
        result.error = "failed to hash Git blob prefix with mbedTLS";
        return result;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        cleanup();
        result.error = "failed to open file for hash verification";
        return result;
    }

    constexpr size_t buffer_size = 1024 * 1024;
    std::vector<unsigned char> buffer(buffer_size);
    while (file.good()) {
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = file.gcount();
        if (count > 0 &&
            mbedtls_md_update(&ctx, buffer.data(), static_cast<size_t>(count)) != 0) {
            cleanup();
            result.error = "failed while hashing file with mbedTLS";
            return result;
        }
    }
    if (file.bad()) {
        cleanup();
        result.error = "failed while reading file for hash verification";
        return result;
    }

    unsigned char digest[MBEDTLS_MD_MAX_SIZE] = {};
    if (mbedtls_md_finish(&ctx, digest) != 0) {
        cleanup();
        result.error = "failed to finalize mbedTLS digest";
        return result;
    }

    const size_t digest_len = static_cast<size_t>(mbedtls_md_get_size(md_info));
    cleanup();

    result.actual = bytes_to_hex(digest, digest_len);
    result.ok = (lower_copy(result.actual) == expected.value);
    return result;
}

static HashCheckResult calculate_file_hash(const fs::path& path, const ExpectedHash& expected) {
    HashCheckResult result;
    if (!expected.present()) {
        result.ok = true;
        return result;
    }

    std::string git_blob_prefix;
    if (expected.algorithm == "git-sha1") {
        std::error_code ec;
        const auto size = fs::file_size(path, ec);
        if (ec) {
            result.error = "failed to get file size for git-sha1 verification: " + ec.message();
            return result;
        }
        git_blob_prefix = "blob " + std::to_string(size) + std::string(1, '\0');
    }

    result = digest_file_with_library(path, expected, git_blob_prefix);
    if (!result.ok && result.error.empty()) {
        result.error = "hash mismatch: expected " + expected.algorithm + ":" + expected.value +
                       ", got " + expected.algorithm + ":" + result.actual;
    }
    return result;
}

static HashCheckResult verify_file_hash(const fs::path& path, const ExpectedHash& expected) {
    auto result = calculate_file_hash(path, expected);
    if (!expected.present() || result.ok) {
        return result;
    }
    LOG(ERROR, "Download") << "Content verification failed for " << path.string()
                            << ": " << result.error << std::endl;
    return result;
}

// A ".verified" sidecar records that the file at this path passed a full content
// verification: algorithm, digest, size and mtime, one per line. A later check
// whose expected digest, size and mtime all match skips the full re-hash — on
// large models that re-hash dominates repeat pulls (hashing runs ~280 MB/s, so
// a 340 GiB model costs ~20 min per pull with no bytes to download). Any
// mismatch, including a touched or replaced file, falls back to the full hash.
// Accepted tradeoff: corruption that preserves both size and mtime (bit rot) is
// no longer caught on repeat pulls; deleting the .verified file forces a full
// re-verification.

static fs::path verified_sidecar_path(const fs::path& file) {
    fs::path p = file;
    p += ".verified";
    return p;
}

static void remove_verified_sidecar(const fs::path& file) {
    std::error_code ec;
    fs::remove(verified_sidecar_path(file), ec);
}

static bool read_file_facts(const fs::path& file, uintmax_t& size, long long& mtime) {
    std::error_code ec;
    size = fs::file_size(file, ec);
    if (ec) {
        return false;
    }
    const auto t = fs::last_write_time(file, ec);
    if (ec) {
        return false;
    }
    mtime = static_cast<long long>(t.time_since_epoch().count());
    return true;
}

static void write_verified_sidecar(const fs::path& file, const ExpectedHash& expected) {
    uintmax_t size = 0;
    long long mtime = 0;
    if (!read_file_facts(file, size, mtime)) {
        return;
    }
    std::ofstream out(verified_sidecar_path(file), std::ios::trunc);
    if (!out.is_open()) {
        LOG(WARNING, "Download") << "Could not record verification sidecar (repeat pulls will re-hash): "
                                 << verified_sidecar_path(file).string() << std::endl;
        return;
    }
    out << expected.algorithm << "\n"
        << expected.value << "\n"
        << size << "\n"
        << mtime << "\n";
}

static bool sidecar_attests(const fs::path& file, const ExpectedHash& expected) {
    std::ifstream in(verified_sidecar_path(file));
    if (!in.is_open()) {
        return false;
    }
    std::string algorithm, digest, size_line, mtime_line;
    if (!std::getline(in, algorithm) || !std::getline(in, digest) ||
        !std::getline(in, size_line) || !std::getline(in, mtime_line)) {
        return false;
    }
    if (algorithm != expected.algorithm || lower_copy(digest) != expected.value) {
        return false;
    }
    uintmax_t size = 0;
    long long mtime = 0;
    if (!read_file_facts(file, size, mtime)) {
        return false;
    }
    try {
        if (std::stoull(size_line) != size || std::stoll(mtime_line) != mtime) {
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

// verify_file_hash plus the sidecar fast path: attested files skip the re-hash;
// full verifications record a fresh attestation, failed ones drop it.
static HashCheckResult verify_file_hash_cached(const fs::path& path, const ExpectedHash& expected) {
    if (sidecar_attests(path, expected)) {
        LOG(INFO, "Download") << "Hash previously verified (sidecar match); skipping re-hash: "
                              << path.string() << std::endl;
        HashCheckResult result;
        result.ok = true;
        result.actual = expected.value;
        return result;
    }
    auto result = verify_file_hash(path, expected);
    if (result.ok && expected.present()) {
        write_verified_sidecar(path, expected);
    } else if (!result.ok) {
        remove_verified_sidecar(path);
    }
    return result;
}

} // namespace

// Callback for writing response data to string
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    std::string* str = static_cast<std::string*>(userp);
    str->append(static_cast<char*>(contents), total_size);
    return total_size;
}

// Collects response headers into HttpResponse::headers, lowercasing names so
// lookups do not depend on the casing a given provider happens to send.
static size_t response_header_callback(char* buffer, size_t size, size_t nitems,
                                       void* userdata) {
    const size_t total = size * nitems;
    auto* headers = static_cast<std::map<std::string, std::string>*>(userdata);
    if (!headers) {
        return total;
    }

    std::string line(buffer, total);
    const size_t colon = line.find(':');
    if (colon == std::string::npos) {
        // Status line, or the blank line ending a header block. A redirect or
        // an informational 1xx starts a fresh block, so drop what came before.
        if (line.rfind("HTTP/", 0) == 0) {
            headers->clear();
        }
        return total;
    }

    std::string name = line.substr(0, colon);
    for (auto& c : name) c = std::tolower(static_cast<unsigned char>(c));

    std::string value = line.substr(colon + 1);
    const size_t first = value.find_first_not_of(" \t");
    const size_t last = value.find_last_not_of(" \t\r\n");
    value = (first == std::string::npos) ? std::string()
                                         : value.substr(first, last - first + 1);

    (*headers)[name] = value;
    return total;
}

// Callback for writing to file
static size_t write_file_callback(void* ptr, size_t size, size_t nmemb, void* stream) {
    size_t written = fwrite(ptr, size, nmemb, static_cast<FILE*>(stream));
    return written;
}

static int cancel_xferinfo_callback(void* clientp, curl_off_t, curl_off_t, curl_off_t,
                                    curl_off_t) {
    auto* flag = static_cast<std::atomic<bool>*>(clientp);
    return (flag && flag->load()) ? 1 : 0;
}

static int stream_cancel_xferinfo_callback(void* clientp, curl_off_t, curl_off_t,
                                           curl_off_t, curl_off_t) {
    auto* should_cancel = static_cast<std::function<bool()>*>(clientp);
    if (!should_cancel || !*should_cancel) {
        return 0;
    }

    try {
        return (*should_cancel)() ? 1 : 0;
    } catch (...) {
        // Never allow a C++ exception to cross libcurl's C callback boundary.
        // Failing closed is safer than leaving an orphaned upstream request.
        return 1;
    }
}

struct ProgressData {
    ProgressCallback callback;
    bool cancelled = false;
    bool stalled = false;
    bool waiting_for_first_byte = false;
    long current_response_code = 0;
    int no_progress_timeout = 0;
    curl_off_t last_downloaded = 0;
    std::chrono::steady_clock::time_point last_progress_time = std::chrono::steady_clock::now();
};

static fs::path get_disk_space_probe_path(const fs::path& output_path) {
    fs::path probe_path = output_path.parent_path();
    if (!probe_path.empty()) {
        return probe_path;
    }

    std::error_code ec;
    fs::path current_path = fs::current_path(ec);
    if (!ec && !current_path.empty()) {
        return current_path;
    }

    return fs::path(".");
}

static void classify_curl_failure(CURLcode res, bool& retryable, bool& permanent) {
    switch (res) {
        case CURLE_COULDNT_CONNECT:
        case CURLE_COULDNT_RESOLVE_HOST:
        case CURLE_COULDNT_RESOLVE_PROXY:
        case CURLE_OPERATION_TIMEDOUT:
        case CURLE_SEND_ERROR:
        case CURLE_RECV_ERROR:
        case CURLE_GOT_NOTHING:
        case CURLE_PARTIAL_FILE:
        case CURLE_SSL_CONNECT_ERROR:
            retryable = true;
            break;
        // A rejected scheme (e.g. an https-only policy hit on an http URL or
        // a disallowed redirect target) and a malformed URL can never
        // succeed on retry. Fail permanently so the outer loop stops
        // immediately and preserves any existing partial file.
        case CURLE_UNSUPPORTED_PROTOCOL:
        case CURLE_URL_MALFORMAT:
            permanent = true;
            retryable = false;
            break;
        default:
            retryable = false;
    }
}

// CURLE_WRITE_ERROR typically means disk full; confirm against free space.
static bool disk_nearly_full(const fs::path& probe_path) {
    std::error_code ec;
    auto si = fs::space(probe_path, ec);
    return !ec && si.available < 1024 * 1024;
}

static size_t header_callback(char* buffer, size_t size, size_t nitems, void* userdata) {
    const size_t total = size * nitems;
    ProgressData* data = static_cast<ProgressData*>(userdata);
    if (!data || total == 0) {
        return total;
    }

    std::string line(buffer, total);

    if (line.rfind("HTTP/", 0) == 0) {
        data->waiting_for_first_byte = false;
        data->current_response_code = 0;
        data->last_progress_time = std::chrono::steady_clock::now();

        std::istringstream iss(line);
        std::string http_version;
        long status_code = 0;
        iss >> http_version >> status_code;
        data->current_response_code = status_code;
        return total;
    }

    if (line == "\r\n" || line == "\n") {
        const bool response_can_have_body =
            data->current_response_code >= 200 &&
            data->current_response_code < 300 &&
            data->current_response_code != 204 &&
            data->current_response_code != 304;

        if (response_can_have_body) {
            data->waiting_for_first_byte = true;
            data->last_progress_time = std::chrono::steady_clock::now();
        }
    }

    return total;
}

static int progress_callback(void* clientp, curl_off_t dltotal, curl_off_t dlnow,
                             curl_off_t ultotal, curl_off_t ulnow) {
    (void)ultotal;
    (void)ulnow;

    if (g_download_cancelled.load()) {
        return 1;  // Abort transfer
    }

    ProgressData* data = static_cast<ProgressData*>(clientp);
    if (!data) return 0;

    if (data->cancelled) {
        return 1;
    }

    if (data->callback &&
        !data->callback(curl_off_to_size(dlnow), curl_off_to_size(dltotal))) {
        data->cancelled = true;
        return 1;
    }

    const auto now = std::chrono::steady_clock::now();
    if (dlnow > data->last_downloaded) {
        data->waiting_for_first_byte = false;
        data->last_downloaded = dlnow;
        data->last_progress_time = now;
    } else if (data->no_progress_timeout > 0) {
        const auto idle_seconds = std::chrono::duration_cast<std::chrono::seconds>(
            now - data->last_progress_time).count();

        const long stall_timeout =
            data->waiting_for_first_byte
                ? static_cast<long>(data->no_progress_timeout) * 5L
                : static_cast<long>(data->no_progress_timeout);

        if (idle_seconds >= stall_timeout) {
            data->stalled = true;
            return 1;
        }
    }

    return 0;
}

namespace {
// Applies scheme and redirect restrictions for a trust boundary. Redirect
// behavior is supplied separately so extending policy enforcement to POST
// paths does not silently change their existing no-redirect semantics.
// Returns false if any security-relevant option fails to apply, allowing each
// caller to fail closed instead of issuing an unrestricted request.
bool apply_http_security_policy(
    CURL* curl,
    HttpSecurityPolicy policy,
    bool follow_redirects) {
    auto set = [curl](CURLoption option, auto value) {
        return curl_easy_setopt(curl, option, value) == CURLE_OK;
    };

    const auto set_proto = [&](CURLoption opt_str, CURLoption opt_bit, const char* str_val, long bit_val) {
        if (set(opt_str, str_val)) {
            return true;
        }
        return set(opt_bit, bit_val);
    };

    const auto apply_protocols = [&](const char* protocols,
                                     const char* redirect_protocols) {
        std::string_view p(protocols);
        std::string_view r(redirect_protocols);
        bool p_has_https = (p.find("https") != std::string_view::npos || p.find("HTTPS") != std::string_view::npos);
        bool p_has_http = (p.find("http,") != std::string_view::npos || p.find("HTTP,") != std::string_view::npos ||
                           p.find(",http") != std::string_view::npos || p.find(",HTTP") != std::string_view::npos ||
                           p == "http" || p == "HTTP");
        bool r_has_https = (r.find("https") != std::string_view::npos || r.find("HTTPS") != std::string_view::npos);
        bool r_has_http = (r.find("http,") != std::string_view::npos || r.find("HTTP,") != std::string_view::npos ||
                           r.find(",http") != std::string_view::npos || r.find(",HTTP") != std::string_view::npos ||
                           r == "http" || r == "HTTP");

        long proto_mask = (p_has_https ? 2L : 0L) | (p_has_http ? 1L : 0L);
        long redir_mask = (r_has_https ? 2L : 0L) | (r_has_http ? 1L : 0L);

        if (!set(CURLOPT_FOLLOWLOCATION, follow_redirects ? 1L : 0L) ||
            !set_proto(CURLOPT_PROTOCOLS_STR, CURLOPT_PROTOCOLS, protocols, proto_mask)) {
            return false;
        }
        if (!follow_redirects) {
            return true;
        }
        return set(CURLOPT_MAXREDIRS, 5L) &&
               set_proto(CURLOPT_REDIR_PROTOCOLS_STR, CURLOPT_REDIR_PROTOCOLS, redirect_protocols, redir_mask);
    };

    switch (policy) {
        case HttpSecurityPolicy::TrustedLoopback:
            // Managed loopback backends are plain HTTP and must never redirect.
            return set(CURLOPT_FOLLOWLOCATION, 0L) &&
                   set_proto(CURLOPT_PROTOCOLS_STR, CURLOPT_PROTOCOLS, "HTTP", 1L /* CURLPROTO_HTTP */);
        case HttpSecurityPolicy::AllowInsecureHttp:
            return apply_protocols("HTTP,HTTPS", "HTTP,HTTPS");
        case HttpSecurityPolicy::ExternalHttpsOnly:
        default:
            return apply_protocols("HTTPS", "HTTPS");
    }
}
} // namespace

HttpResponse HttpClient::get(const std::string& url,
                             const std::map<std::string, std::string>& headers,
                             long timeout_seconds,
                             HttpSecurityPolicy policy) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }

    HttpResponse response;
    std::string response_body;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    if (!apply_http_security_policy(curl, policy, true)) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("Failed to apply HTTP security policy");
    }
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, effective_timeout(timeout_seconds));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, kConnectTimeoutSeconds);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "lemon.cpp/1.0");

    // Add custom headers
    struct curl_slist* header_list = nullptr;
    for (const auto& header : headers) {
        std::string header_str = header.first + ": " + header.second;
        header_list = curl_slist_append(header_list, header_str.c_str());
    }
    if (header_list) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
    }

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        std::string error = "CURL error: " + std::string(curl_easy_strerror(res));
        curl_slist_free_all(header_list);
        curl_easy_cleanup(curl);
        throw std::runtime_error(error);
    }

    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    response.status_code = static_cast<int>(response_code);
    response.body = response_body;

    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);

    return response;
}

HttpResponse HttpClient::post(const std::string& url,
                              const std::string& body,
                              const std::map<std::string, std::string>& headers,
                              long timeout_seconds,
                              HttpSecurityPolicy policy,
                              std::atomic<bool>* cancel_flag) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }

    HttpResponse response;
    std::string response_body;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.data());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(body.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, response_header_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &response.headers);
    if (!apply_http_security_policy(curl, policy, false)) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("Failed to apply HTTP security policy");
    }
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, effective_timeout(timeout_seconds));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, kConnectTimeoutSeconds);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "lemon.cpp/1.0");

    if (cancel_flag) {
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, cancel_xferinfo_callback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, cancel_flag);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    }

    // Add custom headers
    bool has_content_type = false;
    for (const auto& header : headers) {
        std::string key = header.first;
        for (auto& c : key) c = std::tolower(static_cast<unsigned char>(c));
        if (key == "content-type") {
            has_content_type = true;
            break;
        }
    }

    struct curl_slist* header_list = nullptr;
    if (!has_content_type) {
        header_list = curl_slist_append(header_list, "Content-Type: application/json");
    }
    for (const auto& header : headers) {
        std::string header_str = header.first + ": " + header.second;
        header_list = curl_slist_append(header_list, header_str.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        std::string error = "CURL error: " + std::string(curl_easy_strerror(res));
        curl_slist_free_all(header_list);
        curl_easy_cleanup(curl);
        throw std::runtime_error(error);
    }

    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    response.status_code = static_cast<int>(response_code);
    response.body = response_body;

    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);

    return response;
}

HttpResponse HttpClient::post_multipart(const std::string& url,
                                         const std::vector<MultipartField>& fields,
                                         long timeout_seconds,
                                         HttpSecurityPolicy policy) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }

    HttpResponse response;
    std::string response_body;

    curl_mime* mime = curl_mime_init(curl);

    for (const auto& field : fields) {
        curl_mimepart* part = curl_mime_addpart(mime);
        curl_mime_name(part, field.name.c_str());
        curl_mime_data(part, field.data.c_str(), field.data.size());
        if (!field.filename.empty()) {
            curl_mime_filename(part, field.filename.c_str());
        }
        if (!field.content_type.empty()) {
            curl_mime_type(part, field.content_type.c_str());
        }
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    if (!apply_http_security_policy(curl, policy, false)) {
        curl_mime_free(mime);
        curl_easy_cleanup(curl);
        throw std::runtime_error("Failed to apply HTTP security policy");
    }
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, effective_timeout(timeout_seconds));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, kConnectTimeoutSeconds);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "lemon.cpp/1.0");

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        std::string error = "CURL error: " + std::string(curl_easy_strerror(res));
        curl_mime_free(mime);
        curl_easy_cleanup(curl);
        throw std::runtime_error(error);
    }

    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    response.status_code = static_cast<int>(response_code);
    response.body = response_body;

    curl_mime_free(mime);
    curl_easy_cleanup(curl);

    return response;
}

// Helper struct to pass stream callback through C interface
struct StreamCallbackData {
    StreamCallback* callback;
    std::string* buffer;
    CURL* curl = nullptr;
    std::function<void(int)>* on_status = nullptr;
    bool status_reported = false;
};

// Static C-style callback function
static size_t stream_write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    try {
        StreamCallbackData* data = static_cast<StreamCallbackData*>(userdata);
        size_t total_size = size * nmemb;

        if (!data || !data->callback || !*(data->callback)) {
            LOG(ERROR, "HttpClient") << "Callback data is null!" << std::endl;
            return 0;
        }

        if (!data->status_reported && data->on_status && *(data->on_status) && data->curl) {
            long code = 0;
            curl_easy_getinfo(data->curl, CURLINFO_RESPONSE_CODE, &code);
            (*(data->on_status))(static_cast<int>(code));
            data->status_reported = true;
        }

        if (!(*(data->callback))(ptr, total_size)) {
            return 0; // Signal error to stop transfer
        }

        return total_size;
    } catch (const std::exception& e) {
        LOG(ERROR, "HttpClient") << "Exception in stream callback: " << e.what() << std::endl;
        return 0;
    } catch (...) {
        LOG(ERROR, "HttpClient") << "Unknown exception in stream callback" << std::endl;
        return 0;
    }
}

HttpResponse HttpClient::post_stream(const std::string& url,
                                     const std::string& body,
                                     StreamCallback stream_callback,
                                     const std::map<std::string, std::string>& headers,
                                     long timeout_seconds,
                                     std::function<void(int)> on_status,
                                     HttpSecurityPolicy policy,
                                     std::function<bool()> should_cancel,
                                     std::map<std::string, std::string>* out_response_headers) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }

    HttpResponse response;

    // Create callback data
    StreamCallbackData callback_data;
    callback_data.callback = &stream_callback;
    callback_data.buffer = nullptr;
    callback_data.curl = curl;
    callback_data.on_status = &on_status;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.data());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(body.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stream_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &callback_data);
    if (out_response_headers) {
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, response_header_callback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, out_response_headers);
    }
    if (!apply_http_security_policy(curl, policy, false)) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("Failed to apply HTTP security policy");
    }
    // A total timeout would kill a long but healthy generation, so the timeout
    // bounds upstream silence instead of duration.
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, effective_timeout(timeout_seconds));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, kConnectTimeoutSeconds);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "lemon.cpp/1.0");

    if (should_cancel) {
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, stream_cancel_xferinfo_callback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &should_cancel);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    }

    // Add custom headers
    bool has_content_type = false;
    for (const auto& header : headers) {
        std::string key = header.first;
        for (auto& c : key) c = std::tolower(static_cast<unsigned char>(c));
        if (key == "content-type") {
            has_content_type = true;
            break;
        }
    }

    struct curl_slist* header_list = nullptr;
    if (!has_content_type) {
        header_list = curl_slist_append(header_list, "Content-Type: application/json");
    }
    for (const auto& header : headers) {
        std::string header_str = header.first + ": " + header.second;
        header_list = curl_slist_append(header_list, header_str.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);

    CURLcode res = curl_easy_perform(curl);

    // Get response code before checking for errors
    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    response.status_code = static_cast<int>(response_code);
    response.curl_code = static_cast<int>(res);
    response.curl_error = (res == CURLE_OK) ? std::string() : std::string(curl_easy_strerror(res));

    // For streaming, preserve transport codes that the stream layer needs
    // to classify. CURLE_ABORTED_BY_CALLBACK is the expected result when the
    // downstream cancellation predicate fires before the backend emits data.
    if (res != CURLE_OK &&
        res != CURLE_PARTIAL_FILE &&
        res != CURLE_RECV_ERROR &&
        res != CURLE_WRITE_ERROR &&
        res != CURLE_ABORTED_BY_CALLBACK) {
        std::string error = "CURL error: " + response.curl_error;
        LOG(ERROR, "HttpClient") << "" << error << std::endl;
        curl_slist_free_all(header_list);
        curl_easy_cleanup(curl);
        throw std::runtime_error(error);
    }

    // Log if we got a non-OK CURL code but continue so the stream layer can make
    // the protocol-aware decision.
    if (res != CURLE_OK) {
        LOG(WARNING, "HttpClient") << "Stream ended with: " << response.curl_error
                  << " (response code: " << response_code << ")" << std::endl;
    }

    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);

    return response;
}

namespace {

FILE* open_binary_file(const fs::path& path, const char* mode) {
#ifdef _WIN32
    const std::string narrow_mode(mode);
    const std::wstring wide_mode(narrow_mode.begin(), narrow_mode.end());
    return _wfopen(path.c_str(), wide_mode.c_str());
#else
    return fopen(path.c_str(), mode);
#endif
}

// Power-loss ordering: the journal may only claim bytes that are already on
// the device, and it must itself be on the device before any out-of-order byte
// is written. Without these syncs a restart could trust zero-filled holes.
bool sync_stdio_file(FILE* fp) {
    if (fflush(fp) != 0) {
        return false;
    }
#ifdef _WIN32
    return _commit(_fileno(fp)) == 0;
#elif defined(__APPLE__)
    return fsync(fileno(fp)) == 0;
#else
    return fdatasync(fileno(fp)) == 0;
#endif
}

bool sync_path(const fs::path& path) {
    FILE* fp = open_binary_file(path, "r+b");
    if (!fp) {
        return false;
    }
    const bool synced = sync_stdio_file(fp);
    return (fclose(fp) == 0) && synced;
}

bool sync_parent_directory(const fs::path& path) {
#ifdef _WIN32
    (void)path;
    return true;
#else
    fs::path dir = path.parent_path();
    if (dir.empty()) {
        dir = ".";
    }
    const int fd = ::open(dir.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    const bool synced = ::fsync(fd) == 0;
    ::close(fd);
    return synced;
#endif
}

// Returns false only when the rename itself failed; *synced reports whether
// the new directory entry is known to be on the device.
bool rename_with_sync(const fs::path& from, const fs::path& to, bool* synced) {
#ifdef _WIN32
    const bool moved = MoveFileExW(from.c_str(), to.c_str(),
                                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
    *synced = moved;
    return moved;
#else
    std::error_code ec;
    fs::rename(from, to, ec);
    if (ec) {
        *synced = false;
        return false;
    }
    *synced = sync_parent_directory(to);
    return true;
#endif
}

bool durable_rename(const fs::path& from, const fs::path& to) {
    bool synced = false;
    return rename_with_sync(from, to, &synced) && synced;
}

// Returns false only when the file could not be removed. The directory sync
// is best effort: a removal lost to power failure resurrects a journal, which
// can only discard progress, never vouch for missing bytes. Windows offers no
// directory sync, so there the delete reaches the disk with NTFS's own log.
bool durable_remove(const fs::path& path) {
    std::error_code ec;
    fs::remove(path, ec);
    if (ec) {
        return false;
    }
    sync_parent_directory(path);
    return true;
}

} // namespace

DownloadResult HttpClient::download_attempt(const std::string& url,
                                            const std::string& output_path,
                                            size_t resume_from,
                                            ProgressCallback callback,
                                            const std::map<std::string, std::string>& headers,
                                            const DownloadOptions& options,
                                            bool initial_range_request,
                                            HttpSecurityPolicy policy) {
    DownloadResult result;

    CURL* curl = curl_easy_init();
    if (!curl) {
        result.error_message = "Failed to initialize CURL";
        return result;
    }

    const char* mode = (resume_from > 0) ? "ab" : "wb";
    fs::path output_path_fs = path_from_utf8(output_path);
#ifdef _WIN32
    std::wstring wide_mode = (resume_from > 0) ? L"ab" : L"wb";
    FILE* fp = _wfopen(output_path_fs.c_str(), wide_mode.c_str());
#else
    FILE* fp = fopen(output_path.c_str(), mode);
#endif
    if (!fp) {
        result.error_message = "Failed to open file for writing: " + output_path;
        curl_easy_cleanup(curl);
        return result;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_file_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    if (!apply_http_security_policy(curl, policy, true)) {
        result.error_message = "Failed to apply HTTP security policy";
        fclose(fp);
        curl_easy_cleanup(curl);
        return result;
    }
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "lemon.cpp/1.0");
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, static_cast<long>(options.connect_timeout));
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, static_cast<long>(options.low_speed_limit));
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, static_cast<long>(options.low_speed_time));

    const int64_t rate_limit = download_rate_limit_bytes_per_second_.load();
    if (rate_limit > 0) {
        curl_easy_setopt(curl, CURLOPT_MAX_RECV_SPEED_LARGE, static_cast<curl_off_t>(rate_limit));
    }

    if (resume_from > 0) {
        curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(resume_from));
    } else if (initial_range_request) {
        curl_easy_setopt(curl, CURLOPT_RANGE, "0-");
    }

    const int no_progress_timeout = options.no_progress_timeout;
    std::unique_ptr<ProgressData> prog_data;
    if (callback || no_progress_timeout > 0) {
        prog_data = std::make_unique<ProgressData>();
        prog_data->callback = callback;
        prog_data->no_progress_timeout = no_progress_timeout;
        prog_data->last_progress_time = std::chrono::steady_clock::now();
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_callback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, prog_data.get());
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, prog_data.get());
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    }

    // Add custom headers including authentication
    struct curl_slist* header_list = nullptr;
    for (const auto& header : headers) {
        std::string header_str = header.first + ": " + header.second;
        header_list = curl_slist_append(header_list, header_str.c_str());
    }
    if (header_list) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
    }

    CURLcode res = curl_easy_perform(curl);

    bool was_cancelled = (prog_data && prog_data->cancelled);
    bool was_stalled = (prog_data && prog_data->stalled);

    curl_off_t downloaded = 0;
    curl_off_t total = 0;
    curl_easy_getinfo(curl, CURLINFO_SIZE_DOWNLOAD_T, &downloaded);
    curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &total);

    result.bytes_downloaded = static_cast<size_t>(downloaded);
    result.total_bytes = (total > 0) ? static_cast<size_t>(total) : 0;

    // A completed transfer must be on the device before the caller renames
    // it, or a power loss can leave a full-size partial of unwritten bytes.
    const bool data_synced = (res != CURLE_OK) || sync_stdio_file(fp);
    const bool closed_cleanly = fclose(fp) == 0;
    curl_slist_free_all(header_list);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &result.http_code);

    result.curl_code = static_cast<int>(res);
    result.curl_error = curl_easy_strerror(res);

    curl_easy_cleanup(curl);

    if (was_stalled) {
        size_t current_file_size = 0;
        if (fs::exists(output_path_fs)) {
            current_file_size = fs::file_size(output_path_fs);
        }

        result.cancelled = false;
        result.can_resume = current_file_size > 0;
        std::ostringstream oss;
        oss << "Download stalled: no bytes received for "
            << no_progress_timeout << " seconds";
        if (current_file_size > 0) {
            oss << "\n  Partial file size: " << (current_file_size / (1024.0 * 1024.0)) << " MB (resumable)";
        }
        result.error_message = oss.str();
        return result;
    }

    // Handle user cancellation
    if (was_cancelled || res == CURLE_ABORTED_BY_CALLBACK) {
        result.cancelled = true;
        result.error_message = "Download cancelled by user";
        result.can_resume = true;  // Partial file can be resumed later
        return result;
    }

    if (res != CURLE_OK) {
        bool retryable = false;
        bool disk_full = false;
        const fs::path disk_space_probe_path = get_disk_space_probe_path(output_path_fs);
        if (res == CURLE_WRITE_ERROR) {
            disk_full = disk_nearly_full(disk_space_probe_path);
        } else {
            classify_curl_failure(res, retryable, result.permanent);
        }

        size_t current_file_size = 0;
        if (fs::exists(output_path_fs)) {
            current_file_size = fs::file_size(output_path_fs);
        }
        result.can_resume = retryable && (current_file_size > 0);
        result.disk_full = disk_full;

        std::ostringstream oss;
        if (disk_full) {
            oss << "Disk full: not enough space to complete download";
            std::error_code ec;
            auto si = fs::space(disk_space_probe_path, ec);
            if (!ec) {
                oss << " (" << std::fixed << std::setprecision(1)
                    << (si.available / (1024.0 * 1024.0)) << " MB free)";
            }
        } else {
            oss << "Download failed: " << result.curl_error << " (CURL code: " << result.curl_code << ")";
        }
        if (result.bytes_downloaded > 0) {
            oss << "\n  Downloaded " << (result.bytes_downloaded / (1024.0 * 1024.0)) << " MB before failure";
        }
        if (current_file_size > 0) {
            oss << "\n  Partial file size: " << (current_file_size / (1024.0 * 1024.0)) << " MB";
            if (result.can_resume) {
                oss << " (resumable)";
            }
        }
        result.error_message = oss.str();
        return result;
    }

    if (result.http_code >= 400) {
        // HTTP 416 (Range Not Satisfiable) when resuming - verify if file is actually complete
        if (result.http_code == 416 && resume_from > 0) {
            // Do a HEAD request to get the actual file size
            CURL* head_curl = curl_easy_init();
            if (head_curl) {
                curl_easy_setopt(head_curl, CURLOPT_URL, url.c_str());
                curl_easy_setopt(head_curl, CURLOPT_NOBODY, 1L);  // HEAD request
                if (!apply_http_security_policy(head_curl, policy, true)) {
                    curl_easy_cleanup(head_curl);
                    result.error_message = "Resume failed (HTTP 416) - could not apply security policy";
                    result.can_resume = false;
                    return result;
                }
                curl_easy_setopt(head_curl, CURLOPT_TIMEOUT, 30L);

                // Add headers
                struct curl_slist* head_headers = nullptr;
                for (const auto& header : headers) {
                    std::string header_str = header.first + ": " + header.second;
                    head_headers = curl_slist_append(head_headers, header_str.c_str());
                }
                if (head_headers) {
                    curl_easy_setopt(head_curl, CURLOPT_HTTPHEADER, head_headers);
                }

                CURLcode head_res = curl_easy_perform(head_curl);

                if (head_res == CURLE_OK) {
                    curl_off_t remote_size = 0;
                    curl_easy_getinfo(head_curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &remote_size);

                    curl_slist_free_all(head_headers);
                    curl_easy_cleanup(head_curl);

                    if (remote_size > 0 && static_cast<size_t>(remote_size) <= resume_from) {
                        // Local file is >= remote size, file is complete
                        LOG(INFO, "Download") << " File verified complete (local: "
                                  << (resume_from / (1024.0 * 1024.0)) << " MB, remote: "
                                  << (remote_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
                        result.success = true;
                        result.bytes_downloaded = 0;
                        return result;
                    } else {
                        // Local file is smaller than remote, or size unknown - corrupted or needs restart
                        std::ostringstream oss;
                        oss << "Resume failed - local file (" << (resume_from / (1024.0 * 1024.0))
                            << " MB) doesn't match remote size (" << (remote_size / (1024.0 * 1024.0)) << " MB)";
                        result.error_message = oss.str();
                        result.can_resume = false;  // Force fresh download
                        return result;
                    }
                }

                curl_slist_free_all(head_headers);
                curl_easy_cleanup(head_curl);
            }

            // HEAD request failed, treat 416 as needing restart
            result.error_message = "Resume failed (HTTP 416) - could not verify file size";
            result.can_resume = false;
            return result;
        }

        std::ostringstream oss;
        oss << "HTTP error " << result.http_code << " for URL: " << url;
        result.error_message = oss.str();
        result.can_resume = false;
        return result;
    }

    if (!data_synced || !closed_cleanly) {
        result.error_message = "Download finished but could not be flushed to disk: " + output_path;
        result.curl_error = result.error_message;
        result.can_resume = true;
        return result;
    }

    result.success = true;
    return result;
}

namespace {

constexpr uint64_t kMaxRangeChunkBytes = 256ULL * 1024 * 1024;
constexpr auto kParallelProgressInterval = std::chrono::milliseconds(100);

// A parallel download writes ranges out of order, so the .partial file's size
// stops meaning "bytes written from the start". While "<partial>.ranges"
// exists it records the contiguous prefix that is known to be written; a
// resume truncates back to that prefix (normalize_partial_file) so every
// size-based resume and progress reading stays truthful.
fs::path range_journal_path(const fs::path& partial) {
    fs::path p = partial;
    p += ".ranges";
    return p;
}

std::optional<uint64_t> read_range_journal(const fs::path& journal) {
    std::ifstream in(journal);
    std::string line;
    if (!in.is_open() || !std::getline(in, line)) {
        return std::nullopt;
    }
    try {
        size_t consumed = 0;
        const unsigned long long value = std::stoull(line, &consumed);
        if (consumed != line.size()) {
            return std::nullopt;
        }
        return static_cast<uint64_t>(value);
    } catch (...) {
        return std::nullopt;
    }
}

bool write_range_journal(const fs::path& journal, uint64_t prefix) {
    fs::path tmp = journal;
    tmp += ".tmp";
    FILE* fp = open_binary_file(tmp, "wb");
    if (!fp) {
        return false;
    }
    const std::string line = std::to_string(prefix) + "\n";
    bool ok = fwrite(line.data(), 1, line.size(), fp) == line.size() && sync_stdio_file(fp);
    ok = (fclose(fp) == 0) && ok;
    return ok && durable_rename(tmp, journal);
}

// Returns false if an out-of-order partial could not be reset; the caller must
// not resume from it. The truncation is synced before the journal goes, so a
// power loss can never leave the old tail without its journal.
bool normalize_partial_file(const fs::path& partial) {
    const fs::path journal = range_journal_path(partial);
    std::error_code ec;
    fs::path tmp = journal;
    tmp += ".tmp";
    fs::remove(tmp, ec);
    if (!fs::exists(journal, ec)) {
        return true;
    }
    if (fs::exists(partial, ec)) {
        const uint64_t prefix = read_range_journal(journal).value_or(0);
        const uint64_t size = fs::file_size(partial, ec);
        if (ec) {
            return false;
        }
        if (size > prefix) {
            fs::resize_file(partial, prefix, ec);
            if (ec || !sync_path(partial)) {
                std::error_code remove_ec;
                fs::remove(partial, remove_ec);
                if (remove_ec) {
                    return false;
                }
            }
        }
    }
    return durable_remove(journal);
}

bool seek_binary_file(FILE* fp, uint64_t offset) {
#ifdef _WIN32
    return _fseeki64(fp, static_cast<__int64>(offset), SEEK_SET) == 0;
#else
    return fseeko(fp, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}

bool parse_u64(const std::string& text, uint64_t& out) {
    if (text.empty() || !std::all_of(text.begin(), text.end(),
                                     [](unsigned char c) { return std::isdigit(c) != 0; })) {
        return false;
    }
    try {
        out = static_cast<uint64_t>(std::stoull(text));
        return true;
    } catch (...) {
        return false;
    }
}

// Final-response status and Content-Range ("bytes S-E/T"). Reset on every
// status line so a redirect hop's headers never describe the final body.
struct RangeHeaderState {
    long status = 0;
    bool have_content_range = false;
    uint64_t range_start = 0;
    uint64_t range_end = 0;
    uint64_t range_total = 0;
};

size_t range_header_callback(char* buffer, size_t size, size_t nitems, void* userdata) {
    const size_t total = size * nitems;
    auto* state = static_cast<RangeHeaderState*>(userdata);
    if (!state) {
        return total;
    }

    std::string line(buffer, total);
    if (line.rfind("HTTP/", 0) == 0) {
        *state = RangeHeaderState{};
        std::istringstream iss(line);
        std::string version;
        iss >> version >> state->status;
        return total;
    }

    const size_t colon = line.find(':');
    if (colon == std::string::npos ||
        lower_copy(trim_copy(line.substr(0, colon))) != "content-range") {
        return total;
    }

    const std::string value = lower_copy(trim_copy(line.substr(colon + 1)));
    const std::string unit = "bytes ";
    const size_t dash = value.find('-');
    const size_t slash = value.find('/');
    if (value.rfind(unit, 0) != 0 || dash == std::string::npos || slash == std::string::npos ||
        dash > slash) {
        return total;
    }
    RangeHeaderState parsed = *state;
    if (parse_u64(value.substr(unit.size(), dash - unit.size()), parsed.range_start) &&
        parse_u64(value.substr(dash + 1, slash - dash - 1), parsed.range_end) &&
        parse_u64(value.substr(slash + 1), parsed.range_total)) {
        parsed.have_content_range = true;
        *state = parsed;
    }
    return total;
}

struct RangeProbe {
    bool supported = false;
    bool cancelled = false;
    uint64_t total = 0;
};

struct ProbeCancel {
    const ProgressCallback* callback = nullptr;
    size_t resume_offset = 0;
    bool cancelled = false;
};

int probe_xferinfo_callback(void* clientp, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    auto* cancel = static_cast<ProbeCancel*>(clientp);
    if (g_download_cancelled.load()) {
        cancel->cancelled = true;
        return 1;
    }
    try {
        if (*cancel->callback && !(*cancel->callback)(cancel->resume_offset, 0)) {
            cancel->cancelled = true;
            return 1;
        }
    } catch (...) {
        cancel->cancelled = true;
        return 1;
    }
    return 0;
}

size_t probe_write_callback(char*, size_t size, size_t nmemb, void* userdata) {
    auto* received = static_cast<size_t*>(userdata);
    *received += size * nmemb;
    // A server that ignores Range starts sending the whole file; stop it.
    return *received <= 1 ? size * nmemb : 0;
}

struct HeaderList {
    curl_slist* list = nullptr;
    explicit HeaderList(const std::map<std::string, std::string>& headers) {
        for (const auto& header : headers) {
            const std::string line = header.first + ": " + header.second;
            list = curl_slist_append(list, line.c_str());
        }
    }
    ~HeaderList() { curl_slist_free_all(list); }
    HeaderList(const HeaderList&) = delete;
    HeaderList& operator=(const HeaderList&) = delete;
};

RangeProbe probe_range_support(const std::string& url,
                               const std::map<std::string, std::string>& headers,
                               const DownloadOptions& options,
                               HttpSecurityPolicy policy,
                               const ProgressCallback& callback,
                               size_t resume_offset) {
    RangeProbe probe;
    if (g_download_cancelled.load()) {
        probe.cancelled = true;
        return probe;
    }
    CURL* curl = curl_easy_init();
    if (!curl) {
        return probe;
    }
    RangeHeaderState state;
    size_t received = 0;
    HeaderList header_list(headers);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    if (!apply_http_security_policy(curl, policy, true)) {
        curl_easy_cleanup(curl);
        return probe;
    }
    curl_easy_setopt(curl, CURLOPT_RANGE, "0-0");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, probe_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &received);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, range_header_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &state);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, static_cast<long>(options.connect_timeout));
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "lemon.cpp/1.0");
    if (header_list.list) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list.list);
    }
    ProbeCancel cancel;
    cancel.callback = &callback;
    cancel.resume_offset = resume_offset;
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, probe_xferinfo_callback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &cancel);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

    const CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    if (cancel.cancelled) {
        probe.cancelled = true;
        return probe;
    }

    if (res == CURLE_OK && state.status == 206 && state.have_content_range &&
        state.range_start == 0 && state.range_end == 0 && state.range_total > 0) {
        probe.supported = true;
        probe.total = state.range_total;
    }
    return probe;
}

struct RangeTransfer {
    CURL* easy = nullptr;
    FILE* fp = nullptr;
    size_t chunk = 0;
    uint64_t start = 0;
    uint64_t length = 0;
    uint64_t total_size = 0;
    uint64_t written = 0;
    RangeHeaderState headers;
    bool body_checked = false;
    bool range_ignored = false;
    bool bad_range = false;
    bool write_failed = false;
    uint64_t last_written = 0;
    std::chrono::steady_clock::time_point last_progress = std::chrono::steady_clock::now();
};

size_t range_write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* t = static_cast<RangeTransfer*>(userdata);
    const size_t n = size * nmemb;
    const RangeHeaderState& h = t->headers;
    if (h.status < 200 || h.status >= 300) {
        return n;  // Error body; the status decides the outcome.
    }
    if (!t->body_checked) {
        t->body_checked = true;
        if (h.status != 206) {
            t->range_ignored = true;
            return 0;
        }
        if (!h.have_content_range || h.range_start != t->start ||
            h.range_end != t->start + t->length - 1 || h.range_total != t->total_size) {
            t->bad_range = true;
            return 0;
        }
    }
    if (t->written + n > t->length) {
        t->bad_range = true;
        return 0;
    }
    if (fwrite(ptr, 1, n, t->fp) != n) {
        t->write_failed = true;
        return 0;
    }
    t->written += n;
    return n;
}

} // namespace

size_t HttpClient::resumable_partial_bytes(const std::string& output_path) {
    const fs::path partial = path_from_utf8(output_path + ".partial");
    std::error_code ec;
    const uintmax_t size = fs::file_size(partial, ec);
    if (ec) {
        return 0;
    }
    const fs::path journal = range_journal_path(partial);
    if (fs::exists(journal, ec)) {
        return static_cast<size_t>((std::min)(static_cast<uint64_t>(size),
                                              read_range_journal(journal).value_or(0)));
    }
    return static_cast<size_t>(size);
}

DownloadResult HttpClient::parallel_download_attempt(const std::string& url,
                                                     const std::string& partial_path,
                                                     size_t resume_from,
                                                     size_t total_size,
                                                     int connections,
                                                     ProgressCallback callback,
                                                     const std::map<std::string, std::string>& headers,
                                                     const DownloadOptions& options,
                                                     HttpSecurityPolicy policy,
                                                     bool& fall_back_single_stream) {
    DownloadResult result;
    fall_back_single_stream = false;
    const fs::path partial_fs = path_from_utf8(partial_path);
    const fs::path journal = range_journal_path(partial_fs);
    const uint64_t remaining = static_cast<uint64_t>(total_size) - resume_from;
    result.total_bytes = static_cast<size_t>(remaining);

    struct FileCloser {
        FILE* fp = nullptr;
        void close() {
            if (fp) {
                fclose(fp);
                fp = nullptr;
            }
        }
        ~FileCloser() { close(); }
    } sync_fp_closer;

    // Local setup failures leave the partial untouched (or honestly
    // normalized), so its bytes stay resumable.
    auto local_failure = [&](const std::string& message) {
        sync_fp_closer.close();
        result.error_message = message;
        result.curl_error = message;
        result.can_resume = normalize_partial_file(partial_fs);
        return result;
    };

    FILE* sync_fp = open_binary_file(partial_fs, resume_from == 0 ? "wb" : "r+b");
    if (!sync_fp) {
        return local_failure("Failed to open file for writing: " + partial_path);
    }
    sync_fp_closer.fp = sync_fp;

    // The resumed prefix and then the journal reach the device before the
    // first out-of-order byte can land.
    if (!sync_stdio_file(sync_fp) || !write_range_journal(journal, resume_from)) {
        return local_failure("Failed to record download progress: " + journal.string());
    }

    const uint64_t min_chunk = (std::max<uint64_t>)(1, options.parallel_min_bytes / 4);
    uint64_t chunk_bytes = remaining / (static_cast<uint64_t>(connections) * 4);
    chunk_bytes = (std::max)(min_chunk, (std::min)(chunk_bytes, kMaxRangeChunkBytes));
    std::vector<std::pair<uint64_t, uint64_t>> chunks;
    for (uint64_t offset = resume_from; offset < total_size; offset += chunk_bytes) {
        chunks.emplace_back(offset, (std::min)(chunk_bytes, static_cast<uint64_t>(total_size) - offset));
    }
    std::vector<bool> chunk_done(chunks.size(), false);
    size_t next_chunk = 0;
    size_t prefix_chunks = 0;
    uint64_t prefix = resume_from;
    uint64_t completed_bytes = 0;

    CURLM* multi = curl_multi_init();
    if (!multi) {
        return local_failure("Failed to initialize CURL multi handle");
    }
    // Multiplexing would put every range on one HTTP/2 connection and share
    // one flow-control window, which is the single-stream bottleneck again.
    curl_multi_setopt(multi, CURLMOPT_PIPELINING, static_cast<long>(CURLPIPE_NOTHING));

    HeaderList header_list(headers);
    std::vector<std::unique_ptr<RangeTransfer>> active;

    bool failed = false;
    bool cancelled = false;
    bool stalled = false;
    bool write_failed = false;
    bool bad_range = false;
    bool setup_failed = false;
    CURLcode fail_code = CURLE_OK;
    long fail_http = 0;
    std::string fail_message;

    auto release = [&](RangeTransfer* t) {
        if (t->easy) {
            curl_multi_remove_handle(multi, t->easy);
            curl_easy_cleanup(t->easy);
            t->easy = nullptr;
        }
        if (t->fp) {
            fclose(t->fp);
            t->fp = nullptr;
        }
    };

    auto start_transfer = [&](size_t index) -> bool {
        auto t = std::make_unique<RangeTransfer>();
        t->chunk = index;
        t->start = chunks[index].first;
        t->length = chunks[index].second;
        t->total_size = total_size;
        t->fp = open_binary_file(partial_fs, "r+b");
        if (!t->fp || !seek_binary_file(t->fp, t->start)) {
            fail_message = "Failed to open file for writing: " + partial_path;
            release(t.get());
            return false;
        }
        t->easy = curl_easy_init();
        if (!t->easy) {
            fail_message = "Failed to initialize CURL";
            release(t.get());
            return false;
        }
        CURL* easy = t->easy;
        curl_easy_setopt(easy, CURLOPT_URL, url.c_str());
        if (!apply_http_security_policy(easy, policy, true)) {
            fail_message = "Failed to apply HTTP security policy";
            release(t.get());
            return false;
        }
        const std::string range = std::to_string(t->start) + "-" +
                                  std::to_string(t->start + t->length - 1);
        curl_easy_setopt(easy, CURLOPT_RANGE, range.c_str());
        curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, range_write_callback);
        curl_easy_setopt(easy, CURLOPT_WRITEDATA, t.get());
        curl_easy_setopt(easy, CURLOPT_HEADERFUNCTION, range_header_callback);
        curl_easy_setopt(easy, CURLOPT_HEADERDATA, &t->headers);
        curl_easy_setopt(easy, CURLOPT_PRIVATE, t.get());
        curl_easy_setopt(easy, CURLOPT_TIMEOUT, 0L);
        curl_easy_setopt(easy, CURLOPT_USERAGENT, "lemon.cpp/1.0");
        curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT, static_cast<long>(options.connect_timeout));
        curl_easy_setopt(easy, CURLOPT_LOW_SPEED_LIMIT, static_cast<long>(options.low_speed_limit));
        curl_easy_setopt(easy, CURLOPT_LOW_SPEED_TIME, static_cast<long>(options.low_speed_time));
        if (header_list.list) {
            curl_easy_setopt(easy, CURLOPT_HTTPHEADER, header_list.list);
        }
        if (curl_multi_add_handle(multi, easy) != CURLM_OK) {
            fail_message = "Failed to start range request";
            curl_easy_cleanup(easy);
            t->easy = nullptr;
            release(t.get());
            return false;
        }
        active.push_back(std::move(t));
        return true;
    };

    auto finish_transfer = [&](RangeTransfer* t, CURLcode res) {
        long http_code = 0;
        curl_easy_getinfo(t->easy, CURLINFO_RESPONSE_CODE, &http_code);
        const bool closed_cleanly = (fclose(t->fp) == 0);
        t->fp = nullptr;
        const bool ok = res == CURLE_OK && http_code == 206 && t->body_checked &&
                        !t->bad_range && t->written == t->length && closed_cleanly;
        if (ok) {
            chunk_done[t->chunk] = true;
            completed_bytes += t->length;
            const uint64_t old_prefix = prefix;
            while (prefix_chunks < chunks.size() && chunk_done[prefix_chunks]) {
                prefix += chunks[prefix_chunks].second;
                ++prefix_chunks;
            }
            // A skipped or failed journal update only understates the prefix,
            // which is safe.
            if (prefix != old_prefix && sync_stdio_file(sync_fp)) {
                write_range_journal(journal, prefix);
            }
        } else if (!failed) {
            failed = true;
            fail_http = http_code;
            if (t->range_ignored) {
                fall_back_single_stream = true;
                fail_message = "Server ignored a Range request (HTTP " +
                               std::to_string(http_code) + "); retrying as a single stream";
            } else if (http_code == 416) {
                // One edge node refusing a range must not cost the journaled
                // prefix; the single-stream path re-checks the remote size.
                fall_back_single_stream = true;
                fail_http = 0;
                fail_message = "Server rejected a Range request (HTTP 416); retrying as a single stream";
            } else if (t->write_failed || !closed_cleanly) {
                write_failed = true;
            } else if (res != CURLE_OK && !t->bad_range) {
                fail_code = res;
            } else if (http_code < 400) {
                bad_range = true;
                std::ostringstream oss;
                oss << "Range response for bytes " << t->start << "-" << (t->start + t->length - 1)
                    << " did not match the request (HTTP " << http_code << ", received "
                    << t->written << " of " << t->length << " bytes)";
                fail_message = oss.str();
            }
        }
        release(t);
    };

    const int no_progress_timeout = options.no_progress_timeout;
    auto last_report = std::chrono::steady_clock::now() - kParallelProgressInterval;

    while (true) {
        while (!failed && active.size() < static_cast<size_t>(connections) &&
               next_chunk < chunks.size()) {
            if (!start_transfer(next_chunk)) {
                failed = true;
                setup_failed = true;
                break;
            }
            ++next_chunk;
        }
        if (failed || active.empty()) {
            break;
        }

        int running = 0;
        const CURLMcode perform_code = curl_multi_perform(multi, &running);
        if (perform_code != CURLM_OK) {
            failed = true;
            setup_failed = true;
            fail_message = std::string("CURL multi error: ") + curl_multi_strerror(perform_code);
            break;
        }

        int queued = 0;
        while (CURLMsg* msg = curl_multi_info_read(multi, &queued)) {
            if (msg->msg != CURLMSG_DONE) {
                continue;
            }
            char* priv = nullptr;
            curl_easy_getinfo(msg->easy_handle, CURLINFO_PRIVATE, &priv);
            auto* t = reinterpret_cast<RangeTransfer*>(priv);
            const CURLcode res = msg->data.result;
            finish_transfer(t, res);
            active.erase(std::remove_if(active.begin(), active.end(),
                                        [t](const std::unique_ptr<RangeTransfer>& p) {
                                            return p.get() == t;
                                        }),
                         active.end());
        }
        if (failed) {
            break;
        }

        const auto now = std::chrono::steady_clock::now();
        uint64_t in_flight = 0;
        for (auto& t : active) {
            in_flight += t->written;
            if (t->written > t->last_written) {
                t->last_written = t->written;
                t->last_progress = now;
            } else if (no_progress_timeout > 0) {
                const long limit = t->written == 0 ? static_cast<long>(no_progress_timeout) * 5L
                                                   : static_cast<long>(no_progress_timeout);
                const auto idle = std::chrono::duration_cast<std::chrono::seconds>(
                    now - t->last_progress).count();
                if (idle >= limit) {
                    stalled = true;
                }
            }
        }

        if (g_download_cancelled.load()) {
            cancelled = true;
        } else if (callback && now - last_report >= kParallelProgressInterval) {
            last_report = now;
            // An exception must not unwind past the live curl handles.
            try {
                if (!callback(static_cast<size_t>(completed_bytes + in_flight),
                              static_cast<size_t>(remaining))) {
                    cancelled = true;
                }
            } catch (...) {
                cancelled = true;
            }
        }
        if (cancelled || stalled) {
            failed = true;
            break;
        }

        curl_multi_poll(multi, nullptr, 0, 100, nullptr);
    }

    uint64_t attempt_bytes = completed_bytes;
    for (auto& t : active) {
        attempt_bytes += t->written;
        release(t.get());
    }
    active.clear();
    curl_multi_cleanup(multi);
    result.bytes_downloaded = static_cast<size_t>(attempt_bytes);

    if (!failed) {
        std::error_code ec;
        const uintmax_t size = fs::file_size(partial_fs, ec);
        if (!ec && size == total_size && sync_stdio_file(sync_fp)) {
            sync_fp_closer.close();
            // Record completion first so a resurrected journal cannot truncate
            // the finished file.
            write_range_journal(journal, total_size);
            durable_remove(journal);
            if (callback) {
                try {
                    callback(static_cast<size_t>(remaining), static_cast<size_t>(remaining));
                } catch (...) {
                }
            }
            result.success = true;
            result.http_code = 206;
            return result;
        }
        failed = true;
        if (!ec && size == total_size) {
            setup_failed = true;
            fail_message = "Could not flush downloaded data to disk: " + partial_path;
        } else {
            bad_range = true;
            fail_message = "Parallel download finished with an unexpected file size";
        }
    }
    sync_fp_closer.close();

    // Drop every byte past the contiguous prefix so the partial is an honest
    // resume point for whichever path (parallel or single stream) runs next.
    if (!normalize_partial_file(partial_fs)) {
        result.error_message = "Download interrupted and the partial file could not be reset: " +
                               partial_path;
        result.can_resume = false;
        return result;
    }
    const bool have_prefix = prefix > 0;
    result.http_code = fail_http;

    if (cancelled) {
        result.cancelled = true;
        result.can_resume = true;
        result.error_message = "Download cancelled by user";
        return result;
    }
    if (setup_failed || fall_back_single_stream) {
        result.can_resume = true;
        result.error_message = fail_message.empty() ? "Parallel download failed" : fail_message;
        result.curl_error = result.error_message;
        return result;
    }
    if (stalled) {
        result.can_resume = have_prefix;
        result.error_message = "Download stalled: no bytes received for " +
                               std::to_string(no_progress_timeout) + " seconds";
        result.curl_error = result.error_message;
        return result;
    }
    if (write_failed) {
        result.can_resume = true;
        result.curl_code = static_cast<int>(CURLE_WRITE_ERROR);
        result.curl_error = curl_easy_strerror(CURLE_WRITE_ERROR);
        result.disk_full = disk_nearly_full(get_disk_space_probe_path(partial_fs));
        result.error_message = result.disk_full
            ? "Disk full: not enough space to complete download"
            : "Download failed: could not write to " + partial_path;
        return result;
    }
    if (fail_code != CURLE_OK) {
        bool retryable = false;
        classify_curl_failure(fail_code, retryable, result.permanent);
        result.curl_code = static_cast<int>(fail_code);
        result.curl_error = curl_easy_strerror(fail_code);
        result.can_resume = retryable && have_prefix;
        result.error_message = "Download failed: " + result.curl_error +
                               " (CURL code: " + std::to_string(result.curl_code) + ")";
        return result;
    }
    if (fail_http >= 400) {
        const bool transient = fail_http == 408 || fail_http == 429 || fail_http >= 500;
        result.can_resume = transient && have_prefix;
        result.error_message = "HTTP error " + std::to_string(fail_http) + " for URL: " + url;
        result.curl_error = "HTTP " + std::to_string(fail_http);
        return result;
    }
    result.can_resume = bad_range && have_prefix;
    result.error_message = fail_message.empty() ? "Parallel download failed" : fail_message;
    result.curl_error = result.error_message;
    return result;
}

DownloadResult HttpClient::download_file(const std::string& url,
                                         const std::string& output_path,
                                         ProgressCallback callback,
                                         const std::map<std::string, std::string>& headers,
                                         const DownloadOptions& options,
                                         HttpSecurityPolicy policy) {
    DownloadResult final_result;
    int retry_delay_ms = options.initial_retry_delay_ms;
    const ExpectedHash expected_hash = parse_expected_hash(options);

    if (!options.expected_hash.empty() && !expected_hash.present()) {
        final_result.success = false;
        final_result.error_message = "Invalid or unsupported expected hash: " + options.expected_hash;
        return final_result;
    }

    // Use .partial extension for in-progress downloads
    std::string partial_path = output_path + ".partial";
    fs::path output_path_fs = path_from_utf8(output_path);
    fs::path partial_path_fs = path_from_utf8(partial_path);

    if (!normalize_partial_file(partial_path_fs)) {
        final_result.success = false;
        final_result.error_message = "Interrupted parallel download could not be reset: " + partial_path;
        return final_result;
    }

    // If a verified final file exists next to a stale .partial file, trust the
    // verified final file and remove the stale partial.
    if (expected_hash.present() && fs::exists(output_path_fs) && fs::exists(partial_path_fs)) {
        auto hash_result = verify_file_hash_cached(output_path_fs, expected_hash);
        if (hash_result.ok) {
            std::error_code remove_partial_ec;
            fs::remove(partial_path_fs, remove_partial_ec);
            final_result.success = true;
            final_result.bytes_downloaded = 0;
            LOG(INFO, "Download") << "File already exists and hash verified; removed stale partial: "
                                  << output_path << std::endl;
            return final_result;
        }

        std::error_code remove_output_ec;
        fs::remove(output_path_fs, remove_output_ec);
        if (remove_output_ec) {
            final_result.success = false;
            final_result.error_message = "Existing file failed verification and could not be removed: "
                                       + remove_output_ec.message();
            return final_result;
        }
    }

    // Check if final file already exists and is complete. When the caller
    // provided a content hash, the final path is only trusted if the hash
    // matches; otherwise remove it and force a fresh download.
    if (fs::exists(output_path_fs) && !fs::exists(partial_path_fs)) {
        if (expected_hash.present()) {
            auto hash_result = verify_file_hash_cached(output_path_fs, expected_hash);
            if (hash_result.ok) {
                final_result.success = true;
                final_result.bytes_downloaded = 0;
                LOG(INFO, "Download") << "File already exists and hash verified: "
                                      << output_path << std::endl;
                return final_result;
            }

            LOG(WARNING, "Download") << "Existing file failed verification; removing for fresh download: "
                                     << output_path << std::endl;
            std::error_code remove_ec;
            fs::remove(output_path_fs, remove_ec);
            if (remove_ec) {
                final_result.success = false;
                final_result.error_message = "Existing file failed verification and could not be removed: "
                                           + remove_ec.message();
                return final_result;
            }
        } else {
            // Final file exists with no partial - consider it complete when no
            // stronger source-of-truth hash is available.
            final_result.success = true;
            final_result.bytes_downloaded = 0;
            LOG(INFO, "Download") << "File already exists: " << output_path << std::endl;
            return final_result;
        }
    }

    // Check for existing partial file to resume
    size_t resume_offset = 0;
    if (options.resume_partial && fs::exists(partial_path_fs)) {
        resume_offset = fs::file_size(partial_path_fs);
        if (resume_offset > 0) {
            LOG(INFO, "Download") << " Found partial file ("
                      << std::fixed << std::setprecision(1)
                      << (resume_offset / (1024.0 * 1024.0))
                      << " MB), resuming..." << std::endl;
        }
    }

    // A rate cap is enforced per transfer and by serializing downloads, so
    // splitting one file across connections would multiply it; stay single.
    // Re-evaluated before every attempt so a cap set mid-download applies.
    auto parallel_allowed = []() {
        return download_connections_.load() > 1 && download_rate_limit_bytes_per_second_.load() <= 0;
    };

    bool use_parallel = false;
    size_t parallel_total = 0;
    if (options.allow_parallel && parallel_allowed() &&
        (options.expected_size == 0 || options.expected_size >= options.parallel_min_bytes)) {
        const RangeProbe probe = probe_range_support(url, headers, options, policy, callback,
                                                     resume_offset);
        if (probe.cancelled) {
            final_result.cancelled = true;
            final_result.can_resume = true;
            final_result.error_message = "Download cancelled by user";
            LOG(INFO, "Download") << " Cancelled by user" << std::endl;
            return final_result;
        }
        if (!probe.supported) {
            LOG(INFO, "Download") << "Server did not honour a Range request; using a single stream"
                                  << std::endl;
        } else if (probe.total >= options.parallel_min_bytes && resume_offset < probe.total) {
            use_parallel = true;
            parallel_total = static_cast<size_t>(probe.total);
            LOG(INFO, "Download") << "Downloading with " << download_connections_.load()
                                  << " parallel connections ("
                                  << std::fixed << std::setprecision(1)
                                  << (probe.total / (1024.0 * 1024.0)) << " MB)" << std::endl;
        }
    }

    for (int attempt = 0; attempt <= options.max_retries; ++attempt) {
        if (attempt > 0) {
            LOG(INFO, "Download") << " Retry " << attempt << "/" << options.max_retries
                      << " after " << (retry_delay_ms / 1000.0) << "s..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));

            // Exponential backoff (parentheses avoid Windows min/max macro)
            retry_delay_ms = (std::min)(retry_delay_ms * 2, options.max_retry_delay_ms);

            if (options.resume_partial && fs::exists(partial_path_fs)) {
                size_t new_offset = fs::file_size(partial_path_fs);
                if (new_offset != resume_offset) {
                    resume_offset = new_offset;
                    LOG(INFO, "Download") << "Resuming from "
                              << std::fixed << std::setprecision(1)
                              << (resume_offset / (1024.0 * 1024.0)) << " MB" << std::endl;
                }
            } else if (!fs::exists(partial_path_fs)) {
                resume_offset = 0;
            }
        }

        if (use_parallel && !parallel_allowed()) {
            LOG(INFO, "Download") << "Parallel download disabled by configuration; continuing as a single stream"
                                  << std::endl;
            use_parallel = false;
        }

        ProgressCallback adjusted_callback = nullptr;
        if (callback) {
            adjusted_callback = [callback, resume_offset](size_t current, size_t total) -> bool {
                if (total > 0) {
                    return callback(resume_offset + current, resume_offset + total);
                }
                return callback(resume_offset + current, 0);
            };
        }

        const bool retrying_without_partial = (attempt > 0 && resume_offset == 0);
        const bool initial_range_request =
            options.force_initial_range_request ||
            (retrying_without_partial && options.range_retry_on_zero_byte_retry);

        if (use_parallel) {
            bool fall_back_single_stream = false;
            final_result = parallel_download_attempt(url, partial_path, resume_offset, parallel_total,
                                                     download_connections_.load(), adjusted_callback,
                                                     headers, options, policy, fall_back_single_stream);
            if (fall_back_single_stream) {
                use_parallel = false;
            }
        } else {
            // Released between attempts so retry backoff does not stall other downloads.
            const int64_t rate_limit = download_rate_limit_bytes_per_second_.load();
            std::optional<std::lock_guard<std::mutex>> gate;
            if (rate_limit > 0) {
                gate.emplace(g_download_gate);
            }
            final_result = download_attempt(url, partial_path, resume_offset,
                                            adjusted_callback, headers, options,
                                            initial_range_request, policy);
        }

        // If cancelled by user, return immediately without retrying
        if (final_result.cancelled) {
            LOG(INFO, "Download") << " Cancelled by user" << std::endl;
            return final_result;
        }

        // If disk is full, fail immediately — retrying will just waste bandwidth
        if (final_result.disk_full) {
            LOG(ERROR, "HttpClient") << "[Download] " << final_result.error_message << std::endl;
            return final_result;
        }

        if (final_result.success) {
            if (expected_hash.present()) {
                auto hash_result = verify_file_hash(partial_path_fs, expected_hash);
                if (!hash_result.ok) {
                    final_result.success = false;
                    final_result.can_resume = false;
                    final_result.error_message = "Download content verification failed for " + output_path +
                                                 ": " + hash_result.error;
                    std::error_code remove_ec;
                    fs::remove(partial_path_fs, remove_ec);
                    resume_offset = 0;

                    if (attempt < options.max_retries) {
                        LOG(ERROR, "HttpClient") << "[Download] " << final_result.error_message
                                                  << "; retrying from scratch" << std::endl;
                        continue;
                    }

                    break;
                }
                LOG(INFO, "Download") << "Hash verified for " << output_path << std::endl;
            }

            // Download complete - rename .partial to final path
            std::error_code ec;
            bool rename_synced = false;
            if (!rename_with_sync(partial_path_fs, output_path_fs, &rename_synced)) {
                // Rename failed - try copy and delete
                fs::copy_file(partial_path_fs, output_path_fs, fs::copy_options::overwrite_existing, ec);
                if (!ec) {
                    rename_synced = sync_path(output_path_fs) && sync_parent_directory(output_path_fs);
                    fs::remove(partial_path_fs, ec);
                }
            }
            if (!ec && !rename_synced) {
                LOG(WARNING, "Download") << "Could not confirm the completed file reached the disk: "
                                         << output_path << std::endl;
            }
            if (ec) {
                final_result.success = false;
                final_result.error_message = "Download succeeded but failed to rename file: " + ec.message();
            } else if (expected_hash.present()) {
                write_verified_sidecar(output_path_fs, expected_hash);
            } else {
                // An unverified replacement invalidates any prior attestation:
                // a sidecar must never attest an inode it did not see.
                remove_verified_sidecar(output_path_fs);
            }
            return final_result;
        }

        // A permanent transport failure (unsupported protocol, malformed URL)
        // can never succeed on retry. Stop immediately and leave any partial
        // file in place rather than removing it on each doomed attempt.
        if (final_result.permanent) {
            LOG(ERROR, "HttpClient") << "[Download] " << final_result.error_message << std::endl;
            break;
        }

        // Don't retry permanent HTTP failures (4xx client errors).
        // 408 Request Timeout and 429 Too Many Requests are transient and still retried.
        bool is_permanent_4xx = (final_result.http_code >= 400 && final_result.http_code < 500
                                 && final_result.http_code != 408
                                 && final_result.http_code != 429);
        if (is_permanent_4xx) {
            LOG(ERROR, "HttpClient") << "[Download] " << final_result.error_message << std::endl;
            if (fs::exists(partial_path_fs)) {
                fs::remove(partial_path_fs);
            }
            break;
        }

        if (!final_result.can_resume && attempt < options.max_retries) {
            LOG(ERROR, "HttpClient") << "\n[Download] Error (attempt " << (attempt + 1) << "): "
                      << final_result.error_message << std::endl;

            if (fs::exists(partial_path_fs)) {
                LOG(WARNING, "HttpClient") << "[Download] Removing incomplete file for fresh retry..." << std::endl;
                fs::remove(partial_path_fs);
            }
            resume_offset = 0;
        } else if (final_result.can_resume) {
            LOG(WARNING, "HttpClient") << "\n[Download] Connection interrupted (attempt " << (attempt + 1) << "): "
                      << final_result.curl_error << std::endl;
        } else {
            break;
        }
    }

    std::ostringstream oss;
    oss << "Download failed after " << (options.max_retries + 1) << " attempts.\n";
    oss << "Last error: " << final_result.error_message;

    if (fs::exists(partial_path_fs)) {
        size_t partial_size = fs::file_size(partial_path_fs);
        if (partial_size > 0) {
            oss << "\n\nPartial file preserved: " << partial_path;
            oss << "\nPartial size: " << std::fixed << std::setprecision(1)
                << (partial_size / (1024.0 * 1024.0)) << " MB";
            oss << "\n\nRun the command again to resume from where it left off.";
        }
    }

    final_result.error_message = oss.str();
    return final_result;
}

bool HttpClient::is_reachable(const std::string& url,
                              int timeout_seconds,
                              HttpSecurityPolicy policy) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }

    std::string response_body;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "lemon.cpp/1.0");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    if (!apply_http_security_policy(curl, policy, false)) {
        curl_easy_cleanup(curl);
        return false;
    }

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        return false;
    }

    // Check HTTP status code
    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    curl_easy_cleanup(curl);

    return response_code == 200;
}

} // namespace utils
} // namespace lemon
