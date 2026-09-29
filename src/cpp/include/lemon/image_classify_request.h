#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lemon {
namespace image_classify {

using json = nlohmann::json;

constexpr std::size_t kMaxImageBytes = 16u * 1024u * 1024u;
// Base64 of kMaxImageBytes plus room for the JSON or multipart framing.
constexpr std::uint64_t kMaxImageClassifyRequestBytes = 23ull * 1024ull * 1024ull;
// Longest JSON "image" string worth decoding: base64 of kMaxImageBytes plus a
// data: URL prefix and line breaks.
constexpr std::size_t kMaxImageBase64Chars = 4u * ((kMaxImageBytes + 2u) / 3u) + 64u * 1024u;

// Each admitted request can hold several copies of a maximum-size body while it
// is parsed and forwarded, and httplib runs up to 256 workers.
constexpr int kMaxConcurrentImageClassify = 2;

constexpr int kDefaultTopK = 5;
constexpr long long kMaxTopK = 1000000;

// A validated /v1/images/classify request. On failure status is the HTTP status
// to return and error says why; params and bytes are then meaningless.
struct ParsedRequest {
    json params = json::object();  // "model" and "top_k" when given; never the image
    std::string bytes;
    std::string mime;
    int status = 200;
    std::string error;

    bool ok() const { return status == 200; }
};

ParsedRequest parse_json(const std::string& body);

// Multipart input: form fields as sent, and every file part named "image" or
// "file". Anything but exactly one such part is a 400.
ParsedRequest from_form(const std::optional<std::string>& model,
                        const std::optional<std::string>& top_k,
                        const std::vector<std::string_view>& file_parts);

// Standard alphabet only, '=' padding, ASCII whitespace ignored, length a
// multiple of 4. Needed because JsonUtils::base64_decode stops silently at the
// first invalid character instead of failing.
bool strict_base64_decode(std::string_view in, std::string& out);

// "image/jpeg", "image/png", or "" for anything else.
std::string sniff_image_mime(std::string_view bytes);

// True for POST targets of this route under any of its four prefixes.
bool is_image_classify_path(const std::string& path);

// Checked before the body is read, so the body can only be what Content-Length
// declares. Returns 0 to proceed; 415 for any Content-Encoding (httplib would
// inflate it past the declared length); 411 for any Transfer-Encoding (even
// with a Content-Length, httplib reads chunked framing first) or no
// Content-Length; 413 when it exceeds kMaxImageClassifyRequestBytes.
int precheck_request_headers(bool has_length, std::uint64_t length,
                             bool has_transfer_encoding, bool has_content_encoding);

class InflightLimiter {
public:
    explicit InflightLimiter(int max_in_flight) : max_(max_in_flight) {}
    InflightLimiter(const InflightLimiter&) = delete;
    InflightLimiter& operator=(const InflightLimiter&) = delete;

    bool try_acquire();
    void release();
    int in_flight() const { return count_.load(std::memory_order_acquire); }

private:
    const int max_;
    std::atomic<int> count_{0};
};

}  // namespace image_classify
}  // namespace lemon
