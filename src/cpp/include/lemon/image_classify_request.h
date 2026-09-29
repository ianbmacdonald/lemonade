#pragma once

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

// Checked before the body is read: 0 to proceed, 411 when there is no
// Content-Length (chunked), 413 when it exceeds kMaxImageClassifyRequestBytes.
int precheck_content_length(bool has_length, std::uint64_t length);

}  // namespace image_classify
}  // namespace lemon
