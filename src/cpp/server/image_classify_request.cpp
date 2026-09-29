#include "lemon/image_classify_request.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <iterator>
#include <string>
#include <utility>

namespace lemon {
namespace image_classify {

namespace {

ParsedRequest fail(int status, std::string message) {
    ParsedRequest r;
    r.status = status;
    r.error = std::move(message);
    return r;
}

const char* const kTopKError = "'top_k' must be an integer from 1 to 1000000";

bool starts_with_ci(std::string_view s, std::string_view prefix) {
    if (s.size() < prefix.size()) return false;
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(s[i])) !=
            std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    return true;
}

bool is_ascii_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

bool parse_top_k_text(const std::string& text, long long& out) {
    if (text.empty() || text.size() > 7) return false;
    long long value = 0;
    for (char c : text) {
        if (c < '0' || c > '9') return false;
        value = value * 10 + (c - '0');
    }
    out = value;
    return value >= 1 && value <= kMaxTopK;
}

// Shared tail of both input forms: size and format checks on decoded bytes.
ParsedRequest finish(ParsedRequest r) {
    if (r.bytes.empty()) return fail(400, "'image' is empty");
    if (r.bytes.size() > kMaxImageBytes) {
        return fail(413, "image exceeds the " + std::to_string(kMaxImageBytes / (1024 * 1024)) +
                             " MiB limit");
    }
    r.mime = sniff_image_mime(r.bytes);
    if (r.mime.empty()) return fail(400, "unsupported image format (JPEG or PNG)");
    return r;
}

}  // namespace

bool strict_base64_decode(std::string_view in, std::string& out) {
    static const std::array<int, 256> kTable = [] {
        std::array<int, 256> t{};
        t.fill(-1);
        const char* alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) t[static_cast<unsigned char>(alphabet[i])] = i;
        return t;
    }();

    std::string compact;
    compact.reserve(in.size());
    for (char c : in) {
        if (!is_ascii_space(c)) compact.push_back(c);
    }
    if (compact.empty() || compact.size() % 4 != 0) return false;

    std::size_t padding = 0;
    if (compact.back() == '=') {
        ++padding;
        if (compact[compact.size() - 2] == '=') ++padding;
    }
    const std::size_t data_len = compact.size() - padding;
    for (std::size_t i = 0; i < data_len; ++i) {
        if (kTable[static_cast<unsigned char>(compact[i])] < 0) return false;
    }

    out.clear();
    out.reserve(compact.size() / 4 * 3);
    for (std::size_t i = 0; i < compact.size(); i += 4) {
        const bool last = i + 4 == compact.size();
        const int a = kTable[static_cast<unsigned char>(compact[i])];
        const int b = kTable[static_cast<unsigned char>(compact[i + 1])];
        const int c = (last && padding == 2) ? 0 : kTable[static_cast<unsigned char>(compact[i + 2])];
        const int d = (last && padding >= 1) ? 0 : kTable[static_cast<unsigned char>(compact[i + 3])];
        const unsigned triple = (static_cast<unsigned>(a) << 18) | (static_cast<unsigned>(b) << 12) |
                                (static_cast<unsigned>(c) << 6) | static_cast<unsigned>(d);
        out.push_back(static_cast<char>((triple >> 16) & 0xFF));
        if (!(last && padding == 2)) out.push_back(static_cast<char>((triple >> 8) & 0xFF));
        if (!(last && padding >= 1)) out.push_back(static_cast<char>(triple & 0xFF));
    }
    return true;
}

std::string sniff_image_mime(std::string_view bytes) {
    static constexpr unsigned char kPng[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    auto at = [&](std::size_t i) { return static_cast<unsigned char>(bytes[i]); };
    if (bytes.size() >= 3 && at(0) == 0xFF && at(1) == 0xD8 && at(2) == 0xFF) {
        return "image/jpeg";
    }
    if (bytes.size() >= sizeof(kPng) &&
        std::equal(std::begin(kPng), std::end(kPng), bytes.begin(),
                   [](unsigned char a, char b) { return a == static_cast<unsigned char>(b); })) {
        return "image/png";
    }
    return "";
}

bool is_image_classify_path(const std::string& path) {
    return path == "/api/v0/images/classify" || path == "/api/v1/images/classify" ||
           path == "/v0/images/classify" || path == "/v1/images/classify";
}

int precheck_content_length(bool has_length, std::uint64_t length) {
    if (!has_length) return 411;
    if (length > kMaxImageClassifyRequestBytes) return 413;
    return 0;
}

ParsedRequest parse_json(const std::string& body) {
    json doc = json::parse(body, nullptr, false);
    if (doc.is_discarded()) return fail(400, "Invalid JSON in request body");
    if (!doc.is_object()) return fail(400, "Request body must be a JSON object");

    if (doc.contains("model") && !doc["model"].is_string()) {
        return fail(400, "'model' must be a string");
    }
    if (doc.contains("top_k")) {
        const json& k = doc["top_k"];
        if (!k.is_number_integer() || k.get<long long>() < 1 || k.get<long long>() > kMaxTopK) {
            return fail(400, kTopKError);
        }
    }
    if (!doc.contains("image") || !doc["image"].is_string()) {
        return fail(400, "Missing 'image' (base64 or data: URL string)");
    }

    ParsedRequest r;
    {
        const std::string& image = doc["image"].get_ref<const std::string&>();
        std::string_view view(image);
        while (!view.empty() && is_ascii_space(view.front())) view.remove_prefix(1);

        if (starts_with_ci(view, "http://") || starts_with_ci(view, "https://")) {
            return fail(400, "remote image URLs are not supported; send base64 or a data: URL");
        }
        if (starts_with_ci(view, "data:")) {
            const std::size_t comma = view.find(',');
            if (comma == std::string_view::npos) return fail(400, "malformed data: URL");
            std::string header(view.substr(5, comma - 5));
            std::transform(header.begin(), header.end(), header.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (header != "image/jpeg;base64" && header != "image/jpg;base64" &&
                header != "image/png;base64") {
                return fail(400, "data: URL must be image/jpeg or image/png with base64 encoding");
            }
            view.remove_prefix(comma + 1);
        }
        if (!strict_base64_decode(view, r.bytes)) {
            return fail(400, "'image' is not valid base64");
        }
    }

    doc.erase("image");
    r.params = std::move(doc);
    return finish(std::move(r));
}

ParsedRequest from_form(const std::optional<std::string>& model,
                        const std::optional<std::string>& top_k,
                        const std::vector<std::string_view>& file_parts) {
    if (file_parts.size() != 1) {
        return fail(400, "exactly one image part ('image' or 'file') required");
    }
    ParsedRequest r;
    if (model && !model->empty()) r.params["model"] = *model;
    if (top_k) {
        long long k = 0;
        if (!parse_top_k_text(*top_k, k)) return fail(400, kTopKError);
        r.params["top_k"] = k;
    }
    r.bytes.assign(file_parts.front().data(), file_parts.front().size());
    return finish(std::move(r));
}

}  // namespace image_classify
}  // namespace lemon
