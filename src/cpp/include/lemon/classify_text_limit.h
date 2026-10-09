#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace lemon {
namespace classify_text {

// Upper bound on the text a classify backend forwards to its engine server.
// The engine servers refuse far smaller bodies themselves (64 KiB); checking
// here keeps lemond from serializing and sending a large request at all.
constexpr size_t kMaxTextBytes = 1024 * 1024;

// The error response for text over the limit, or nullopt when it fits.
inline std::optional<nlohmann::json> too_large_error(const std::string& text) {
    if (text.size() <= kMaxTextBytes) {
        return std::nullopt;
    }
    return nlohmann::json{
        {"error", {
            {"message", "Classify text exceeds " + std::to_string(kMaxTextBytes) + " bytes"},
            {"type", "invalid_request_error"},
            {"status_code", 413},
        }}
    };
}

} // namespace classify_text
} // namespace lemon
