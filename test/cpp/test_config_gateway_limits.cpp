#include <cstdio>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>
#include <lemon/classify_text_limit.h>
#include <lemon/runtime_config.h>

#include "test_config_helpers.h"

using json = nlohmann::json;
using lemon::RuntimeConfig;
using test_helpers::check;
using test_helpers::report_results;

static bool construct_throws(const json& config) {
    try {
        RuntimeConfig{config};
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

static std::string construct_error(const json& config) {
    try {
        RuntimeConfig{config};
        return "";
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
}

static bool set_throws(const json& changes) {
    try {
        RuntimeConfig(json::object()).set(changes);
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

int main() {
    std::puts("=== RUNNING GATEWAY LIMITS CONFIG TESTS ===");

    const RuntimeConfig empty(json::object());
    check(empty.max_request_body_mb() == 100, "body limit defaults to 100 MB");
    check(empty.request_receive_timeout() == 0, "receive timeout defaults to off");
    check(empty.max_connections_per_client() == 0, "per-client cap defaults to off");

    const RuntimeConfig gateway(json{{"max_request_body_mb", 32},
                                     {"request_receive_timeout", 30},
                                     {"max_connections_per_client", 8}});
    check(gateway.max_request_body_mb() == 32, "body limit read from config");
    check(gateway.request_receive_timeout() == 30, "receive timeout read from config");
    check(gateway.max_connections_per_client() == 8, "per-client cap read from config");

    const RuntimeConfig bad(json{{"max_request_body_mb", 0}, {"request_receive_timeout", -1}});
    check(bad.max_request_body_mb() == 100, "out-of-range body limit falls back to the default");
    check(bad.request_receive_timeout() == 0, "negative timeout falls back to off");

    check(construct_throws({{"max_connections_per_client", "8"}}),
          "a string cap in config.json is refused, not read as off");
    check(construct_throws({{"request_receive_timeout", "30"}}),
          "a string timeout in config.json is refused, not read as off");
    check(construct_throws({{"max_request_body_mb", 32.5}}),
          "a fractional body limit in config.json is refused");
    check(construct_error({{"max_connections_per_client", "8"}}) ==
              "'max_connections_per_client' must be an integer (got string)",
          "the refusal names the type it got, so a quoted number is obvious");

    const RuntimeConfig too_big(json{{"max_request_body_mb", 100000},
                                     {"request_receive_timeout", 7200},
                                     {"max_connections_per_client", 5000}});
    check(too_big.max_request_body_mb() == RuntimeConfig::kMaxMaxRequestBodyMb,
          "a body limit above the maximum is clamped, not reset");
    check(too_big.request_receive_timeout() == RuntimeConfig::kMaxRequestReceiveTimeout,
          "a timeout above the maximum is clamped, not switched off");
    check(too_big.max_connections_per_client() == RuntimeConfig::kMaxConnectionsPerClient,
          "a cap above the maximum is clamped, not switched off");

    check(!set_throws({{"max_request_body_mb", 32}}), "set accepts a valid body limit");
    check(set_throws({{"max_request_body_mb", 0}}), "set refuses a zero body limit");
    check(set_throws({{"max_request_body_mb", RuntimeConfig::kMaxMaxRequestBodyMb + 1}}),
          "set refuses a body limit above the maximum");
    check(set_throws({{"max_request_body_mb", "32"}}), "set refuses a string body limit");
    check(!set_throws({{"request_receive_timeout", 0}}), "set accepts turning the timeout off");
    check(set_throws({{"request_receive_timeout", -5}}), "set refuses a negative timeout");
    check(!set_throws({{"max_connections_per_client", 8}}), "set accepts a valid cap");
    check(set_throws({{"max_connections_per_client", 1.5}}), "set refuses a fractional cap");

    RuntimeConfig live(json::object());
    live.set({{"request_receive_timeout", 45}});
    check(live.request_receive_timeout() == 45, "set value is read back");

    using lemon::classify_text::kMaxTextBytes;
    using lemon::classify_text::too_large_error;
    check(!too_large_error(std::string(kMaxTextBytes, 'a')), "text at the limit is accepted");
    auto err = too_large_error(std::string(kMaxTextBytes + 1, 'a'));
    check(err.has_value() && (*err)["error"]["status_code"] == 413,
          "text over the limit gets a 413 error");

    return report_results("gateway limits config");
}
