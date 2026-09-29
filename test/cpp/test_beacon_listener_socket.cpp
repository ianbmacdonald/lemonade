#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>

#include <lemon/utils/beacon_listener.h>
#include <lemon/utils/network_beacon.h>
#include <nlohmann/json.hpp>

#include "test_config_helpers.h"

using json = nlohmann::json;
using lemon::BeaconListener;
using test_helpers::check;

namespace {

int bind_wildcard_no_reuse() {
    int fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (fd < 0) return -1;
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = htons(lemon::kBeaconPort);
    sa.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(fd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) {
        int err = errno;
        close(fd);
        errno = err;
        return -1;
    }
    return fd;
}

bool wait_for(const std::function<bool()>& pred, std::chrono::milliseconds timeout) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return pred();
}

} // namespace

int main() {
    std::puts("=== RUNNING BEACON LISTENER SOCKET TESTS ===");

    int probe = bind_wildcard_no_reuse();
    if (probe < 0) {
        std::printf("[SKIP] UDP %d is already held by another process (errno=%d)\n", lemon::kBeaconPort, errno);
        return 0;
    }
    close(probe);

    {
        BeaconListener listener;
        listener.start("1111111111111111", 18999);
        bool listening = wait_for([&] { return listener.status_json().value("listening", false); },
                                  std::chrono::milliseconds(2000));
        check(listening, "listener reports listening with at least one bound socket");
        json st = listener.status_json();
        check(st["enabled"] == true && st["supported"] == true, "status is enabled and supported");
        check(st["sockets"].is_array() && !st["sockets"].empty(), "sockets[] is non-empty");
        bool has_limited = false;
        for (const auto& s : st["sockets"]) {
            if (s["address"] == "255.255.255.255:13305") has_limited = true;
            if (s["address"] == "0.0.0.0:13305") check(false, "listener never binds INADDR_ANY");
        }
        check(has_limited, "limited broadcast address is bound");

        int fd = bind_wildcard_no_reuse();
        check(fd < 0 && errno == EADDRINUSE, "non-reuse wildcard bind fails while listener runs");
        if (fd >= 0) close(fd);

        NetworkBeacon nb;
        std::string iface_ip;
        std::string iface_bcast;
        for (const auto& iface : nb.getLocalRFC1918Interfaces()) {
            if (iface.ipAddress.rfind("127.", 0) == 0) continue;
            iface_ip = iface.ipAddress;
            iface_bcast = iface.broadcastAddress;
            break;
        }
        if (iface_ip.empty()) {
            std::puts("[SKIP] no RFC1918 interface; broadcast receive not tested");
        } else {
            int tx = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
            int one = 1;
            setsockopt(tx, SOL_SOCKET, SO_BROADCAST, &one, sizeof(one));
            sockaddr_in to{};
            to.sin_family = AF_INET;
            to.sin_port = htons(lemon::kBeaconPort);
            inet_pton(AF_INET, iface_bcast.c_str(), &to.sin_addr);
            std::string good_url = "http://" + iface_ip + ":65001/api/v1/";
            std::string good = json({{"service", "lemonade"},
                                     {"hostname", "peer-test"},
                                     {"url", good_url},
                                     {"instance_id", "abcdef0123456789"}})
                                   .dump();
            std::string mismatch = json({{"service", "lemonade"},
                                         {"hostname", "peer-mismatch"},
                                         {"url", "http://10.9.9.9:1/api/v1/"}})
                                       .dump();
            std::string self = json({{"service", "lemonade"},
                                     {"hostname", "self"},
                                     {"url", "http://" + iface_ip + ":18999/api/v1/"},
                                     {"instance_id", "1111111111111111"}})
                                   .dump();
            for (const auto* p : {&good, &mismatch, &self}) {
                sendto(tx, p->data(), p->size(), 0, reinterpret_cast<sockaddr*>(&to), sizeof(to));
            }
            close(tx);

            bool heard = wait_for(
                [&] {
                    json st_now = listener.status_json();
                    for (const auto& h : st_now["hosts"]) {
                        if (h["url"] == good_url) return true;
                    }
                    return false;
                },
                std::chrono::milliseconds(3000));
            check(heard, ("directed broadcast to " + iface_bcast + " is listed").c_str());
            json after = listener.status_json();
            check(after["stats"]["url_mismatch"].get<int>() >= 1, "url mismatch beacon counted, not listed");
            check(after["stats"]["self"].get<int>() >= 1, "own instance_id beacon counted as self");
            bool self_listed = false;
            for (const auto& h : after["hosts"]) {
                if (h["instance_id"] == "1111111111111111" || h["url"] == "http://10.9.9.9:1/api/v1/") {
                    self_listed = true;
                }
            }
            check(!self_listed, "self and mismatched beacons are not listed");
        }

        auto t0 = std::chrono::steady_clock::now();
        listener.stop();
        auto dt = std::chrono::steady_clock::now() - t0;
        check(dt < std::chrono::seconds(1), "stop() joins in under 1s");
        check(listener.status_json() == json({{"enabled", false}}), "status is {enabled:false} after stop");

        fd = bind_wildcard_no_reuse();
        check(fd >= 0, "non-reuse wildcard bind succeeds after stop");
        if (fd >= 0) close(fd);
    }

    {
        int blocker = bind_wildcard_no_reuse();
        check(blocker >= 0, "pre-occupy UDP 13305 without reuse");
        BeaconListener listener;
        listener.start("2222222222222222", 18999);
        std::this_thread::sleep_for(std::chrono::milliseconds(1200));
        json st = listener.status_json();
        check(st["listening"] == false, "listener reports not listening when every bind fails");
        bool has_error = false;
        for (const auto& s : st["sockets"]) {
            if (!s["error"].is_null()) has_error = true;
        }
        check(has_error, "bind failure is reported per socket");
        auto t0 = std::chrono::steady_clock::now();
        listener.stop();
        auto dt = std::chrono::steady_clock::now() - t0;
        check(dt < std::chrono::seconds(1), "stop() during bind-retry joins in under 1s");
        if (blocker >= 0) close(blocker);
    }

    {
        BeaconListener listener;
        listener.start("3333333333333333", 18999);
        listener.start("3333333333333333", 18999);
        check(listener.is_running(), "start() is idempotent");
        listener.stop();
        listener.stop();
        check(!listener.is_running(), "stop() is idempotent");
        listener.start("3333333333333333", 18999);
        check(listener.is_running(), "listener restarts after stop()");
    }

    return test_helpers::report_results("beacon listener socket");
}
