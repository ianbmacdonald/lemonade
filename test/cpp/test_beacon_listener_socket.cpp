#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

bool write_file(const char* path, const std::string& content) {
    int fd = open(path, O_WRONLY);
    if (fd < 0) return false;
    bool ok = write(fd, content.data(), content.size()) == static_cast<ssize_t>(content.size());
    close(fd);
    return ok;
}

// A private user+network namespace lets the test create interfaces whose
// addresses carry no configured broadcast (as OpenWrt/prplOS LAN bridges do)
// without root, and keeps UDP 13305 clear of any production lemond.
bool enter_private_netns() {
    uid_t uid = getuid();
    gid_t gid = getgid();
    if (unshare(CLONE_NEWUSER | CLONE_NEWNET) != 0) {
        std::printf("[SKIP] unshare(CLONE_NEWUSER|CLONE_NEWNET) failed: %s\n", std::strerror(errno));
        return false;
    }
    write_file("/proc/self/setgroups", "deny");
    if (!write_file("/proc/self/uid_map", "0 " + std::to_string(uid) + " 1") ||
        !write_file("/proc/self/gid_map", "0 " + std::to_string(gid) + " 1")) {
        std::printf("[SKIP] could not map uid/gid in the new user namespace\n");
        return false;
    }
    const char* setup =
        "ip link set lo up && "
        "ip link add lbt0 type dummy && ip link set lbt0 up && ip addr add 10.213.0.1/24 dev lbt0 && "
        "ip link add lbt1 type dummy && ip link set lbt1 up && ip addr add 10.214.0.1/24 dev lbt1";
    if (std::system(setup) != 0) {
        std::printf("[SKIP] could not create dummy interfaces with ip(8)\n");
        return false;
    }
    return true;
}

bool send_beacon(const std::string& src_ip, const std::string& dst_ip, const std::string& url,
                 const std::string& instance_id) {
    int tx = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (tx < 0) return false;
    int one = 1;
    setsockopt(tx, SOL_SOCKET, SO_BROADCAST, &one, sizeof(one));
    sockaddr_in from{};
    from.sin_family = AF_INET;
    inet_pton(AF_INET, src_ip.c_str(), &from.sin_addr);
    if (bind(tx, reinterpret_cast<sockaddr*>(&from), sizeof(from)) != 0) {
        close(tx);
        return false;
    }
    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port = htons(lemon::kBeaconPort);
    inet_pton(AF_INET, dst_ip.c_str(), &to.sin_addr);
    std::string payload = json({{"service", "lemonade"},
                                {"hostname", "netns-peer"},
                                {"url", url},
                                {"instance_id", instance_id}})
                              .dump();
    bool ok = sendto(tx, payload.data(), payload.size(), 0, reinterpret_cast<sockaddr*>(&to), sizeof(to)) ==
              static_cast<ssize_t>(payload.size());
    close(tx);
    return ok;
}

bool host_listed(const BeaconListener& listener, const std::string& url) {
    json st = listener.status_json();
    for (const auto& h : st["hosts"]) {
        if (h["url"] == url) return true;
    }
    return false;
}

bool socket_bound(const json& st, const std::string& address) {
    for (const auto& s : st["sockets"]) {
        if (s["address"] == address && s["bound"] == true) return true;
    }
    return false;
}

int run_netns_tests() {
    std::puts("=== RUNNING BEACON LISTENER NETNS BROADCAST TESTS ===");

    check(!lemon::beacon_arrival_accepted(true, 0, {2}),
          "allow-list active, arrival interface unknown: rejected (fail closed)");
    check(lemon::beacon_arrival_accepted(false, 0, {2}), "no allow-list, arrival interface unknown: accepted");
    check(lemon::beacon_arrival_accepted(true, 2, {2}), "allow-list active, arrival on listened interface: accepted");
    check(!lemon::beacon_arrival_accepted(true, 3, {2}), "allow-list active, arrival on other interface: rejected");
    if (!enter_private_netns()) return 0;

    NetworkBeacon nb;
    std::string lbt0_bcast;
    for (const auto& iface : nb.getLocalRFC1918Interfaces()) {
        if (iface.ipAddress == "10.213.0.1") lbt0_bcast = iface.broadcastAddress;
    }
    check(lbt0_bcast == "10.213.0.255",
          ("subnet broadcast of 10.213.0.1/24 without brd is 10.213.0.255 (got '" + lbt0_bcast + "')").c_str());

    {
        BeaconListener listener;
        listener.start("4444444444444444", 18999);
        check(wait_for([&] { return listener.status_json().value("listening", false); },
                       std::chrono::milliseconds(2000)),
              "listener is listening in the private netns");
        json st = listener.status_json();
        check(socket_bound(st, "10.213.0.255:13305"), "lbt0 subnet broadcast 10.213.0.255 is bound");
        check(socket_bound(st, "10.214.0.255:13305"), "lbt1 subnet broadcast 10.214.0.255 is bound");
        check(!socket_bound(st, "10.213.0.1:13305"), "lbt0 unicast address is not bound");

        const std::string url_a = "http://10.213.0.1:65001/api/v1/";
        check(send_beacon("10.213.0.1", "10.213.0.255", url_a, "abcdef0123456789"),
              "sent beacon to subnet broadcast 10.213.0.255");
        check(wait_for([&] { return host_listed(listener, url_a); }, std::chrono::milliseconds(3000)),
              "beacon sent to the subnet broadcast is heard");

        const std::string url_b = "http://10.214.0.1:65002/api/v1/";
        check(send_beacon("10.214.0.1", "255.255.255.255", url_b, "bcdef0123456789a"),
              "sent beacon to limited broadcast 255.255.255.255");
        check(wait_for([&] { return host_listed(listener, url_b); }, std::chrono::milliseconds(3000)),
              "beacon sent to the limited broadcast is heard");
        listener.stop();
    }

    {
        BeaconListener listener;
        listener.set_interface_allowlist({"lbt0"});
        listener.start("5555555555555555", 18999);
        check(wait_for([&] { return listener.status_json().value("listening", false); },
                       std::chrono::milliseconds(2000)),
              "allow-listed listener is listening");
        json st = listener.status_json();
        check(st["interface_allowlist"] == json::array({"lbt0"}), "status echoes the interface allow-list");
        check(st["interfaces"].size() == 1 && st["interfaces"][0]["name"] == "lbt0" &&
                  st["interfaces"][0]["broadcast"] == "10.213.0.255",
              "only lbt0 is listed as a listened interface");
        check(socket_bound(st, "10.213.0.255:13305"), "allow-listed lbt0 broadcast is bound");
        check(!socket_bound(st, "10.214.0.255:13305"), "lbt1 broadcast is not bound");

        const std::string url_c = "http://10.214.0.1:65003/api/v1/";
        send_beacon("10.214.0.1", "255.255.255.255", url_c, "cdef0123456789ab");
        send_beacon("10.214.0.1", "10.214.0.255", url_c, "cdef0123456789ab");
        const std::string url_d = "http://10.213.0.1:65004/api/v1/";
        send_beacon("10.213.0.1", "255.255.255.255", url_d, "def0123456789abc");
        check(wait_for([&] { return host_listed(listener, url_d); }, std::chrono::milliseconds(3000)),
              "limited broadcast arriving on allow-listed lbt0 is heard");
        check(!host_listed(listener, url_c), "beacons arriving on lbt1 are not listed");
        check(listener.status_json()["stats"]["wrong_interface"].get<int>() >= 1,
              "limited broadcast arriving on lbt1 is counted as wrong_interface");

        listener.set_interface_allowlist({"lbt1"});
        check(wait_for([&] { return socket_bound(listener.status_json(), "10.214.0.255:13305"); },
                       std::chrono::milliseconds(2000)),
              "changing the allow-list rebinds without a restart");
        check(!socket_bound(listener.status_json(), "10.213.0.255:13305"), "old interface is released");
        check(listener.status_json().value("unmatched_interfaces", json()) == json::array(),
              "a fully matched allow-list reports no unmatched interfaces");

        listener.set_interface_allowlist({"lbt0", "nope0"});
        check(wait_for([&] { return socket_bound(listener.status_json(), "10.213.0.255:13305"); },
                       std::chrono::milliseconds(2000)),
              "partially matched allow-list still listens on lbt0");
        json partial = listener.status_json();
        check(partial.value("unmatched_interfaces", json()) == json::array({"nope0"}),
              "partially matched allow-list reports nope0 as unmatched");
        check(partial["listening"] == true && partial["error"].is_null(),
              "partially matched allow-list is listening without an error");

        listener.set_interface_allowlist({"nope0", "nope1"});
        check(wait_for([&] { return !listener.status_json().value("listening", true); },
                       std::chrono::milliseconds(2000)),
              "allow-list matching no interface is not listening");
        json none = listener.status_json();
        check(none.value("unmatched_interfaces", json()) == json::array({"nope0", "nope1"}),
              "allow-list matching no interface reports every entry as unmatched");
        check(none["error"].is_string() && none["error"].get<std::string>().find("nope0") != std::string::npos,
              "allow-list matching no interface explains itself in error");
        listener.stop();
    }

    {
        BeaconListener listener;
        listener.start("6666666666666666", 18999);
        check(wait_for([&] { return listener.status_json().value("listening", false); },
                       std::chrono::milliseconds(2000)),
              "relay listener is listening");
        const std::string relayed_url = "http://10.99.0.20:65005/api/v1/";
        check(send_beacon("10.213.0.1", "10.213.0.255", relayed_url, "0123456789abcdef"),
              "relay sent a beacon advertising another host's URL");
        check(wait_for([&] { return listener.status_json()["stats"]["url_mismatch"].get<int>() >= 1; },
                       std::chrono::milliseconds(3000)),
              "without a trusted relay the relayed beacon is a url_mismatch");
        check(!host_listed(listener, relayed_url), "without a trusted relay the relayed host is not listed");

        listener.set_trusted_relays(lemon::parse_beacon_trusted_relays(
            json::array({{{"source", "10.213.0.1"}, {"allow_hosts", json::array({"10.99.0.0/24"})}}})));
        check(listener.status_json()["trusted_relays"] ==
                  json::array({{{"source", "10.213.0.1"}, {"allow_hosts", json::array({"10.99.0.0/24"})}}}),
              "status echoes the trusted relays");
        send_beacon("10.214.0.1", "10.214.0.255", relayed_url, "0123456789abcdef");
        send_beacon("10.213.0.1", "10.213.0.255", "http://10.98.0.20:65006/api/v1/", "123456789abcdef0");
        check(send_beacon("10.213.0.1", "10.213.0.255", relayed_url, "0123456789abcdef"),
              "trusted relay re-sent the beacon");
        check(wait_for([&] { return host_listed(listener, relayed_url); }, std::chrono::milliseconds(3000)),
              "relayed beacon from a trusted relay is listed");
        json st = listener.status_json();
        bool via_ok = false;
        for (const auto& h : st["hosts"]) {
            if (h["url"] == relayed_url) via_ok = h["via"] == "10.213.0.1" && h["source_ip"] == "10.213.0.1";
        }
        check(via_ok, "relayed host shows via = the relay's address");
        check(st["stats"].value("relayed_accepted", -1) == 1 && st["stats"].value("accepted", -1) == 0,
              "relayed host counted in relayed_accepted, not accepted");
        check(st["stats"].value("relay_host_refused", -1) >= 1,
              "relayed URL outside allow_hosts counted as relay_host_refused");
        check(st["stats"]["url_mismatch"].get<int>() >= 2, "an untrusted sender advertising the host stays url_mismatch");
        check(!host_listed(listener, "http://10.98.0.20:65006/api/v1/"), "host outside allow_hosts is not listed");

        listener.set_trusted_relays({});
        check(!host_listed(listener, relayed_url), "clearing the trusted relays drops the relayed host");
        listener.stop();
    }

    return test_helpers::report_results("beacon listener netns broadcast");
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--netns") return run_netns_tests();

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
