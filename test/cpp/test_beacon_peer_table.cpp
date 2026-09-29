#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <lemon/utils/beacon_listener.h>
#include <lemon/utils/network_beacon.h>
#include <nlohmann/json.hpp>

#include "test_config_helpers.h"

using json = nlohmann::json;
using lemon::BeaconPeerTable;
using lemon::IngestResult;
using test_helpers::check;
using Clock = std::chrono::steady_clock;

namespace {

uint32_t ip(int a, int b, int c, int d) {
    return (static_cast<uint32_t>(a) << 24) | (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(c) << 8) | static_cast<uint32_t>(d);
}

std::string dotted(uint32_t v) {
    return lemon::utils::ipv4_to_string(v);
}

std::string beacon(const std::string& url, const std::string& hostname = "peer",
                   const std::string& instance_id = "") {
    json j = {{"service", "lemonade"}, {"hostname", hostname}, {"url", url}};
    if (!instance_id.empty()) j["instance_id"] = instance_id;
    return j.dump();
}

std::string url_for(uint32_t host, int port = 13305) {
    return "http://" + dotted(host) + ":" + std::to_string(port) + "/api/v1/";
}

IngestResult feed(BeaconPeerTable& t, const std::string& payload, uint32_t src, Clock::time_point now,
                  bool truncated = false) {
    return t.ingest(payload.data(), payload.size(), truncated, src, now);
}

void test_basic(Clock::time_point t0) {
    BeaconPeerTable t;
    t.set_self("ffffffffffffffff", 13305);
    uint32_t src = ip(192, 168, 1, 40);
    std::string p = beacon(url_for(src), "ai4", "0a1b2c3d4e5f6071");
    check(feed(t, p, src, t0) == IngestResult::Accepted, "basic: first beacon is Accepted");
    check(feed(t, p, src, t0 + std::chrono::seconds(2)) == IngestResult::Refreshed,
          "basic: repeat 2s later is Refreshed");
    check(t.size() == 1, "basic: table size stays 1");
    auto hosts = t.hosts();
    check(hosts.size() == 1 && hosts[0].last_seen == t0 + std::chrono::seconds(2),
          "basic: last_seen advanced");
    json j = t.to_json(t0 + std::chrono::seconds(2));
    check(j["hosts"].size() == 1 && j["hosts"][0]["hostname"] == "ai4" &&
              j["hosts"][0]["source_ip"] == "192.168.1.40" &&
              j["hosts"][0]["instance_id"] == "0a1b2c3d4e5f6071",
          "basic: to_json lists the host");
    check(j["distinct_instances"] == 1, "basic: distinct_instances is 1");
    check(j["stats"]["accepted"] == 1 && j["stats"]["refreshed"] == 1, "basic: stats counted");
}

void test_source_class(Clock::time_point t0) {
    std::vector<uint32_t> bad = {ip(8, 8, 8, 8),   ip(169, 254, 1, 2), ip(100, 64, 0, 1),
                                 ip(224, 0, 0, 1), ip(0, 0, 0, 0),     ip(255, 255, 255, 255),
                                 ip(127, 0, 0, 1), ip(172, 32, 0, 1)};
    for (uint32_t src : bad) {
        BeaconPeerTable t;
        bool ok = feed(t, beacon(url_for(src)), src, t0) == IngestResult::BadSource;
        check(ok, ("source class: " + dotted(src) + " is BadSource").c_str());
    }
    std::vector<uint32_t> good = {ip(10, 0, 0, 1), ip(172, 16, 0, 1), ip(172, 31, 255, 254),
                                  ip(192, 168, 0, 1)};
    for (uint32_t src : good) {
        BeaconPeerTable t;
        bool ok = feed(t, beacon(url_for(src)), src, t0) == IngestResult::Accepted;
        check(ok, ("source class: " + dotted(src) + " is accepted").c_str());
    }
}

void test_url_binding(Clock::time_point t0) {
    uint32_t src = ip(192, 168, 1, 40);
    {
        BeaconPeerTable t;
        check(feed(t, beacon(url_for(ip(192, 168, 1, 41))), src, t0) == IngestResult::UrlMismatch,
              "url: host differing from source is UrlMismatch");
        check(t.stat(IngestResult::UrlMismatch) == 1, "url: url_mismatch counted");
    }
    std::vector<std::string> bad = {
        "https://192.168.1.40:13305/api/v1/",
        "http://ai4.local:13305/api/v1/",
        "http://192.168.1.40/api/v1/",
        "http://192.168.1.40:0/api/v1/",
        "http://192.168.1.40:70000/api/v1/",
        "http://192.168.001.40:13305/api/v1/",
        "http://192.168.1.40:013305/api/v1/",
        "http://192.168.1.40:13305/v1/",
        "http://192.168.1.40:13305/api/v1/x",
        "http://user@192.168.1.40:13305/api/v1/",
        "http://192.168.1.40:13305/api/v1",
        "http://192.168.1:13305/api/v1/",
        "http://192.168.1.256:13305/api/v1/",
    };
    for (const auto& url : bad) {
        BeaconPeerTable t;
        bool ok = feed(t, beacon(url), src, t0) == IngestResult::BadPayload;
        check(ok, ("url: '" + url + "' is BadPayload").c_str());
    }
}

void test_payload_structure(Clock::time_point t0) {
    uint32_t src = ip(10, 1, 2, 3);
    std::string good_url = url_for(src);
    std::vector<std::pair<std::string, std::string>> bad = {
        {"not json", "non-JSON"},
        {"[1,2,3]", "JSON array"},
        {json({{"service", "other"}, {"hostname", "h"}, {"url", good_url}}).dump(), "service != lemonade"},
        {json({{"service", "lemonade"}, {"hostname", "h"}, {"url", 5}}).dump(), "non-string url"},
        {json({{"service", "lemonade"}, {"hostname", 7}, {"url", good_url}}).dump(), "non-string hostname"},
        {json({{"service", "lemonade"}, {"url", good_url}}).dump(), "missing hostname"},
        {beacon(good_url, "h", "XYZ"), "non-hex instance_id"},
        {beacon(good_url, "h", std::string(33, 'a')), "33-char instance_id"},
        {json({{"service", "lemonade"}, {"hostname", "h"}, {"url", good_url}, {"instance_id", 12}}).dump(),
         "non-string instance_id"},
        {"{\"service\":\"lemonade\",\"hostname\":\"h\",\"url\":", "truncated JSON"},
    };
    for (const auto& [payload, desc] : bad) {
        BeaconPeerTable t;
        bool ok = false;
        try {
            ok = feed(t, payload, src, t0) == IngestResult::BadPayload;
        } catch (...) {
            ok = false;
        }
        check(ok, ("payload: " + desc + " is BadPayload and does not throw").c_str());
    }
    {
        BeaconPeerTable t;
        check(feed(t, beacon(good_url), src, t0, true) == IngestResult::BadPayload,
              "payload: truncated flag is BadPayload");
    }
    {
        BeaconPeerTable t;
        check(t.ingest("", 0, false, src, t0) == IngestResult::BadPayload, "payload: len 0 is BadPayload");
    }
    {
        BeaconPeerTable t;
        std::string big = beacon(good_url, std::string(1100, 'a'));
        check(feed(t, big, src, t0) == IngestResult::BadPayload, "payload: >= 1024 bytes is BadPayload");
    }
    {
        BeaconPeerTable t;
        json j = {{"service", "lemonade"}, {"hostname", "h"}, {"url", good_url}, {"extra", {1, 2}}};
        check(feed(t, j.dump(), src, t0) == IngestResult::Accepted, "payload: unknown fields are ignored");
    }
}

void test_hostname_sanitising(Clock::time_point t0) {
    uint32_t src = ip(192, 168, 7, 7);
    std::vector<std::string> raws = {std::string(300, 'x'), "we\"ird", "new\nline", "<script>",
                                     std::string("bad\xC3\x28utf8")};
    for (const auto& raw : raws) {
        BeaconPeerTable t;
        std::string payload = "{\"service\":\"lemonade\",\"hostname\":\"";
        for (char c : raw) {
            if (c == '"') payload += "\\\"";
            else if (c == '\n') payload += "\\n";
            else payload += c;
        }
        payload += "\",\"url\":\"" + url_for(src) + "\"}";
        IngestResult r = feed(t, payload, src, t0);
        bool accepted = r == IngestResult::Accepted;
        auto hosts = t.hosts();
        bool clean = accepted && hosts.size() == 1 && hosts[0].hostname.size() <= 253 &&
                     hosts[0].hostname.find_first_not_of(
                         "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-") ==
                         std::string::npos;
        check(clean, "hostname: accepted and sanitised to [A-Za-z0-9._-]{1,253}");
        json j = t.to_json(t0);
        json round = json::parse(j.dump(), nullptr, false);
        check(!round.is_discarded() && round["hosts"].size() == 1, "hostname: to_json round-trips");
    }
    {
        BeaconPeerTable t;
        std::string payload = json({{"service", "lemonade"}, {"hostname", "b\xEF\xBF\xBDx"}, {"url", url_for(src)}})
                                  .dump();
        check(feed(t, payload, src, t0) == IngestResult::Accepted && t.hosts()[0].hostname == "b___x",
              "hostname: U+FFFD from a sanitising broadcaster becomes underscores");
    }
    {
        BeaconPeerTable t;
        check(feed(t, beacon(url_for(src), ""), src, t0) == IngestResult::Accepted &&
                  t.hosts()[0].hostname == "_",
              "hostname: empty becomes '_'");
    }
}

void test_broadcaster_payload() {
    NetworkBeacon nb;
    std::string bad_host = std::string("h\xC3\x28") + "\"q";
    bool threw = false;
    std::string out;
    try {
        out = nb.buildStandardPayloadPattern(bad_host, "http://192.168.1.40:13305/api/v1/");
    } catch (...) {
        threw = true;
    }
    check(!threw, "broadcaster: invalid UTF-8 hostname does not throw");
    json j = json::parse(out, nullptr, false);
    check(!j.is_discarded() && j["service"] == "lemonade" && j["hostname"].is_string() &&
              j["url"] == "http://192.168.1.40:13305/api/v1/",
          "broadcaster: output parses with service/hostname/url");
    check(!j.is_discarded() && !j.contains("instance_id"), "broadcaster: no instance_id until set");

    nb.setInstanceId("9f3c2a61d04b7e18");
    json j2 = json::parse(nb.buildStandardPayloadPattern("ai4", "http://192.168.1.40:13305/api/v1/"),
                          nullptr, false);
    check(!j2.is_discarded() && j2.value("instance_id", "") == "9f3c2a61d04b7e18",
          "broadcaster: instance_id included after setInstanceId");
}

void test_self_filter(Clock::time_point t0) {
    uint32_t local = ip(192, 168, 1, 10);
    BeaconPeerTable t;
    t.set_self("9f3c2a61d04b7e18", 13305);
    t.set_local_addresses({local, ip(127, 0, 0, 1)});
    check(feed(t, beacon(url_for(local), "me", "9f3c2a61d04b7e18"), local, t0) == IngestResult::Self,
          "self: matching instance_id is Self");
    check(feed(t, beacon(url_for(local), "me"), local, t0) == IngestResult::Self,
          "self: legacy beacon from local address with our port is Self");
    check(feed(t, beacon(url_for(local, 13306), "other"), local, t0) == IngestResult::Accepted,
          "self: local address with a different port is Accepted");
    check(feed(t, beacon(url_for(local), "me2", "0000000000000001"), local, t0 + std::chrono::seconds(1)) ==
              IngestResult::Accepted,
          "self: different instance_id from a local address is Accepted");

    BeaconPeerTable t2;
    t2.set_self("9f3c2a61d04b7e18", 13305);
    t2.set_local_addresses({local});
    t2.set_self_port(9000);
    check(feed(t2, beacon(url_for(local), "me"), local, t0) == IngestResult::Accepted,
          "self: after set_self_port(9000), legacy port 13305 from local is Accepted");
}

void test_expiry(Clock::time_point t0) {
    BeaconPeerTable t;
    uint32_t src = ip(10, 0, 0, 5);
    feed(t, beacon(url_for(src)), src, t0);
    t.expire(t0 + std::chrono::seconds(14));
    check(t.size() == 1, "expiry: row kept at 14s");
    check(t.to_json(t0 + std::chrono::seconds(16))["hosts"].empty(), "expiry: to_json hides expired rows");
    t.expire(t0 + std::chrono::seconds(16));
    check(t.size() == 0, "expiry: row removed at 16s");

    // Once expired, the source is no longer pinned and must pay the global
    // bucket again; exhaust the global bucket and confirm it is refused.
    BeaconPeerTable t2;
    feed(t2, beacon(url_for(src)), src, t0);
    auto t1 = t0 + std::chrono::seconds(16);
    for (int i = 0; i < 1100; ++i) {
        uint32_t s = ip(10, 1, (i >> 8) & 0xFF, i & 0xFF);
        feed(t2, beacon(url_for(s)), s, t1);
    }
    check(feed(t2, beacon(url_for(src)), src, t1) == IngestResult::RateLimited,
          "expiry: expired source is unpinned and subject to the global bucket");
}

void test_cap_and_eviction(Clock::time_point t0) {
    BeaconPeerTable t;
    for (int i = 0; i < 32; ++i) {
        uint32_t s = ip(10, 0, 1, i + 1);
        feed(t, beacon(url_for(s)), s, t0 + std::chrono::milliseconds(i));
    }
    check(t.size() == 32, "cap: 32 rows accepted");
    uint32_t extra = ip(10, 0, 2, 1);
    check(feed(t, beacon(url_for(extra)), extra, t0 + std::chrono::seconds(1)) == IngestResult::TableFull,
          "cap: 33rd key while all rows are fresh is TableFull");
    check(t.size() == 32, "cap: size stays 32");

    BeaconPeerTable t2;
    uint32_t stale = ip(10, 0, 3, 1);
    feed(t2, beacon(url_for(stale)), stale, t0);
    for (int i = 0; i < 31; ++i) {
        uint32_t s = ip(10, 0, 4, i + 1);
        feed(t2, beacon(url_for(s)), s, t0 + std::chrono::seconds(8));
    }
    uint32_t newcomer = ip(10, 0, 5, 1);
    check(feed(t2, beacon(url_for(newcomer)), newcomer, t0 + std::chrono::seconds(9)) == IngestResult::Accepted,
          "eviction: new key accepted when oldest row is 9s stale");
    check(t2.evicted() == 1, "eviction: evicted counter is 1");
    check(t2.size() == 32, "eviction: size stays 32");
}

void test_rate_limit_global(Clock::time_point t0) {
    BeaconPeerTable t;
    int admitted = 0;
    for (int i = 0; i < 2000; ++i) {
        uint32_t s = ip(10, 2, (i >> 8) & 0xFF, i & 0xFF);
        if (feed(t, beacon(url_for(s)), s, t0) != IngestResult::RateLimited) ++admitted;
    }
    check(admitted <= 1000, "global rate: at most 1000 admitted at one instant");
    check(admitted >= 990, "global rate: burst of about 1000 admitted");
    check(t.tracked_source_count() <= lemon::kMaxSources + lemon::kMaxHosts,
          "global rate: tracked sources stay bounded");

    int refill = 0;
    auto t1 = t0 + std::chrono::seconds(1);
    for (int i = 0; i < 1000; ++i) {
        uint32_t s = ip(10, 3, (i >> 8) & 0xFF, i & 0xFF);
        if (feed(t, beacon(url_for(s)), s, t1) != IngestResult::RateLimited) ++refill;
    }
    check(refill >= 490 && refill <= 510, "global rate: about 500 refilled after 1s");
}

void test_rate_limit_per_source(Clock::time_point t0) {
    BeaconPeerTable t;
    uint32_t src = ip(192, 168, 5, 5);
    int admitted = 0;
    int limited = 0;
    for (int i = 0; i < 10; ++i) {
        IngestResult r = feed(t, beacon(url_for(src)), src, t0);
        if (r == IngestResult::RateLimited) ++limited;
        else ++admitted;
    }
    check(admitted == 4 && limited == 6, "per-source: 4 admitted and 6 RateLimited");

    BeaconPeerTable t2;
    for (int i = 0; i < 500; ++i) {
        uint32_t s = ip(10, 9, (i >> 8) & 0xFF, i & 0xFF);
        feed(t2, beacon(url_for(ip(10, 250, 0, 1))), s, t0 + std::chrono::milliseconds(i));
    }
    check(t2.tracked_source_count() <= lemon::kMaxSources, "per-source: unpinned LRU never exceeds 64");
}

void test_flood_survival(Clock::time_point t0) {
    BeaconPeerTable t;
    std::vector<uint32_t> legit;
    for (int i = 0; i < 32; ++i) legit.push_back(ip(192, 168, 1, 100 + i));
    for (uint32_t s : legit) {
        feed(t, beacon(url_for(s), "legit"), s, t0);
    }

    bool legit_ok = true;
    uint64_t spoof_bad = 0;
    uint64_t spoof_total = 0;
    uint32_t rotating = 0;
    for (int ms = 1; ms <= 20000; ++ms) {
        auto now = t0 + std::chrono::milliseconds(ms);
        uint32_t s = ip(10, 50 + ((rotating >> 16) & 0x3F), (rotating >> 8) & 0xFF, rotating & 0xFF);
        ++rotating;
        IngestResult r = feed(t, beacon(url_for(s), "spoof"), s, now);
        ++spoof_total;
        if (r == IngestResult::RateLimited || r == IngestResult::TableFull) ++spoof_bad;
        if (ms % 2000 == 0) {
            for (uint32_t l : legit) {
                IngestResult lr = feed(t, beacon(url_for(l), "legit"), l, now);
                if (lr != IngestResult::Refreshed) legit_ok = false;
            }
        }
    }
    auto end = t0 + std::chrono::milliseconds(20000);
    t.expire(end);
    auto hosts = t.hosts();
    size_t legit_rows = 0;
    for (const auto& h : hosts) {
        if (h.hostname == "legit") ++legit_rows;
    }
    check(legit_ok, "flood: every legitimate refresh was admitted");
    check(legit_rows == 32, "flood: all 32 legitimate rows present after 20s");
    check(spoof_bad == spoof_total, "flood: every spoofed beacon was TableFull or RateLimited");
    check(t.evicted() == 0, "flood: evicted stays 0");
}


void test_pinned_junk_flood(Clock::time_point t0) {
    BeaconPeerTable t;
    uint32_t peer = ip(192, 168, 1, 40);
    std::string good = beacon(url_for(peer), "legit");
    check(feed(t, good, peer, t0) == IngestResult::Accepted, "pinned junk: peer accepted");

    std::vector<std::string> junk = {"not json at all", "[]", beacon(url_for(ip(192, 168, 1, 41)), "x"),
                                     beacon("https://192.168.1.40:13305/api/v1/", "x")};
    bool refreshes_ok = true;
    for (int ms = 1; ms <= 20000; ++ms) {
        auto now = t0 + std::chrono::milliseconds(ms);
        feed(t, junk[ms % junk.size()], peer, now);
        if (ms % 2000 == 0 && feed(t, good, peer, now) != IngestResult::Refreshed) refreshes_ok = false;
    }
    t.expire(t0 + std::chrono::milliseconds(20000));
    check(refreshes_ok, "pinned junk: every real refresh admitted during a 1000 pps junk flood");
    check(t.size() == 1, "pinned junk: legitimate row survives the flood");
}

void test_single_source_flood(Clock::time_point t0) {
    BeaconPeerTable t;
    uint32_t noisy = ip(10, 0, 0, 7);
    for (int i = 0; i < 5000; ++i) {
        feed(t, "garbage", noisy, t0);
    }
    check(t.stat(IngestResult::RateLimited) >= 5000 - static_cast<uint64_t>(lemon::kPerSourceBurst),
          "single source: the noisy host is limited to its own bucket");
    uint32_t fresh = ip(10, 0, 0, 8);
    check(feed(t, beacon(url_for(fresh)), fresh, t0) == IngestResult::Accepted,
          "single source: a new peer is still accepted during the flood");
}
} // namespace

int main() {
    std::puts("=== RUNNING BEACON PEER TABLE TESTS ===");
    auto t0 = Clock::now();

    test_basic(t0);
    test_source_class(t0);
    test_url_binding(t0);
    test_payload_structure(t0);
    test_hostname_sanitising(t0);
    test_broadcaster_payload();
    test_self_filter(t0);
    test_expiry(t0);
    test_cap_and_eviction(t0);
    test_rate_limit_global(t0);
    test_rate_limit_per_source(t0);
    test_flood_survival(t0);
    test_pinned_junk_flood(t0);
    test_single_source_flood(t0);

    return test_helpers::report_results("beacon peer table");
}
