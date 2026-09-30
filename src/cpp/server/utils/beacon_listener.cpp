#include "lemon/utils/beacon_listener.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "lemon/utils/aixlog.hpp"
#include "lemon/utils/network_beacon.h"

#ifndef _WIN32
    #include <arpa/inet.h>
    #include <net/if.h>
    #include <netinet/in.h>
    #include <poll.h>
    #include <sys/socket.h>
    #include <sys/uio.h>
    #include <unistd.h>
#endif

namespace lemon {

namespace utils {

bool is_rfc1918_ipv4(uint32_t ip) {
    if ((ip & 0xFF000000u) == 0x0A000000u) return true;
    if ((ip & 0xFFF00000u) == 0xAC100000u) return true;
    if ((ip & 0xFFFF0000u) == 0xC0A80000u) return true;
    return false;
}

std::string ipv4_to_string(uint32_t ip) {
    return std::to_string((ip >> 24) & 0xFF) + "." + std::to_string((ip >> 16) & 0xFF) + "." +
           std::to_string((ip >> 8) & 0xFF) + "." + std::to_string(ip & 0xFF);
}

} // namespace utils

namespace {

using steady_clock = std::chrono::steady_clock;

bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

bool parse_decimal(const std::string& s, size_t& pos, size_t max_digits, uint32_t& out) {
    size_t start = pos;
    uint32_t value = 0;
    while (pos < s.size() && is_digit(s[pos])) {
        if (pos - start >= max_digits) return false;
        value = value * 10 + static_cast<uint32_t>(s[pos] - '0');
        ++pos;
    }
    size_t digits = pos - start;
    if (digits == 0) return false;
    if (digits > 1 && s[start] == '0') return false;
    out = value;
    return true;
}

// Accepts exactly http://<dotted-quad>:<port>/api/v1/ with no leading-zero
// octets, userinfo, alternate paths or trailing data.
bool parse_beacon_url(const std::string& url, uint32_t& host, uint32_t& port) {
    static const std::string kScheme = "http://";
    static const std::string kPath = "/api/v1/";
    if (url.compare(0, kScheme.size(), kScheme) != 0) return false;
    size_t pos = kScheme.size();
    uint32_t addr = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t octet = 0;
        if (!parse_decimal(url, pos, 3, octet) || octet > 255) return false;
        addr = (addr << 8) | octet;
        if (i < 3) {
            if (pos >= url.size() || url[pos] != '.') return false;
            ++pos;
        }
    }
    if (pos >= url.size() || url[pos] != ':') return false;
    ++pos;
    uint32_t p = 0;
    if (!parse_decimal(url, pos, 5, p) || p == 0 || p > 65535) return false;
    if (url.compare(pos, std::string::npos, kPath) != 0) return false;
    host = addr;
    port = p;
    return true;
}

bool valid_instance_id(const std::string& id) {
    if (id.empty() || id.size() > 32) return false;
    return std::all_of(id.begin(), id.end(), [](char c) {
        return is_digit(c) || (c >= 'a' && c <= 'f');
    });
}

std::string sanitize_hostname(const std::string& raw) {
    std::string out;
    out.reserve(std::min<size_t>(raw.size(), 253));
    for (char c : raw) {
        if (out.size() >= 253) break;
        bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || is_digit(c) ||
                  c == '.' || c == '_' || c == '-';
        out.push_back(ok ? c : '_');
    }
    if (out.empty()) out = "_";
    return out;
}

void refill(double& tokens, steady_clock::time_point& last, steady_clock::time_point now,
            double rate, double burst) {
    if (now > last) {
        double elapsed = std::chrono::duration<double>(now - last).count();
        tokens = std::min(burst, tokens + elapsed * rate);
        last = now;
    }
}

uint32_t prefix_mask(int prefix) {
    return prefix <= 0 ? 0u : 0xFFFFFFFFu << (32 - prefix);
}

bool parse_ipv4_literal(const std::string& s, uint32_t& out) {
    size_t pos = 0;
    uint32_t addr = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t octet = 0;
        if (!parse_decimal(s, pos, 3, octet) || octet > 255) return false;
        addr = (addr << 8) | octet;
        if (i < 3) {
            if (pos >= s.size() || s[pos] != '.') return false;
            ++pos;
        }
    }
    if (pos != s.size()) return false;
    out = addr;
    return true;
}

// A relay may only vouch for RFC1918 hosts, so a range must sit wholly
// inside one RFC1918 block.
bool parse_host_range(const std::string& text, BeaconHostRange& range, std::string& why) {
    auto slash = text.find('/');
    uint32_t addr = 0;
    if (!parse_ipv4_literal(text.substr(0, slash), addr)) {
        why = "is not an IPv4 address or CIDR like 192.168.79.20 or 192.168.79.0/24";
        return false;
    }
    int prefix = 32;
    if (slash != std::string::npos) {
        size_t pos = slash + 1;
        uint32_t p = 0;
        if (!parse_decimal(text, pos, 2, p) || pos != text.size() || p > 32) {
            why = "has an invalid prefix length (0-32)";
            return false;
        }
        prefix = static_cast<int>(p);
    }
    if ((addr & ~prefix_mask(prefix)) != 0) {
        why = "has host bits set; use " + utils::ipv4_to_string(addr & prefix_mask(prefix)) + "/" +
              std::to_string(prefix);
        return false;
    }
    static const std::pair<uint32_t, int> kBlocks[] = {{0x0A000000u, 8}, {0xAC100000u, 12}, {0xC0A80000u, 16}};
    for (const auto& [net, len] : kBlocks) {
        if (prefix >= len && (addr & prefix_mask(len)) == net) {
            range.network = addr;
            range.prefix = prefix;
            return true;
        }
    }
    why = "is not inside 10.0.0.0/8, 172.16.0.0/12 or 192.168.0.0/16";
    return false;
}

int64_t epoch_seconds(std::chrono::system_clock::time_point tp) {
    return std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch()).count();
}

} // namespace

const char* ingest_result_name(IngestResult result) {
    switch (result) {
        case IngestResult::Accepted: return "accepted";
        case IngestResult::Refreshed: return "refreshed";
        case IngestResult::Self: return "self";
        case IngestResult::BadSource: return "bad_source";
        case IngestResult::BadPayload: return "bad_payload";
        case IngestResult::UrlMismatch: return "url_mismatch";
        case IngestResult::RateLimited: return "rate_limited";
        case IngestResult::TableFull: return "table_full";
        case IngestResult::WrongInterface: return "wrong_interface";
        case IngestResult::UnknownInterface: return "unknown_interface";
        case IngestResult::RelayedAccepted: return "relayed_accepted";
        case IngestResult::RelayedRefreshed: return "relayed_refreshed";
        case IngestResult::RelayHostRefused: return "relay_host_refused";
    }
    return "unknown";
}

bool beacon_arrival_accepted(bool allowlist_active, unsigned ifindex,
                             const std::vector<unsigned>& listened_ifindexes) {
    if (ifindex == 0) return !allowlist_active;
    return std::find(listened_ifindexes.begin(), listened_ifindexes.end(), ifindex) != listened_ifindexes.end();
}

bool BeaconHostRange::contains(uint32_t ip) const {
    return (ip & prefix_mask(prefix)) == network;
}

std::string BeaconHostRange::to_string() const {
    return utils::ipv4_to_string(network) + "/" + std::to_string(prefix);
}

std::vector<BeaconTrustedRelay> parse_beacon_trusted_relays(const nlohmann::json& value) {
    auto fail = [](const std::string& why) {
        throw std::invalid_argument("'beacon_trusted_relays' " + why);
    };
    if (!value.is_array()) fail("must be an array of {\"source\", \"allow_hosts\"} objects");
    if (value.size() > kMaxTrustedRelays) {
        fail("may list at most " + std::to_string(kMaxTrustedRelays) + " relays");
    }
    std::vector<BeaconTrustedRelay> relays;
    for (size_t i = 0; i < value.size(); ++i) {
        const auto& entry = value[i];
        const std::string where = "entry " + std::to_string(i);
        if (!entry.is_object()) fail(where + " must be an object");
        for (const auto& item : entry.items()) {
            if (item.key() != "source" && item.key() != "allow_hosts") {
                fail(where + " has unknown field '" + item.key() + "'");
            }
        }
        auto src_it = entry.find("source");
        if (src_it == entry.end() || !src_it->is_string()) fail(where + " needs \"source\": an IPv4 address");
        BeaconTrustedRelay relay;
        const std::string src = src_it->get<std::string>();
        if (!parse_ipv4_literal(src, relay.source)) {
            fail(where + " source '" + src + "' is not an IPv4 address like 192.168.60.10");
        }
        if (!utils::is_rfc1918_ipv4(relay.source)) {
            fail(where + " source '" + src + "' is not an RFC1918 address");
        }
        for (const auto& other : relays) {
            if (other.source == relay.source) fail(where + " repeats source '" + src + "'");
        }
        auto hosts_it = entry.find("allow_hosts");
        if (hosts_it == entry.end() || !hosts_it->is_array() || hosts_it->empty()) {
            fail(where + " needs \"allow_hosts\": a non-empty array of IPv4 addresses or CIDRs");
        }
        if (hosts_it->size() > kMaxRelayAllowHosts) {
            fail(where + " allow_hosts may list at most " + std::to_string(kMaxRelayAllowHosts) + " entries");
        }
        for (const auto& h : *hosts_it) {
            if (!h.is_string()) fail(where + " allow_hosts entries must be strings");
            const std::string text = h.get<std::string>();
            BeaconHostRange range;
            std::string why;
            if (!parse_host_range(text, range, why)) fail(where + " allow_hosts '" + text + "' " + why);
            relay.allow_hosts.push_back(range);
        }
        relays.push_back(std::move(relay));
    }
    return relays;
}

nlohmann::json beacon_trusted_relays_to_json(const std::vector<BeaconTrustedRelay>& relays) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& relay : relays) {
        nlohmann::json hosts = nlohmann::json::array();
        for (const auto& range : relay.allow_hosts) hosts.push_back(range.to_string());
        out.push_back({{"source", utils::ipv4_to_string(relay.source)}, {"allow_hosts", hosts}});
    }
    return out;
}

BeaconPeerTable::BeaconPeerTable() = default;

namespace {
bool relay_allows(const BeaconTrustedRelay& relay, uint32_t url_host) {
    if (!utils::is_rfc1918_ipv4(url_host)) return false;
    return std::any_of(relay.allow_hosts.begin(), relay.allow_hosts.end(),
                       [url_host](const BeaconHostRange& r) { return r.contains(url_host); });
}
} // namespace

void BeaconPeerTable::set_trusted_relays(std::vector<BeaconTrustedRelay> relays) {
    std::lock_guard<std::mutex> lock(mutex_);
    relays_ = std::move(relays);
    for (auto it = rows_.begin(); it != rows_.end();) {
        const auto* relay = it->second.via == 0 ? nullptr : relay_for_locked(it->second.via);
        if (it->second.via != 0 && (relay == nullptr || !relay_allows(*relay, it->second.url_host))) {
            it = rows_.erase(it);
        } else {
            ++it;
        }
    }
}

const BeaconTrustedRelay* BeaconPeerTable::relay_for_locked(uint32_t src_ip) const {
    for (const auto& relay : relays_) {
        if (relay.source == src_ip) return &relay;
    }
    return nullptr;
}

void BeaconPeerTable::set_self(const std::string& instance_id, int self_port) {
    std::lock_guard<std::mutex> lock(mutex_);
    self_instance_id_ = instance_id;
    self_port_ = self_port;
}

void BeaconPeerTable::set_self_port(int port) {
    std::lock_guard<std::mutex> lock(mutex_);
    self_port_ = port;
}

void BeaconPeerTable::set_local_addresses(std::vector<uint32_t> addrs) {
    std::lock_guard<std::mutex> lock(mutex_);
    local_addresses_ = std::move(addrs);
}

bool BeaconPeerTable::is_pinned_locked(uint32_t src_ip) const {
    for (const auto& [url, row] : rows_) {
        if (row.source_ip_host_order == src_ip) return true;
    }
    return false;
}

BeaconPeerTable::SourceState& BeaconPeerTable::source_state_locked(uint32_t src_ip,
                                                                   steady_clock::time_point now) {
    auto it = sources_.find(src_ip);
    if (it == sources_.end()) {
        size_t unpinned = 0;
        auto oldest = sources_.end();
        for (auto s = sources_.begin(); s != sources_.end(); ++s) {
            if (is_pinned_locked(s->first)) continue;
            ++unpinned;
            if (oldest == sources_.end() || s->second.last_seen < oldest->second.last_seen) {
                oldest = s;
            }
        }
        if (unpinned >= kMaxSources && oldest != sources_.end()) {
            sources_.erase(oldest);
        }
        it = sources_.emplace(src_ip, SourceState{{0.0, now}, now}).first;
        it->second.bucket.tokens = relay_for_locked(src_ip) != nullptr ? kRelayBurst : kPerSourceBurst;
    }
    auto& st = it->second;
    if (relay_for_locked(src_ip) != nullptr) {
        refill(st.bucket.tokens, st.bucket.last, now, kRelayRate, kRelayBurst);
    } else {
        // A source dropped from beacon_trusted_relays must not keep its larger relay burst.
        st.bucket.tokens = std::min(st.bucket.tokens, kPerSourceBurst);
        refill(st.bucket.tokens, st.bucket.last, now, kPerSourceRate, kPerSourceBurst);
    }
    st.last_seen = now;
    return st;
}

BeaconPeerTable::SourceState& BeaconPeerTable::relayed_host_state_locked(uint32_t url_host,
                                                                         steady_clock::time_point now) {
    auto it = relayed_hosts_.find(url_host);
    if (it == relayed_hosts_.end()) {
        if (relayed_hosts_.size() >= kMaxSources) {
            auto oldest = std::min_element(relayed_hosts_.begin(), relayed_hosts_.end(), [](const auto& a, const auto& b) {
                return a.second.last_seen < b.second.last_seen;
            });
            relayed_hosts_.erase(oldest);
        }
        it = relayed_hosts_.emplace(url_host, SourceState{{kPerSourceBurst, now}, now}).first;
    }
    auto& st = it->second;
    refill(st.bucket.tokens, st.bucket.last, now, kPerSourceRate, kPerSourceBurst);
    st.last_seen = now;
    return st;
}

bool BeaconPeerTable::take_global_token_locked(steady_clock::time_point now) {
    if (!global_bucket_primed_) {
        global_bucket_ = {kGlobalBurst, now};
        global_bucket_primed_ = true;
    }
    refill(global_bucket_.tokens, global_bucket_.last, now, kGlobalRate, kGlobalBurst);
    if (global_bucket_.tokens < 1.0) return false;
    global_bucket_.tokens -= 1.0;
    return true;
}

IngestResult BeaconPeerTable::record_locked(IngestResult result) {
    ++stats_[result];
    return result;
}

IngestResult BeaconPeerTable::ingest(const char* buf, std::size_t len, bool truncated, uint32_t src_ip,
                                     steady_clock::time_point now) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!utils::is_rfc1918_ipv4(src_ip)) return record_locked(IngestResult::BadSource);
    expire_locked(now);

    // Pinned sources (those owning a live row) skip the global bucket so a
    // flood from other addresses cannot starve refreshes from known peers.
    // Their per-source token is only spent on a structurally valid beacon:
    // junk forged with a known peer's source IP must not drain the budget of
    // that peer's real refreshes, so it is charged to the global bucket.
    if (is_pinned_locked(src_ip)) {
        if (source_state_locked(src_ip, now).bucket.tokens < 1.0) {
            return record_locked(IngestResult::RateLimited);
        }
        IngestResult result = ingest_admitted_locked(buf, len, truncated, src_ip, now);
        if (result == IngestResult::BadPayload || result == IngestResult::UrlMismatch ||
            result == IngestResult::RelayHostRefused) {
            take_global_token_locked(now);
        } else {
            auto it = sources_.find(src_ip);
            if (it != sources_.end()) it->second.bucket.tokens -= 1.0;
        }
        return result;
    }

    // The per-source check comes first so one noisy host only ever spends its
    // own small bucket, never the global budget that admits new peers.
    SourceState& st = source_state_locked(src_ip, now);
    if (st.bucket.tokens < 1.0) return record_locked(IngestResult::RateLimited);
    if (!take_global_token_locked(now)) return record_locked(IngestResult::RateLimited);
    st.bucket.tokens -= 1.0;
    return ingest_admitted_locked(buf, len, truncated, src_ip, now);
}

IngestResult BeaconPeerTable::ingest_admitted_locked(const char* buf, std::size_t len, bool truncated,
                                                     uint32_t src_ip, steady_clock::time_point now) {
    if (buf == nullptr || len == 0 || len >= kMaxDatagram || truncated) {
        return record_locked(IngestResult::BadPayload);
    }

    // Every field we keep is ASCII, and the hostname sanitiser maps each
    // non-ASCII byte to '_' anyway, so folding high bytes first lets a peer
    // with a non-UTF-8 hostname still be listed instead of failing the parse.
    std::string text(buf, len);
    for (char& c : text) {
        if (static_cast<unsigned char>(c) >= 0x80) c = '_';
    }
    nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object()) return record_locked(IngestResult::BadPayload);
    auto service = j.find("service");
    auto url_it = j.find("url");
    auto host_it = j.find("hostname");
    if (service == j.end() || !service->is_string() || service->get<std::string>() != "lemonade" ||
        url_it == j.end() || !url_it->is_string() || host_it == j.end() || !host_it->is_string()) {
        return record_locked(IngestResult::BadPayload);
    }
    std::string instance_id;
    auto id_it = j.find("instance_id");
    if (id_it != j.end()) {
        if (!id_it->is_string()) return record_locked(IngestResult::BadPayload);
        instance_id = id_it->get<std::string>();
        if (!valid_instance_id(instance_id)) return record_locked(IngestResult::BadPayload);
    }

    const std::string url = url_it->get<std::string>();
    uint32_t url_host = 0;
    uint32_t url_port = 0;
    if (!parse_beacon_url(url, url_host, url_port)) return record_locked(IngestResult::BadPayload);
    const BeaconTrustedRelay* relay = nullptr;
    if (url_host != src_ip) {
        relay = relay_for_locked(src_ip);
        if (relay == nullptr) return record_locked(IngestResult::UrlMismatch);
        if (!relay_allows(*relay, url_host)) return record_locked(IngestResult::RelayHostRefused);
    }
    const uint32_t via = relay != nullptr ? src_ip : 0;

    std::string hostname = sanitize_hostname(host_it->get<std::string>());

    if (!instance_id.empty()) {
        if (instance_id == self_instance_id_) return record_locked(IngestResult::Self);
    } else if (static_cast<int>(url_port) == self_port_ &&
               std::find(local_addresses_.begin(), local_addresses_.end(), url_host) !=
                   local_addresses_.end()) {
        return record_locked(IngestResult::Self);
    }

    if (relay != nullptr) {
        SourceState& host_state = relayed_host_state_locked(url_host, now);
        if (host_state.bucket.tokens < 1.0) return record_locked(IngestResult::RateLimited);
        host_state.bucket.tokens -= 1.0;
    }

    auto wall = std::chrono::system_clock::now();
    auto existing = rows_.find(url);
    if (existing != rows_.end()) {
        auto& row = existing->second;
        row.hostname = hostname;
        row.instance_id = instance_id;
        row.source_ip = utils::ipv4_to_string(src_ip);
        row.source_ip_host_order = src_ip;
        row.url_host = url_host;
        row.via = via;
        row.last_seen = now;
        row.last_seen_wall = wall;
        return record_locked(relay != nullptr ? IngestResult::RelayedRefreshed : IngestResult::Refreshed);
    }

    if (relay != nullptr) {
        auto vouched = std::count_if(rows_.begin(), rows_.end(), [&](const auto& r) {
            return r.second.via == src_ip && now - r.second.last_seen <= kPeerTtl;
        });
        if (static_cast<std::size_t>(vouched) >= kMaxRelayedHostsPerRelay) {
            return record_locked(IngestResult::TableFull);
        }
    }

    if (rows_.size() >= kMaxHosts) {
        expire_locked(now);
    }
    if (rows_.size() >= kMaxHosts) {
        auto oldest = rows_.begin();
        for (auto r = rows_.begin(); r != rows_.end(); ++r) {
            if (r->second.last_seen < oldest->second.last_seen) oldest = r;
        }
        if (now - oldest->second.last_seen > kEvictStale) {
            rows_.erase(oldest);
            ++evicted_;
        } else {
            return record_locked(IngestResult::TableFull);
        }
    }

    HeardHost row;
    row.hostname = hostname;
    row.url = url;
    row.source_ip = utils::ipv4_to_string(src_ip);
    row.source_ip_host_order = src_ip;
    row.url_host = url_host;
    row.via = via;
    row.instance_id = instance_id;
    row.first_seen_wall = wall;
    row.last_seen_wall = wall;
    row.last_seen = now;
    rows_.emplace(url, std::move(row));
    return record_locked(relay != nullptr ? IngestResult::RelayedAccepted : IngestResult::Accepted);
}

void BeaconPeerTable::expire_locked(steady_clock::time_point now) {
    for (auto it = rows_.begin(); it != rows_.end();) {
        if (now - it->second.last_seen > kPeerTtl) {
            it = rows_.erase(it);
        } else {
            ++it;
        }
    }
}

void BeaconPeerTable::expire(steady_clock::time_point now) {
    std::lock_guard<std::mutex> lock(mutex_);
    expire_locked(now);
}

IngestResult BeaconPeerTable::reject(IngestResult result) {
    std::lock_guard<std::mutex> lock(mutex_);
    return record_locked(result);
}

nlohmann::json BeaconPeerTable::to_json(steady_clock::time_point now) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const HeardHost*> live;
    for (const auto& [url, row] : rows_) {
        if (now - row.last_seen <= kPeerTtl) live.push_back(&row);
    }
    std::sort(live.begin(), live.end(), [](const HeardHost* a, const HeardHost* b) {
        return std::tie(a->hostname, a->url) < std::tie(b->hostname, b->url);
    });

    nlohmann::json hosts = nlohmann::json::array();
    std::set<std::string> ids;
    size_t without_id = 0;
    for (const auto* row : live) {
        if (row->instance_id.empty()) {
            ++without_id;
        } else {
            ids.insert(row->instance_id);
        }
        hosts.push_back({
            {"hostname", row->hostname},
            {"url", row->url},
            {"source_ip", row->source_ip},
            {"via", row->via == 0 ? nlohmann::json(nullptr) : nlohmann::json(utils::ipv4_to_string(row->via))},
            {"instance_id", row->instance_id.empty() ? nlohmann::json(nullptr) : nlohmann::json(row->instance_id)},
            {"first_seen", epoch_seconds(row->first_seen_wall)},
            {"last_seen", epoch_seconds(row->last_seen_wall)},
            {"age_seconds", std::chrono::duration<double>(now - row->last_seen).count()},
        });
    }

    nlohmann::json stats = nlohmann::json::object();
    for (auto r : {IngestResult::Accepted, IngestResult::Refreshed, IngestResult::Self,
                   IngestResult::BadSource, IngestResult::BadPayload, IngestResult::UrlMismatch,
                   IngestResult::RateLimited, IngestResult::TableFull, IngestResult::WrongInterface,
                   IngestResult::UnknownInterface, IngestResult::RelayedAccepted,
                   IngestResult::RelayedRefreshed, IngestResult::RelayHostRefused}) {
        auto it = stats_.find(r);
        stats[ingest_result_name(r)] = it == stats_.end() ? 0 : it->second;
    }
    stats["evicted"] = evicted_;

    return {
        {"self_instance_id", self_instance_id_},
        {"self_port", self_port_},
        {"ttl_seconds", kPeerTtl.count()},
        {"max_hosts", kMaxHosts},
        {"distinct_instances", ids.size() + without_id},
        {"trusted_relays", beacon_trusted_relays_to_json(relays_)},
        {"hosts", hosts},
        {"stats", stats},
    };
}

std::vector<HeardHost> BeaconPeerTable::hosts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<HeardHost> out;
    for (const auto& [url, row] : rows_) out.push_back(row);
    return out;
}

std::size_t BeaconPeerTable::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return rows_.size();
}

std::size_t BeaconPeerTable::tracked_source_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sources_.size();
}

uint64_t BeaconPeerTable::stat(IngestResult result) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stats_.find(result);
    return it == stats_.end() ? 0 : it->second;
}

uint64_t BeaconPeerTable::evicted() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return evicted_;
}

void BeaconPeerTable::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    rows_.clear();
    sources_.clear();
    relayed_hosts_.clear();
    stats_.clear();
    evicted_ = 0;
    global_bucket_primed_ = false;
}

BeaconListener::BeaconListener() = default;

BeaconListener::~BeaconListener() {
    stop();
}

void BeaconListener::set_trusted_relays(std::vector<BeaconTrustedRelay> relays) {
    table_.set_trusted_relays(std::move(relays));
}

bool BeaconListener::is_running() const {
    return running_.load();
}

void BeaconListener::set_self_port(int port) {
    table_.set_self_port(port);
}

void BeaconListener::set_interface_allowlist(std::vector<std::string> names) {
    {
        std::lock_guard<std::mutex> lock(allowlist_mtx_);
        if (names == allowlist_) return;
        allowlist_ = std::move(names);
    }
    refresh_requested_ = true;
}

void BeaconListener::start(const std::string& instance_id, int self_port) {
    std::lock_guard<std::mutex> lock(lifecycle_mtx_);
    table_.set_self(instance_id, self_port);
    if (running_.load()) return;

#ifdef _WIN32
    {
        std::lock_guard<std::mutex> status_lock(status_mtx_);
        supported_ = false;
        error_ = "beacon_listen is not supported on Windows in this release";
        sockets_status_ = nlohmann::json::array();
    }
    running_ = true;
    LOG(WARNING, "BeaconListener") << "beacon_listen is not supported on Windows in this release"
                                   << std::endl;
#else
    {
        std::lock_guard<std::mutex> status_lock(status_mtx_);
        supported_ = true;
        error_.clear();
        start_failed_ = false;
        sockets_status_ = nlohmann::json::array();
    }
    {
        std::lock_guard<std::mutex> cv_lock(cv_mtx_);
        stop_ = false;
    }
    // running_ is only set once the worker exists: a std::system_error from
    // thread creation (EAGAIN on a constrained gateway) must leave the
    // listener cleanly stopped, and must not escape into Server::run().
    try {
        worker_ = std::thread(&BeaconListener::thread_loop, this);
    } catch (const std::exception& e) {
        {
            std::lock_guard<std::mutex> status_lock(status_mtx_);
            error_ = std::string("failed to start beacon listener thread: ") + e.what();
            start_failed_ = true;
        }
        LOG(ERROR, "BeaconListener") << "Failed to start beacon listener thread: " << e.what() << std::endl;
        return;
    }
    running_ = true;
    LOG(INFO, "BeaconListener") << "Listening for LAN beacons on UDP " << kBeaconPort << std::endl;
#endif
}

void BeaconListener::stop() {
    std::lock_guard<std::mutex> lock(lifecycle_mtx_);
    {
        std::lock_guard<std::mutex> status_lock(status_mtx_);
        if (start_failed_) {
            start_failed_ = false;
            error_.clear();
        }
    }
    if (!running_.load()) return;
    {
        std::lock_guard<std::mutex> cv_lock(cv_mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    running_ = false;
    table_.clear();
    last_warning_.clear();
    {
        std::lock_guard<std::mutex> status_lock(status_mtx_);
        sockets_status_ = nlohmann::json::array();
        interfaces_status_ = nlohmann::json::array();
        unmatched_status_ = nlohmann::json::array();
        allowlist_error_.clear();
        error_.clear();
    }
    unmatched_.clear();
    LOG(INFO, "BeaconListener") << "Stopped listening for LAN beacons" << std::endl;
}

nlohmann::json BeaconListener::status_json() const {
    if (!running_.load()) {
        std::lock_guard<std::mutex> status_lock(status_mtx_);
        if (start_failed_) {
            return {{"enabled", true}, {"supported", supported_}, {"listening", false},
                    {"error", error_}, {"sockets", nlohmann::json::array()}};
        }
        return {{"enabled", false}};
    }
    auto now = std::chrono::steady_clock::now();
    nlohmann::json out = table_.to_json(now);
    {
        std::lock_guard<std::mutex> lock(allowlist_mtx_);
        out["interface_allowlist"] = allowlist_;
    }
    std::lock_guard<std::mutex> status_lock(status_mtx_);
    bool listening = false;
    for (const auto& s : sockets_status_) {
        if (s.value("bound", false)) listening = true;
    }
    out["enabled"] = true;
    out["supported"] = supported_;
    out["listening"] = listening;
    if (!error_.empty()) {
        out["error"] = error_;
    } else if (!allowlist_error_.empty()) {
        out["error"] = allowlist_error_;
    } else {
        out["error"] = nullptr;
    }
    out["sockets"] = sockets_status_;
    out["interfaces"] = interfaces_status_;
    out["unmatched_interfaces"] = unmatched_status_;
    return out;
}

void BeaconListener::wait_slice() {
    std::unique_lock<std::mutex> lock(cv_mtx_);
    cv_.wait_for(lock, kPollSlice, [this] { return stop_.load(); });
}

void BeaconListener::log_rejection(IngestResult result, uint32_t src_ip) {
    switch (result) {
        case IngestResult::Accepted:
        case IngestResult::Refreshed:
        case IngestResult::RelayedAccepted:
        case IngestResult::RelayedRefreshed:
        case IngestResult::Self:
            return;
        default:
            break;
    }
    auto now = std::chrono::steady_clock::now();
    auto it = last_warning_.find(result);
    if (it != last_warning_.end() && now - it->second < std::chrono::seconds(60)) return;
    last_warning_[result] = now;
    LOG(WARNING, "BeaconListener") << "Dropped beacon from " << utils::ipv4_to_string(src_ip) << ": "
                                   << ingest_result_name(result)
                                   << " (further drops for this reason suppressed for 60s)" << std::endl;
}

void BeaconListener::publish_socket_status() {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& s : sockets_) {
        arr.push_back({
            {"address", s.address + ":" + std::to_string(kBeaconPort)},
            {"bound", s.bound},
            {"error", s.error.empty() ? nlohmann::json(nullptr) : nlohmann::json(s.error)},
        });
    }
    nlohmann::json ifaces = nlohmann::json::array();
    for (const auto& i : interfaces_) {
        ifaces.push_back({{"name", i.name}, {"address", i.address}, {"netmask", i.netmask},
                          {"broadcast", i.broadcast}});
    }
    std::string allowlist_error;
    if (allowlist_active_ && interfaces_.empty()) {
        allowlist_error = "no interface named in beacon_listen_interfaces has an RFC1918 address:";
        for (const auto& name : unmatched_) allowlist_error += " " + name;
    }
    std::lock_guard<std::mutex> status_lock(status_mtx_);
    sockets_status_ = std::move(arr);
    interfaces_status_ = std::move(ifaces);
    unmatched_status_ = unmatched_;
    allowlist_error_ = std::move(allowlist_error);
}

#ifdef _WIN32

void BeaconListener::thread_loop() {}
void BeaconListener::refresh_sockets() {}
void BeaconListener::close_all_sockets() {}

#else

void BeaconListener::close_all_sockets() {
    for (auto& s : sockets_) {
        if (s.fd >= 0) close(s.fd);
        s.fd = -1;
        s.bound = false;
    }
    sockets_.clear();
}

void BeaconListener::refresh_sockets() {
    NetworkBeacon probe;
    auto ifaces = probe.getLocalRFC1918Interfaces();
    std::vector<std::string> allowlist;
    {
        std::lock_guard<std::mutex> lock(allowlist_mtx_);
        allowlist = allowlist_;
    }
    allowlist_active_ = !allowlist.empty();

    std::vector<uint32_t> local_addrs;
    std::set<uint32_t> desired;
    std::set<std::string> present;
    interfaces_.clear();
    for (const auto& iface : ifaces) {
        in_addr ip{};
        if (inet_pton(AF_INET, iface.ipAddress.c_str(), &ip) != 1) continue;
        uint32_t ip_host = ntohl(ip.s_addr);
        local_addrs.push_back(ip_host);
        if ((ip_host & 0xFF000000u) == 0x7F000000u) continue;
        present.insert(iface.name);
        if (allowlist_active_ &&
            std::find(allowlist.begin(), allowlist.end(), iface.name) == allowlist.end()) {
            continue;
        }
        in_addr bcast{};
        if (inet_pton(AF_INET, iface.broadcastAddress.c_str(), &bcast) != 1) continue;
        uint32_t bcast_host = ntohl(bcast.s_addr);
        if (bcast_host != ip_host) desired.insert(bcast_host);
        interfaces_.push_back({iface.name, if_nametoindex(iface.name.c_str()), iface.ipAddress, iface.netmask,
                               iface.broadcastAddress});
    }
    if (!interfaces_.empty()) desired.insert(0xFFFFFFFFu);
    table_.set_local_addresses(std::move(local_addrs));

    std::vector<std::string> unmatched;
    for (const auto& name : allowlist) {
        if (present.count(name) == 0) unmatched.push_back(name);
    }
    if (unmatched != unmatched_) {
        if (!unmatched.empty()) {
            std::string names;
            for (const auto& name : unmatched) names += (names.empty() ? "" : ", ") + name;
            LOG(WARNING, "BeaconListener") << "beacon_listen_interfaces names no interface with an RFC1918 address: "
                                           << names << (interfaces_.empty() ? " (listening on nothing)" : "")
                                           << std::endl;
        }
        unmatched_ = std::move(unmatched);
    }

    for (auto it = sockets_.begin(); it != sockets_.end();) {
        if (desired.count(it->bind_ip) == 0) {
            if (it->fd >= 0) close(it->fd);
            it = sockets_.erase(it);
        } else {
            ++it;
        }
    }

    for (uint32_t addr : desired) {
        auto existing = std::find_if(sockets_.begin(), sockets_.end(),
                                     [addr](const SocketState& s) { return s.bind_ip == addr; });
        if (existing == sockets_.end()) {
            SocketState s;
            s.address = utils::ipv4_to_string(addr);
            s.bind_ip = addr;
            sockets_.push_back(s);
            existing = sockets_.end() - 1;
        }
        if (existing->bound) continue;

        int fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (fd < 0) {
            existing->error = std::string("socket: ") + std::strerror(errno);
            continue;
        }
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#ifdef IP_PKTINFO
        if (setsockopt(fd, IPPROTO_IP, IP_PKTINFO, &one, sizeof(one)) != 0) {
            LOG(WARNING, "BeaconListener") << "IP_PKTINFO unavailable on " << existing->address << ": "
                                           << std::strerror(errno)
                                           << "; beacons on it are dropped while beacon_listen_interfaces is set"
                                           << std::endl;
        }
#endif
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port = htons(kBeaconPort);
        sa.sin_addr.s_addr = htonl(addr);
        if (bind(fd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) {
            existing->error = std::string("bind: ") + std::strerror(errno);
            close(fd);
            continue;
        }
        existing->fd = fd;
        existing->bound = true;
        existing->error.clear();
    }
    publish_socket_status();
}

bool BeaconListener::arrived_on_listened_interface(unsigned ifindex) const {
    std::vector<unsigned> listened;
    for (const auto& i : interfaces_) listened.push_back(i.index);
    return beacon_arrival_accepted(allowlist_active_, ifindex, listened);
}

void BeaconListener::thread_loop() {
    auto next_refresh = std::chrono::steady_clock::now();
    char buf[kMaxDatagram];

    while (!stop_.load()) {
        auto now = std::chrono::steady_clock::now();
        if (now >= next_refresh || refresh_requested_.exchange(false)) {
            refresh_sockets();
            next_refresh = now + kIfaceRefresh;
        }

        std::vector<pollfd> fds;
        std::vector<size_t> owners;
        for (size_t i = 0; i < sockets_.size(); ++i) {
            if (!sockets_[i].bound) continue;
            fds.push_back({sockets_[i].fd, POLLIN, 0});
            owners.push_back(i);
        }
        if (fds.empty()) {
            wait_slice();
            continue;
        }

        int rc = poll(fds.data(), static_cast<nfds_t>(fds.size()),
                      static_cast<int>(kPollSlice.count()));
        if (rc < 0) {
            if (errno != EINTR) wait_slice();
            continue;
        }

        bool socket_state_changed = false;
        for (size_t k = 0; k < fds.size(); ++k) {
            if (fds[k].revents == 0) continue;
            auto& sock = sockets_[owners[k]];
            if (fds[k].revents & (POLLERR | POLLNVAL)) {
                close(sock.fd);
                sock.fd = -1;
                sock.bound = false;
                sock.error = "socket error; rebinding at next refresh";
                socket_state_changed = true;
                continue;
            }
            for (int drained = 0; drained < 64; ++drained) {
                sockaddr_in from{};
                iovec iov{buf, sizeof(buf)};
                alignas(cmsghdr) char control[256];
                msghdr msg{};
                msg.msg_name = &from;
                msg.msg_namelen = sizeof(from);
                msg.msg_iov = &iov;
                msg.msg_iovlen = 1;
                msg.msg_control = control;
                msg.msg_controllen = sizeof(control);
                int flags = MSG_DONTWAIT;
#ifdef __linux__
                flags |= MSG_TRUNC;
#endif
                ssize_t n = recvmsg(sock.fd, &msg, flags);
                if (n < 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) break;
                    sock.error = std::string("recvfrom: ") + std::strerror(errno);
                    close(sock.fd);
                    sock.fd = -1;
                    sock.bound = false;
                    socket_state_changed = true;
                    break;
                }
                size_t received = static_cast<size_t>(n);
                bool truncated = received >= sizeof(buf);
                size_t len = std::min(received, sizeof(buf));
                uint32_t src = ntohl(from.sin_addr.s_addr);
                unsigned ifindex = 0;
#ifdef IP_PKTINFO
                for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c != nullptr; c = CMSG_NXTHDR(&msg, c)) {
                    if (c->cmsg_level == IPPROTO_IP && c->cmsg_type == IP_PKTINFO) {
                        in_pktinfo info{};
                        std::memcpy(&info, CMSG_DATA(c), sizeof(info));
                        ifindex = static_cast<unsigned>(info.ipi_ifindex);
                    }
                }
#endif
                IngestResult result = arrived_on_listened_interface(ifindex)
                                          ? table_.ingest(buf, len, truncated, src, std::chrono::steady_clock::now())
                                          : table_.reject(ifindex == 0 ? IngestResult::UnknownInterface
                                                                       : IngestResult::WrongInterface);
                log_rejection(result, src);
            }
        }
        if (socket_state_changed) publish_socket_status();
        table_.expire(std::chrono::steady_clock::now());
    }

    close_all_sockets();
}

#endif

} // namespace lemon
