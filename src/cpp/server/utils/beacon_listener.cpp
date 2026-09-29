#include "lemon/utils/beacon_listener.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <set>
#include <tuple>
#include <utility>

#include "lemon/utils/aixlog.hpp"
#include "lemon/utils/network_beacon.h"

#ifndef _WIN32
    #include <arpa/inet.h>
    #include <netinet/in.h>
    #include <poll.h>
    #include <sys/socket.h>
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
    }
    return "unknown";
}

BeaconPeerTable::BeaconPeerTable() = default;

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

// Pinned sources (those owning a live row) skip the global bucket so a flood
// from other addresses cannot starve refreshes from already-known peers.
bool BeaconPeerTable::admit_locked(uint32_t src_ip, steady_clock::time_point now) {
    auto it = sources_.find(src_ip);
    if (is_pinned_locked(src_ip)) {
        if (it == sources_.end()) {
            it = sources_.emplace(src_ip, SourceState{{kPerSourceBurst, now}, now}).first;
        }
        auto& st = it->second;
        refill(st.bucket.tokens, st.bucket.last, now, kPerSourceRate, kPerSourceBurst);
        st.last_seen = now;
        if (st.bucket.tokens < 1.0) return false;
        st.bucket.tokens -= 1.0;
        return true;
    }

    if (!global_bucket_primed_) {
        global_bucket_ = {kGlobalBurst, now};
        global_bucket_primed_ = true;
    }
    refill(global_bucket_.tokens, global_bucket_.last, now, kGlobalRate, kGlobalBurst);
    if (global_bucket_.tokens < 1.0) return false;
    global_bucket_.tokens -= 1.0;

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
        it = sources_.emplace(src_ip, SourceState{{kPerSourceBurst, now}, now}).first;
    }
    auto& st = it->second;
    refill(st.bucket.tokens, st.bucket.last, now, kPerSourceRate, kPerSourceBurst);
    st.last_seen = now;
    if (st.bucket.tokens < 1.0) return false;
    st.bucket.tokens -= 1.0;
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
    if (!admit_locked(src_ip, now)) return record_locked(IngestResult::RateLimited);
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
    if (url_host != src_ip) return record_locked(IngestResult::UrlMismatch);

    std::string hostname = sanitize_hostname(host_it->get<std::string>());

    if (!instance_id.empty()) {
        if (instance_id == self_instance_id_) return record_locked(IngestResult::Self);
    } else if (static_cast<int>(url_port) == self_port_ &&
               std::find(local_addresses_.begin(), local_addresses_.end(), src_ip) !=
                   local_addresses_.end()) {
        return record_locked(IngestResult::Self);
    }

    auto wall = std::chrono::system_clock::now();
    auto existing = rows_.find(url);
    if (existing != rows_.end()) {
        auto& row = existing->second;
        row.hostname = hostname;
        row.instance_id = instance_id;
        row.source_ip = utils::ipv4_to_string(src_ip);
        row.source_ip_host_order = src_ip;
        row.last_seen = now;
        row.last_seen_wall = wall;
        return record_locked(IngestResult::Refreshed);
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
    row.instance_id = instance_id;
    row.first_seen_wall = wall;
    row.last_seen_wall = wall;
    row.last_seen = now;
    rows_.emplace(url, std::move(row));
    return record_locked(IngestResult::Accepted);
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
            {"instance_id", row->instance_id.empty() ? nlohmann::json(nullptr) : nlohmann::json(row->instance_id)},
            {"first_seen", epoch_seconds(row->first_seen_wall)},
            {"last_seen", epoch_seconds(row->last_seen_wall)},
            {"age_seconds", std::chrono::duration<double>(now - row->last_seen).count()},
        });
    }

    nlohmann::json stats = nlohmann::json::object();
    for (auto r : {IngestResult::Accepted, IngestResult::Refreshed, IngestResult::Self,
                   IngestResult::BadSource, IngestResult::BadPayload, IngestResult::UrlMismatch,
                   IngestResult::RateLimited, IngestResult::TableFull}) {
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
    stats_.clear();
    evicted_ = 0;
    global_bucket_primed_ = false;
}

BeaconListener::BeaconListener() = default;

BeaconListener::~BeaconListener() {
    stop();
}

bool BeaconListener::is_running() const {
    return running_.load();
}

void BeaconListener::set_self_port(int port) {
    table_.set_self_port(port);
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
        sockets_status_ = nlohmann::json::array();
    }
    {
        std::lock_guard<std::mutex> cv_lock(cv_mtx_);
        stop_ = false;
    }
    running_ = true;
    worker_ = std::thread(&BeaconListener::thread_loop, this);
    LOG(INFO, "BeaconListener") << "Listening for LAN beacons on UDP " << kBeaconPort << std::endl;
#endif
}

void BeaconListener::stop() {
    std::lock_guard<std::mutex> lock(lifecycle_mtx_);
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
        error_.clear();
    }
    LOG(INFO, "BeaconListener") << "Stopped listening for LAN beacons" << std::endl;
}

nlohmann::json BeaconListener::status_json() const {
    if (!running_.load()) {
        return {{"enabled", false}};
    }
    auto now = std::chrono::steady_clock::now();
    nlohmann::json out = table_.to_json(now);
    std::lock_guard<std::mutex> status_lock(status_mtx_);
    bool listening = false;
    for (const auto& s : sockets_status_) {
        if (s.value("bound", false)) listening = true;
    }
    out["enabled"] = true;
    out["supported"] = supported_;
    out["listening"] = listening;
    out["error"] = error_.empty() ? nlohmann::json(nullptr) : nlohmann::json(error_);
    out["sockets"] = sockets_status_;
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
    std::lock_guard<std::mutex> status_lock(status_mtx_);
    sockets_status_ = std::move(arr);
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

    std::vector<uint32_t> local_addrs;
    std::set<uint32_t> desired;
    for (const auto& iface : ifaces) {
        in_addr ip{};
        if (inet_pton(AF_INET, iface.ipAddress.c_str(), &ip) != 1) continue;
        uint32_t ip_host = ntohl(ip.s_addr);
        local_addrs.push_back(ip_host);
        if ((ip_host & 0xFF000000u) == 0x7F000000u) continue;
        in_addr bcast{};
        if (inet_pton(AF_INET, iface.broadcastAddress.c_str(), &bcast) != 1) continue;
        desired.insert(ntohl(bcast.s_addr));
    }
    desired.insert(0xFFFFFFFFu);
    table_.set_local_addresses(std::move(local_addrs));

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

void BeaconListener::thread_loop() {
    auto next_refresh = std::chrono::steady_clock::now();
    char buf[kMaxDatagram];

    while (!stop_.load()) {
        auto now = std::chrono::steady_clock::now();
        if (now >= next_refresh) {
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
                socklen_t from_len = sizeof(from);
                int flags = MSG_DONTWAIT;
#ifdef __linux__
                flags |= MSG_TRUNC;
#endif
                ssize_t n = recvfrom(sock.fd, buf, sizeof(buf), flags,
                                     reinterpret_cast<sockaddr*>(&from), &from_len);
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
                IngestResult result =
                    table_.ingest(buf, len, truncated, src, std::chrono::steady_clock::now());
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
