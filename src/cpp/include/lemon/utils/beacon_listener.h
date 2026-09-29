#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

namespace lemon {

constexpr int kBeaconPort = 13305;
constexpr std::chrono::seconds kPeerTtl{15};
constexpr std::chrono::seconds kEvictStale{8};
constexpr std::size_t kMaxHosts = 32;
constexpr double kGlobalRate = 500.0;
constexpr double kGlobalBurst = 1000.0;
constexpr double kPerSourceRate = 2.0;
constexpr double kPerSourceBurst = 4.0;
constexpr std::size_t kMaxSources = 64;
constexpr std::size_t kMaxDatagram = 1024;
constexpr std::chrono::seconds kIfaceRefresh{10};
constexpr std::chrono::milliseconds kPollSlice{500};

namespace utils {
bool is_rfc1918_ipv4(uint32_t host_order);
std::string ipv4_to_string(uint32_t host_order);
} // namespace utils

enum class IngestResult {
    Accepted,
    Refreshed,
    Self,
    BadSource,
    BadPayload,
    UrlMismatch,
    RateLimited,
    TableFull,
    WrongInterface
};

const char* ingest_result_name(IngestResult result);

// Whether a beacon that arrived on ifindex (0 = unknown) may be ingested.
bool beacon_arrival_accepted(bool allowlist_active, unsigned ifindex,
                             const std::vector<unsigned>& listened_ifindexes);

struct HeardHost {
    std::string hostname;
    std::string url;
    std::string source_ip;
    std::string instance_id;
    uint32_t source_ip_host_order = 0;
    std::chrono::system_clock::time_point first_seen_wall;
    std::chrono::system_clock::time_point last_seen_wall;
    std::chrono::steady_clock::time_point last_seen;
};

// Socket-free acceptance logic for received beacons. All time is injected so
// rate limiting and expiry are deterministic under test.
class BeaconPeerTable {
public:
    BeaconPeerTable();

    void set_self(const std::string& instance_id, int self_port);
    void set_self_port(int port);
    void set_local_addresses(std::vector<uint32_t> addrs);

    IngestResult ingest(const char* buf, std::size_t len, bool truncated, uint32_t src_ip,
                        std::chrono::steady_clock::time_point now);
    void expire(std::chrono::steady_clock::time_point now);
    IngestResult reject(IngestResult result);

    nlohmann::json to_json(std::chrono::steady_clock::time_point now) const;
    std::vector<HeardHost> hosts() const;
    std::size_t size() const;
    std::size_t tracked_source_count() const;
    uint64_t stat(IngestResult result) const;
    uint64_t evicted() const;
    void clear();

private:
    struct Bucket {
        double tokens = 0.0;
        std::chrono::steady_clock::time_point last;
    };
    struct SourceState {
        Bucket bucket;
        std::chrono::steady_clock::time_point last_seen;
    };

    IngestResult ingest_admitted_locked(const char* buf, std::size_t len, bool truncated, uint32_t src_ip,
                                        std::chrono::steady_clock::time_point now);
    SourceState& source_state_locked(uint32_t src_ip, std::chrono::steady_clock::time_point now);
    bool take_global_token_locked(std::chrono::steady_clock::time_point now);
    bool is_pinned_locked(uint32_t src_ip) const;
    void expire_locked(std::chrono::steady_clock::time_point now);
    IngestResult record_locked(IngestResult result);

    mutable std::mutex mutex_;
    std::string self_instance_id_;
    int self_port_ = 0;
    std::vector<uint32_t> local_addresses_;
    std::map<std::string, HeardHost> rows_;
    Bucket global_bucket_;
    bool global_bucket_primed_ = false;
    std::map<uint32_t, SourceState> sources_;
    std::map<IngestResult, uint64_t> stats_;
    uint64_t evicted_ = 0;
};

// Owns the UDP sockets and one worker thread. Binds only broadcast addresses
// (never INADDR_ANY): with SO_REUSEADDR the kernel hands each 127.0.0.1
// unicast beacon to the most recently bound wildcard socket only, so a
// wildcard bind here would take them from the CLI and desktop app listeners.
class BeaconListener {
public:
    BeaconListener();
    ~BeaconListener();

    BeaconListener(const BeaconListener&) = delete;
    BeaconListener& operator=(const BeaconListener&) = delete;

    void start(const std::string& instance_id, int self_port);
    void stop();
    void set_self_port(int port);
    // Interface names to listen on; empty means every RFC1918 interface.
    void set_interface_allowlist(std::vector<std::string> names);
    bool is_running() const;
    nlohmann::json status_json() const;

private:
    struct SocketState {
        std::string address;
        uint32_t bind_ip = 0;
        int fd = -1;
        bool bound = false;
        std::string error;
    };
    struct ListenedInterface {
        std::string name;
        unsigned index = 0;
        std::string address;
        std::string netmask;
        std::string broadcast;
    };

    void thread_loop();
    void refresh_sockets();
    void close_all_sockets();
    void publish_socket_status();
    bool arrived_on_listened_interface(const SocketState& sock, unsigned ifindex) const;
    void wait_slice();
    void log_rejection(IngestResult result, uint32_t src_ip);

    std::mutex lifecycle_mtx_;
    std::thread worker_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};
    std::condition_variable cv_;
    std::mutex cv_mtx_;

    mutable std::mutex status_mtx_;
    bool supported_ = true;
    std::string error_;
    bool start_failed_ = false;
    nlohmann::json sockets_status_ = nlohmann::json::array();
    nlohmann::json interfaces_status_ = nlohmann::json::array();

    mutable std::mutex allowlist_mtx_;
    std::vector<std::string> allowlist_;
    std::atomic<bool> refresh_requested_{false};

    std::vector<SocketState> sockets_;
    std::vector<ListenedInterface> interfaces_;
    bool allowlist_active_ = false;
    std::map<IngestResult, std::chrono::steady_clock::time_point> last_warning_;
    BeaconPeerTable table_;
};

} // namespace lemon
