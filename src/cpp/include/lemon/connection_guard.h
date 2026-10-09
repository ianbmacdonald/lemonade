#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "lemon/utils/network_utils.h"

namespace lemon {

// Per-connection limits for the LAN-facing HTTP listener: a cap on concurrent
// connections per client address, and a deadline for receiving each request.
//
// cpp-httplib's read timeout is per read, so a client that sends one byte just
// inside it can hold a worker thread indefinitely. The deadline runs from when
// a worker thread picks the connection up (time queued for a worker is not
// counted) until the route handler starts, which is the point where the
// headers and body have both been read. The server limits each connection to
// one request while the deadline is enabled, so every request starts on a
// fresh connection. A watchdog thread shuts down connections that miss the
// deadline. A request whose body is read just before
// a watchdog tick but whose handler has not yet started can still be closed;
// the window is the few microseconds between the two inside httplib.
class ConnectionGuard {
public:
    using Clock = std::chrono::steady_clock;

    struct Limits {
        std::chrono::seconds receive_timeout{0};  // 0 = no deadline
        size_t max_connections_per_client = 0;     // 0 = no cap
    };

    struct Entry {
        socket_t watch_sock;  // a dup of the accepted socket; owned by the entry
        std::string client;
        Clock::time_point accepted;
        std::atomic<bool> receiving{true};
        std::atomic<bool> expired{false};
    };

    // Held by the thread serving a connection for its whole lifetime.
    class Ticket {
    public:
        Ticket() = default;
        Ticket(ConnectionGuard* guard, std::shared_ptr<Entry> entry);
        Ticket(Ticket&& other) noexcept;
        Ticket& operator=(Ticket&& other) noexcept;
        Ticket(const Ticket&) = delete;
        Ticket& operator=(const Ticket&) = delete;
        ~Ticket();

    private:
        void release();
        ConnectionGuard* guard_ = nullptr;
        std::shared_ptr<Entry> entry_;
    };

    explicit ConnectionGuard(Limits limits, bool start_watchdog = true);
    ~ConnectionGuard();
    ConnectionGuard(const ConnectionGuard&) = delete;
    ConnectionGuard& operator=(const ConnectionGuard&) = delete;

    const Limits& limits() const { return limits_; }
    bool deadline_enabled() const { return limits_.receive_timeout.count() > 0; }

    enum class Admission { admitted, over_cap, untrackable };

    // Admits a connection from `client` (normalized address). On `admitted`,
    // `ticket` tracks the connection until it is destroyed; otherwise the
    // caller refuses and closes the socket. `untrackable` means the deadline
    // is on but the socket could not be duplicated (descriptor exhaustion):
    // refusing fails closed, where admitting would leave it with no deadline.
    Admission admit(socket_t sock, const std::string& client, Ticket& ticket,
                    Clock::time_point now = Clock::now());

    // Called on the serving thread when the route handler starts.
    static void mark_request_received();

    // Entries whose deadline has passed at `now`, marked expired. The watchdog
    // shuts their sockets down; exposed so tests can drive time directly.
    std::vector<std::shared_ptr<Entry>> collect_expired(Clock::time_point now);

    size_t connections_for(const std::string& client) const;
    size_t tracked() const;

    // Peer address of an accepted socket as a string, with IPv4-mapped IPv6
    // addresses reduced to IPv4. Empty when the address cannot be read.
    static std::string peer_address(socket_t sock);
    static std::string normalize_address(const std::string& addr);
    static bool is_loopback(const std::string& addr);

private:
    void release(const std::shared_ptr<Entry>& entry);
    std::vector<std::shared_ptr<Entry>> expire_locked(Clock::time_point now, bool shutdown_sockets);
    void watchdog_loop();

    const Limits limits_;
    mutable std::mutex mu_;
    std::map<std::string, size_t> per_client_;
    std::vector<std::shared_ptr<Entry>> entries_;

    std::mutex stop_mu_;
    std::condition_variable stop_cv_;
    bool stop_ = false;
    std::thread watchdog_;
};

} // namespace lemon
