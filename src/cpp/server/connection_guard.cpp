#include "lemon/connection_guard.h"

#include <algorithm>
#include <cctype>
#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "lemon/utils/aixlog.hpp"

namespace lemon {

namespace {

thread_local ConnectionGuard::Entry* t_current_entry = nullptr;

#ifndef _WIN32
constexpr socket_t kInvalidSocket = -1;
#else
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#endif

socket_t duplicate_socket(socket_t sock) {
#ifdef _WIN32
    // Windows has no dup() for sockets; the deadline is not enforced there.
    (void)sock;
    return kInvalidSocket;
#else
    return ::dup(sock);
#endif
}

void close_watch_socket(socket_t sock) {
    if (sock == kInvalidSocket) {
        return;
    }
#ifdef _WIN32
    ::closesocket(sock);
#else
    ::close(sock);
#endif
}

void shutdown_watch_socket(socket_t sock) {
    if (sock == kInvalidSocket) {
        return;
    }
#ifdef _WIN32
    ::shutdown(sock, SD_BOTH);
#else
    ::shutdown(sock, SHUT_RDWR);
#endif
}

} // namespace

ConnectionGuard::Ticket::Ticket(ConnectionGuard* guard, std::shared_ptr<Entry> entry)
    : guard_(guard), entry_(std::move(entry)) {}

ConnectionGuard::Ticket::Ticket(Ticket&& other) noexcept
    : guard_(other.guard_), entry_(std::move(other.entry_)) {
    other.guard_ = nullptr;
}

ConnectionGuard::Ticket& ConnectionGuard::Ticket::operator=(Ticket&& other) noexcept {
    if (this != &other) {
        release();
        guard_ = other.guard_;
        entry_ = std::move(other.entry_);
        other.guard_ = nullptr;
    }
    return *this;
}

ConnectionGuard::Ticket::~Ticket() { release(); }

void ConnectionGuard::Ticket::release() {
    if (guard_ && entry_) {
        guard_->release(entry_);
    }
    guard_ = nullptr;
    entry_.reset();
}

ConnectionGuard::ConnectionGuard(Limits limits, bool start_watchdog) : limits_(limits) {
    if (start_watchdog && deadline_enabled()) {
        watchdog_ = std::thread([this] { watchdog_loop(); });
    }
}

ConnectionGuard::~ConnectionGuard() {
    {
        std::lock_guard<std::mutex> lock(stop_mu_);
        stop_ = true;
    }
    stop_cv_.notify_all();
    if (watchdog_.joinable()) {
        watchdog_.join();
    }
}

ConnectionGuard::Admission ConnectionGuard::admit(socket_t sock, const std::string& client,
                                                  Ticket& ticket, Clock::time_point now) {
    auto entry = std::make_shared<Entry>();
    entry->client = client;
    entry->accepted = now;
    entry->watch_sock = kInvalidSocket;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (limits_.max_connections_per_client > 0 && !is_loopback(client)) {
            auto it = per_client_.find(client);
            if (it != per_client_.end() && it->second >= limits_.max_connections_per_client) {
                return Admission::over_cap;
            }
        }
        // The watchdog shuts down a duplicate of the socket, never the original
        // descriptor: once the serving thread closes the original, its number
        // can be reused by another connection, but the duplicate still refers
        // to this connection's socket until the entry is released.
        if (deadline_enabled()) {
            entry->watch_sock = duplicate_socket(sock);
#ifndef _WIN32
            if (entry->watch_sock == kInvalidSocket) {
                return Admission::untrackable;
            }
#endif
        }
        ++per_client_[client];
        entries_.push_back(entry);
    }
    t_current_entry = entry.get();
    ticket = Ticket(this, std::move(entry));
    return Admission::admitted;
}

void ConnectionGuard::release(const std::shared_ptr<Entry>& entry) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = std::find(entries_.begin(), entries_.end(), entry);
        if (it != entries_.end()) {
            entries_.erase(it);
        }
        auto pc = per_client_.find(entry->client);
        if (pc != per_client_.end() && --pc->second == 0) {
            per_client_.erase(pc);
        }
    }
    if (t_current_entry == entry.get()) {
        t_current_entry = nullptr;
    }
    close_watch_socket(entry->watch_sock);
    entry->watch_sock = kInvalidSocket;
}

void ConnectionGuard::mark_request_received() {
    if (t_current_entry) {
        t_current_entry->receiving.store(false);
    }
}

std::vector<std::shared_ptr<ConnectionGuard::Entry>> ConnectionGuard::collect_expired(
    Clock::time_point now) {
    std::lock_guard<std::mutex> lock(mu_);
    return expire_locked(now, false);
}

std::vector<std::shared_ptr<ConnectionGuard::Entry>> ConnectionGuard::expire_locked(
    Clock::time_point now, bool shutdown_sockets) {
    std::vector<std::shared_ptr<Entry>> out;
    if (!deadline_enabled()) {
        return out;
    }
    for (const auto& e : entries_) {
        if (e->receiving.load() && !e->expired.load() &&
            now - e->accepted > limits_.receive_timeout) {
            e->expired.store(true);
            // Called under mu_, so release() cannot close watch_sock meanwhile.
            if (shutdown_sockets) {
                shutdown_watch_socket(e->watch_sock);
            }
            out.push_back(e);
        }
    }
    return out;
}

void ConnectionGuard::watchdog_loop() {
    std::unique_lock<std::mutex> stop_lock(stop_mu_);
    while (!stop_cv_.wait_for(stop_lock, std::chrono::milliseconds(500), [this] { return stop_; })) {
        std::vector<std::shared_ptr<Entry>> closed;
        {
            std::lock_guard<std::mutex> lock(mu_);
            closed = expire_locked(Clock::now(), true);
        }
        if (closed.empty()) {
            continue;
        }
        // One line per tick: a client that reconnects in a loop would
        // otherwise write a line for every connection it opens.
        std::map<std::string, size_t> by_client;
        for (const auto& e : closed) {
            ++by_client[e->client];
        }
        std::string clients;
        size_t listed = 0;
        for (const auto& [client, n] : by_client) {
            if (listed++ == 5) {
                clients += ", ...";
                break;
            }
            clients += (clients.empty() ? "" : ", ") + client + " x" + std::to_string(n);
        }
        LOG(WARNING, "Server") << "Closed " << closed.size()
                               << " connection(s) that did not send a complete request within "
                               << limits_.receive_timeout.count() << "s (" << clients << ")"
                               << std::endl;
    }
}

size_t ConnectionGuard::connections_for(const std::string& client) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = per_client_.find(client);
    return it == per_client_.end() ? 0 : it->second;
}

size_t ConnectionGuard::tracked() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
}

std::string ConnectionGuard::peer_address(socket_t sock) {
    sockaddr_storage addr;
    std::memset(&addr, 0, sizeof(addr));
    socklen_t len = sizeof(addr);
    if (::getpeername(sock, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
        return "";
    }
    char buf[INET6_ADDRSTRLEN] = {0};
    if (addr.ss_family == AF_INET) {
        auto* in = reinterpret_cast<sockaddr_in*>(&addr);
        if (!::inet_ntop(AF_INET, &in->sin_addr, buf, sizeof(buf))) {
            return "";
        }
    } else if (addr.ss_family == AF_INET6) {
        auto* in6 = reinterpret_cast<sockaddr_in6*>(&addr);
        if (!::inet_ntop(AF_INET6, &in6->sin6_addr, buf, sizeof(buf))) {
            return "";
        }
    } else {
        return "";
    }
    return normalize_address(buf);
}

std::string ConnectionGuard::normalize_address(const std::string& addr) {
    std::string out = addr;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const std::string mapped = "::ffff:";
    if (out.compare(0, mapped.size(), mapped) == 0 &&
        out.find('.', mapped.size()) != std::string::npos) {
        out = out.substr(mapped.size());
    }
    return out;
}

bool ConnectionGuard::is_loopback(const std::string& addr) {
    return addr == "::1" || addr.compare(0, 4, "127.") == 0;
}

} // namespace lemon
