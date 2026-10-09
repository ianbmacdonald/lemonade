#pragma once

// Main-port WebSocket upgrade support.
//
// httplib's accept loop dispatches every accepted connection to the virtual
// Server::process_and_close_socket — the only point where a WebSocket upgrade
// can be intercepted before the request is consumed. That member is private,
// so a subclass cannot delegate back to the base implementation directly.
//
// Instead the roles are split across two objects:
//   - RoutedHttpServer: the normal server carrying all routes/handlers. It
//     never listens; it only processes connections.
//   - UpgradableFrontServer: binds and accepts on the main port. WebSocket
//     upgrades for /realtime and /logs/stream are handed to the libwebsockets
//     server (which adopts the raw socket); everything else is dispatched to
//     the routed server. Because the routed server's dynamic type does not
//     override process_and_close_socket, calling it through a member-function
//     pointer virtual-dispatches to httplib's own implementation.
//
// The member-function pointer is extracted with the standard explicit-
// instantiation idiom ([temp.explicit]p12: access rules are not enforced for
// explicit template instantiation arguments). This relies only on the member
// declaration, which is identical in the bundled header-only httplib and in
// distro split builds (e.g. the Debian/Ubuntu libcpp-httplib package), so the
// feature works with both.

#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "lemon/connection_guard.h"
#include "lemon/utils/aixlog.hpp"
#include "lemon/utils/network_utils.h"

namespace lemon {

bool is_websocket_endpoint(const std::string& path);

namespace detail {

using ProcessAndCloseSocketFn = bool (httplib::Server::*)(socket_t);

template <typename Tag>
struct PrivateMemberHolder {
    static typename Tag::type value;
};
template <typename Tag>
typename Tag::type PrivateMemberHolder<Tag>::value;

template <typename Tag, typename Tag::type Member>
struct PrivateMemberInit {
    struct Setter {
        Setter() { PrivateMemberHolder<Tag>::value = Member; }
    };
    static Setter setter;
};
template <typename Tag, typename Tag::type Member>
typename PrivateMemberInit<Tag, Member>::Setter PrivateMemberInit<Tag, Member>::setter;

struct ProcessAndCloseSocketTag {
    using type = ProcessAndCloseSocketFn;
};

// httplib's route tables (private). Spelled out because the Handlers alias is
// private too. Only the non-content-reader tables: lemond registers no
// content-reader routes.
using RouteHandlerList =
    std::vector<std::pair<std::unique_ptr<httplib::detail::MatcherBase>, httplib::Server::Handler>>;
using RouteHandlerListMember = RouteHandlerList httplib::Server::*;
struct GetHandlersTag { using type = RouteHandlerListMember; };
struct PostHandlersTag { using type = RouteHandlerListMember; };
struct PutHandlersTag { using type = RouteHandlerListMember; };
struct PatchHandlersTag { using type = RouteHandlerListMember; };
struct DeleteHandlersTag { using type = RouteHandlerListMember; };
struct OptionsHandlersTag { using type = RouteHandlerListMember; };

// Runs `on_enter` at the start of every route handler registered so far.
// httplib calls a handler only after it has read the request body, so this is
// the first point where a request is known to be fully received.
inline void wrap_route_handlers(httplib::Server& srv, const std::function<void()>& on_enter) {
    for (RouteHandlerListMember member : {PrivateMemberHolder<GetHandlersTag>::value,
                                          PrivateMemberHolder<PostHandlersTag>::value,
                                          PrivateMemberHolder<PutHandlersTag>::value,
                                          PrivateMemberHolder<PatchHandlersTag>::value,
                                          PrivateMemberHolder<DeleteHandlersTag>::value,
                                          PrivateMemberHolder<OptionsHandlersTag>::value}) {
        for (auto& route : srv.*member) {
            route.second = [on_enter, inner = std::move(route.second)](
                               const httplib::Request& req, httplib::Response& res) {
                on_enter();
                inner(req, res);
            };
        }
    }
}

// Peek the first bytes of an accepted connection (without consuming them) and
// decide whether it is a WebSocket upgrade for one of our realtime paths.
inline bool peek_websocket_upgrade(socket_t sock) {
    char buf[4096];
    for (int attempt = 0; attempt < 50; ++attempt) {  // up to ~500 ms for headers
#ifdef _WIN32
        int n = ::recv(sock, buf, static_cast<int>(sizeof(buf)), MSG_PEEK);
#else
        auto n = ::recv(sock, buf, sizeof(buf), MSG_PEEK);
#endif
        if (n <= 0) {
            return false;
        }

        std::string data(buf, static_cast<size_t>(n));
        if (data.size() >= 4 && data.compare(0, 4, "GET ") != 0) {
            return false;  // upgrades are always GET
        }
        if (data.find("\r\n\r\n") == std::string::npos) {
            if (static_cast<size_t>(n) == sizeof(buf)) {
                return false;  // header block too large — let httplib handle it
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;  // headers not fully arrived yet
        }

        // Path filter: only the WebSocket endpoints are handed over
        size_t sp1 = data.find(' ');
        size_t sp2 = data.find(' ', sp1 + 1);
        if (sp1 == std::string::npos || sp2 == std::string::npos) {
            return false;
        }
        std::string path = data.substr(sp1 + 1, sp2 - sp1 - 1);
        size_t query = path.find('?');
        if (query != std::string::npos) {
            path = path.substr(0, query);
        }
        if (!lemon::is_websocket_endpoint(path)) {
            return false;
        }

        std::string lower = data;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        size_t upgrade_pos = lower.find("\r\nupgrade:");
        if (upgrade_pos == std::string::npos) {
            return false;
        }
        size_t line_end = lower.find("\r\n", upgrade_pos + 2);
        return lower.substr(upgrade_pos, line_end - upgrade_pos).find("websocket") != std::string::npos;
    }
    return false;
}

} // namespace detail

// The normal server carrying all routes. It never listens; the front server
// below feeds it accepted connections. httplib's per-connection keep-alive
// loop only runs while svr_sock_ is a valid socket, so the front's listen
// socket is injected after bind and cleared on stop.
class RoutedHttpServer : public httplib::Server {
public:
    void set_listen_socket(socket_t sock) { svr_sock_ = sock; }
};

// Listener for the main port. See file comment.
class UpgradableFrontServer : public httplib::Server {
public:
    using UpgradeHandler = std::function<bool(socket_t)>;

    UpgradableFrontServer(RoutedHttpServer* delegate, UpgradeHandler handler)
        : delegate_(delegate), upgrade_handler_(std::move(handler)) {}

    socket_t listen_socket() const { return svr_sock_; }

    void set_connection_guard(ConnectionGuard* guard) { guard_ = guard; }

private:
    bool process_and_close_socket(socket_t sock) override {
        ConnectionGuard::Ticket ticket;
        if (guard_) {
            const std::string client = ConnectionGuard::peer_address(sock);
            if (!guard_->admit(sock, client, ticket)) {
                refuse_over_cap(sock, client);
                return true;
            }
        }
        if (!lemon::utils::configure_tcp_keepalive(sock)) {
            LOG(DEBUG, "Server") << "Failed to configure TCP keep-alive on accepted socket" << std::endl;
        }
        if (upgrade_handler_ && detail::peek_websocket_upgrade(sock)) {
            if (upgrade_handler_(sock)) {
                return true;  // ownership transferred to the WebSocket server
            }
        }
        auto fn = detail::PrivateMemberHolder<detail::ProcessAndCloseSocketTag>::value;
        return (delegate_->*fn)(sock);
    }

    void refuse_over_cap(socket_t sock, const std::string& client) {
        static const std::string kBody =
            "{\"error\": \"Too many connections from this client; retry shortly\"}";
        static const std::string kBusy =
            "HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nRetry-After: 1\r\n"
            "Content-Type: application/json\r\nContent-Length: " +
            std::to_string(kBody.size()) + "\r\n\r\n" + kBody;
        httplib::detail::send_socket(sock, kBusy.data(), kBusy.size(), 0);
        httplib::detail::shutdown_socket(sock);
        httplib::detail::close_socket(sock);
        // A client at its cap is usually retrying in a loop; log the first
        // refusal and then every 100th rather than every one.
        const auto n = refused_.fetch_add(1) + 1;
        if (n == 1 || n % 100 == 0) {
            LOG(WARNING, "Server") << "Refused connection from " << client << ": "
                                   << guard_->limits().max_connections_per_client
                                   << " connections already open (" << n << " refused so far)"
                                   << std::endl;
        }
    }

    RoutedHttpServer* delegate_;
    UpgradeHandler upgrade_handler_;
    ConnectionGuard* guard_ = nullptr;
    std::atomic<uint64_t> refused_{0};
};

} // namespace lemon
