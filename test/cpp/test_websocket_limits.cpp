// Connection, message-size and authentication-deadline limits of the
// WebSocket server, driven over real sockets against its own port.

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#ifndef _WIN32
#include <arpa/inet.h>
#include <csignal>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include <lemon/websocket_server.h>

#include "test_config_helpers.h"

using lemon::WebSocketServer;
using test_helpers::check;
using test_helpers::report_results;

#ifdef _WIN32

int main() {
    std::puts("WebSocket limits test is POSIX-only; skipped");
    return 0;
}

#else

namespace {

class Client {
public:
    Client(const std::string& ip, int port, const std::string& path) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) return;
        set_timeout(3);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        ::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
        if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) return;
        const std::string request = "GET " + path + " HTTP/1.1\r\nHost: " + ip +
                                    "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                                    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                                    "Sec-WebSocket-Version: 13\r\n\r\n";
        if (!send_all(request.data(), request.size())) return;
        std::string response;
        char c;
        while (response.find("\r\n\r\n") == std::string::npos && ::recv(fd_, &c, 1, 0) == 1) {
            response.push_back(c);
        }
        upgraded_ = response.rfind("HTTP/1.1 101", 0) == 0;
    }
    ~Client() {
        if (fd_ >= 0) ::close(fd_);
    }
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    bool upgraded() const { return upgraded_; }

    void set_timeout(int seconds) {
        timeval tv{seconds, 0};
        ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }

    // One masked client frame.
    bool send_frame(uint8_t opcode, bool fin, const std::string& payload) {
        std::string frame;
        frame.push_back(static_cast<char>((fin ? 0x80 : 0x00) | opcode));
        const uint64_t n = payload.size();
        if (n < 126) {
            frame.push_back(static_cast<char>(0x80 | n));
        } else if (n <= 0xffff) {
            frame.push_back(static_cast<char>(0x80 | 126));
            frame.push_back(static_cast<char>(n >> 8));
            frame.push_back(static_cast<char>(n & 0xff));
        } else {
            frame.push_back(static_cast<char>(0x80 | 127));
            for (int shift = 56; shift >= 0; shift -= 8) {
                frame.push_back(static_cast<char>((n >> shift) & 0xff));
            }
        }
        const char mask[4] = {0x11, 0x22, 0x33, 0x44};
        frame.append(mask, 4);
        for (size_t i = 0; i < payload.size(); ++i) {
            frame.push_back(static_cast<char>(payload[i] ^ mask[i % 4]));
        }
        return send_all(frame.data(), frame.size());
    }

    // True once the server has closed the connection: EOF, a reset, or a
    // close frame. False when nothing arrives before the receive timeout.
    bool closed_by_server() {
        char buf[4096];
        while (true) {
            const auto n = ::recv(fd_, buf, sizeof(buf), 0);
            if (n == 0) return true;
            if (n < 0) return errno != EAGAIN && errno != EWOULDBLOCK;
            if ((static_cast<uint8_t>(buf[0]) & 0x0f) == 0x8) return true;
        }
    }

    std::string read_text_frame() {
        uint8_t header[2];
        if (!recv_all(header, 2)) return "";
        uint64_t n = header[1] & 0x7f;
        if (n == 126 || n == 127) {
            uint8_t ext[8];
            const size_t ext_len = n == 126 ? 2 : 8;
            if (!recv_all(ext, ext_len)) return "";
            n = 0;
            for (size_t i = 0; i < ext_len; ++i) n = (n << 8) | ext[i];
        }
        std::string payload(n, '\0');
        if (n > 0 && !recv_all(&payload[0], n)) return "";
        return payload;
    }

private:
    bool send_all(const char* data, size_t len) {
        while (len > 0) {
            const auto n = ::send(fd_, data, len, 0);
            if (n <= 0) return false;
            data += n;
            len -= static_cast<size_t>(n);
        }
        return true;
    }
    bool recv_all(void* out, size_t len) {
        auto* p = static_cast<char*>(out);
        while (len > 0) {
            const auto n = ::recv(fd_, p, len, 0);
            if (n <= 0) return false;
            p += n;
            len -= static_cast<size_t>(n);
        }
        return true;
    }

    int fd_ = -1;
    bool upgraded_ = false;
};

bool eventually_upgrades(const std::string& ip, int port, const std::string& path) {
    for (int attempt = 0; attempt < 40; ++attempt) {
        Client c(ip, port, path);
        if (c.upgraded()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

std::string first_lan_ipv4() {
    ifaddrs* list = nullptr;
    if (::getifaddrs(&list) != 0) return "";
    std::string found;
    for (ifaddrs* it = list; it != nullptr && found.empty(); it = it->ifa_next) {
        if (it->ifa_addr == nullptr || it->ifa_addr->sa_family != AF_INET) continue;
        if ((it->ifa_flags & IFF_UP) == 0 || (it->ifa_flags & IFF_LOOPBACK) != 0) continue;
        char buf[INET_ADDRSTRLEN] = {0};
        ::inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(it->ifa_addr)->sin_addr, buf, sizeof(buf));
        found = buf;
    }
    ::freeifaddrs(list);
    return found;
}

void test_total_cap() {
    WebSocketServer server(nullptr, "127.0.0.1", 0);
    WebSocketServer::Limits limits;
    limits.max_connections = 2;
    server.set_limits(limits);
    check(server.start(), "total cap: server starts");
    const int port = server.get_port();
    {
        Client a("127.0.0.1", port, "/logs/stream");
        Client b("127.0.0.1", port, "/logs/stream");
        check(a.upgraded() && b.upgraded(), "total cap: two connections are admitted");
        {
            Client c("127.0.0.1", port, "/logs/stream");
            check(!c.upgraded(), "total cap: a third connection is refused");
        }
    }
    check(eventually_upgrades("127.0.0.1", port, "/logs/stream"),
          "total cap: closing connections frees their slots");
    server.stop();
}

void test_per_client_cap() {
    WebSocketServer server(nullptr, "0.0.0.0", 0);
    WebSocketServer::Limits limits;
    limits.max_connections_per_client = 1;
    server.set_limits(limits);
    check(server.start(), "per-client cap: server starts");
    const int port = server.get_port();
    {
        Client a("127.0.0.1", port, "/logs/stream");
        Client b("127.0.0.1", port, "/logs/stream");
        check(a.upgraded() && b.upgraded(), "per-client cap: loopback is exempt");
    }
    const std::string lan = first_lan_ipv4();
    if (lan.empty()) {
        std::puts("[SKIP] per-client cap: no non-loopback IPv4 address on this host");
    } else {
        Client a(lan, port, "/logs/stream");
        check(a.upgraded(), "per-client cap: a LAN client's first connection is admitted");
        {
            Client b(lan, port, "/logs/stream");
            check(!b.upgraded(), "per-client cap: the same LAN client's second connection is refused");
        }
    }
    server.stop();
}

void test_message_cap() {
    WebSocketServer server(nullptr, "127.0.0.1", 0);
    check(server.start(), "message cap: server starts");
    Client c("127.0.0.1", server.get_port(), "/logs/stream");
    check(c.upgraded(), "message cap: connection admitted");
    const std::string chunk(1024 * 1024, 'a');
    bool sent = c.send_frame(0x1, false, chunk);
    for (size_t total = chunk.size(); sent && total <= WebSocketServer::kMaxMessageBytes;
         total += chunk.size()) {
        sent = c.send_frame(0x0, false, chunk);
    }
    check(c.closed_by_server(), "message cap: an unfinished message past the cap closes the connection");
    server.stop();
}

void test_spans_auth_deadline() {
    ::setenv("LEMONADE_API_KEY", "test-key", 1);
    WebSocketServer server(nullptr, "127.0.0.1", 0);
    ::unsetenv("LEMONADE_API_KEY");
    WebSocketServer::Limits limits;
    limits.auth_timeout_seconds = 1;
    server.set_limits(limits);
    check(server.start(), "auth deadline: server starts");
    const int port = server.get_port();

    {
        Client c("127.0.0.1", port, "/spans/stream");
        c.set_timeout(5);
        check(c.upgraded(), "auth deadline: an uncredentialed spans upgrade is accepted");
        check(c.closed_by_server(), "auth deadline: it is closed when it never authenticates");
    }
    {
        Client c("127.0.0.1", port, "/spans/stream");
        check(c.upgraded() && c.send_frame(0x1, true, R"({"type":"auth","token":"test-key"})"),
              "auth deadline: a second spans connection sends its credential");
        check(c.read_text_frame().find("auth.ok") != std::string::npos,
              "auth deadline: the credential is accepted");
        std::this_thread::sleep_for(std::chrono::seconds(3));
        c.set_timeout(1);
        check(!c.closed_by_server(), "auth deadline: an authenticated connection stays open");
    }
    server.stop();
}

}  // namespace

int main() {
    std::signal(SIGPIPE, SIG_IGN);
    ::unsetenv("LEMONADE_API_KEY");
    ::unsetenv("LEMONADE_ADMIN_API_KEY");
    test_total_cap();
    test_per_client_cap();
    test_message_cap();
    test_spans_auth_deadline();
    return report_results("websocket limits");
}

#endif
