#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#ifndef _WIN32
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include <lemon/connection_guard.h>

#include "test_config_helpers.h"

using lemon::ConnectionGuard;
using test_helpers::check;
using test_helpers::report_results;
using namespace std::chrono_literals;

static void test_addresses() {
    check(ConnectionGuard::normalize_address("::FFFF:192.168.1.5") == "192.168.1.5",
          "IPv4-mapped IPv6 reduces to IPv4");
    check(ConnectionGuard::normalize_address("FE80::1") == "fe80::1", "IPv6 is lowercased");
    check(ConnectionGuard::normalize_address("::ffff:abcd") == "::ffff:abcd",
          "a non-dotted ::ffff: address is left alone");
    check(ConnectionGuard::is_loopback("127.0.0.1"), "127.0.0.1 is loopback");
    check(ConnectionGuard::is_loopback("127.4.5.6"), "127/8 is loopback");
    check(ConnectionGuard::is_loopback("::1"), "::1 is loopback");
    check(!ConnectionGuard::is_loopback("192.168.1.5"), "a LAN address is not loopback");
    check(!ConnectionGuard::is_loopback(""), "an unreadable address is not loopback");
}

static void test_cap() {
    ConnectionGuard::Limits limits;
    limits.max_connections_per_client = 2;
    ConnectionGuard guard(limits, false);
    check(!guard.deadline_enabled(), "cap-only guard has no deadline");

    ConnectionGuard::Ticket a1, a2, a3, b1;
    check(guard.admit(-1, "192.168.1.5", a1), "first connection admitted");
    check(guard.admit(-1, "192.168.1.5", a2), "second connection admitted");
    check(!guard.admit(-1, "192.168.1.5", a3), "third connection from the same client refused");
    check(guard.admit(-1, "192.168.1.6", b1), "another client is not affected");
    check(guard.connections_for("192.168.1.5") == 2, "refused connection is not counted");

    {
        ConnectionGuard::Ticket l1, l2, l3;
        check(guard.admit(-1, "127.0.0.1", l1) && guard.admit(-1, "127.0.0.1", l2) &&
                  guard.admit(-1, "127.0.0.1", l3),
              "loopback is exempt from the cap");
    }
    check(guard.connections_for("127.0.0.1") == 0, "loopback tickets released on scope exit");

    a1 = ConnectionGuard::Ticket();
    check(guard.connections_for("192.168.1.5") == 1, "releasing a ticket frees a slot");
    check(guard.admit(-1, "192.168.1.5", a3), "a freed slot can be reused");
    a2 = ConnectionGuard::Ticket();
    a3 = ConnectionGuard::Ticket();
    b1 = ConnectionGuard::Ticket();
    check(guard.tracked() == 0, "no entries left after all tickets are released");
}

#ifndef _WIN32
static void test_deadline_logic() {
    ConnectionGuard::Limits limits;
    limits.receive_timeout = 30s;
    ConnectionGuard guard(limits, false);
    check(guard.deadline_enabled(), "deadline enabled with a non-zero timeout");

    int fds[2];
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair");
    const auto t0 = ConnectionGuard::Clock::now();
    ConnectionGuard::Ticket slow, fast;
    check(guard.admit(fds[0], "192.168.1.5", slow, t0), "slow connection admitted");
    check(guard.admit(fds[0], "192.168.1.7", fast, t0), "fast connection admitted");
    // admit() binds the serving thread to the most recent entry, as the front
    // server does for the connection it is about to serve.
    ConnectionGuard::mark_request_received();

    check(guard.collect_expired(t0 + 29s).empty(), "nothing expires before the deadline");
    auto expired = guard.collect_expired(t0 + 31s);
    check(expired.size() == 1 && expired[0]->client == "192.168.1.5",
          "only the connection still receiving expires");
    check(guard.collect_expired(t0 + 60s).empty(), "an expired connection is reported once");

    slow = ConnectionGuard::Ticket();
    fast = ConnectionGuard::Ticket();
    check(guard.tracked() == 0, "entries released");
    ::close(fds[0]);
    ::close(fds[1]);
}

static void test_watchdog_shuts_down_socket() {
    ConnectionGuard::Limits limits;
    limits.receive_timeout = 1s;
    ConnectionGuard guard(limits);

    int fds[2];
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair");
    // Without a shutdown the recv below would block forever; fail instead.
    timeval tv{5, 0};
    ::setsockopt(fds[1], SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    {
        ConnectionGuard::Ticket t;
        check(guard.admit(fds[0], "192.168.1.5", t), "connection admitted");
        // The original descriptor stays open the whole time: the watchdog acts
        // through its own duplicate.
        char buf[1];
        const auto start = std::chrono::steady_clock::now();
        const auto n = ::recv(fds[1], buf, sizeof(buf), 0);
        const auto waited = std::chrono::steady_clock::now() - start;
        check(n == 0, "peer sees end of stream after the deadline");
        check(waited >= 900ms && waited < 5s, "shutdown happens at about the deadline");
    }
    check(guard.tracked() == 0, "entry released after the ticket goes away");
    ::close(fds[0]);
    ::close(fds[1]);
}

static void test_received_request_is_not_shut_down() {
    ConnectionGuard::Limits limits;
    limits.receive_timeout = 1s;
    ConnectionGuard guard(limits);

    int fds[2];
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair");
    ConnectionGuard::Ticket t;
    check(guard.admit(fds[0], "192.168.1.5", t), "connection admitted");
    ConnectionGuard::mark_request_received();
    std::this_thread::sleep_for(2500ms);
    const char ping = 'p';
    check(::send(fds[0], &ping, 1, 0) == 1, "socket still writable after the deadline");
    char buf[1];
    check(::recv(fds[1], buf, 1, 0) == 1 && buf[0] == 'p', "peer still receives data");
    t = ConnectionGuard::Ticket();
    ::close(fds[0]);
    ::close(fds[1]);
}
#endif

int main() {
    std::puts("=== RUNNING CONNECTION GUARD TESTS ===");
    test_addresses();
    test_cap();
#ifndef _WIN32
    test_deadline_logic();
    test_watchdog_shuts_down_socket();
    test_received_request_is_not_shut_down();
#endif
    return report_results("connection guard");
}
