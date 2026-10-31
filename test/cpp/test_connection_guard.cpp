#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include <lemon/connection_guard.h>
#include <lemon/upgradable_http_server.h>

#include "test_config_helpers.h"

using lemon::ConnectionGuard;
using test_helpers::check;
using test_helpers::report_results;
using namespace std::chrono_literals;

static bool admitted(ConnectionGuard::Admission a) {
    return a == ConnectionGuard::Admission::admitted;
}

static void test_addresses() {
    check(ConnectionGuard::normalize_address("::FFFF:192.0.2.5") == "192.0.2.5",
          "IPv4-mapped IPv6 reduces to IPv4");
    check(ConnectionGuard::normalize_address("FE80::1") == "fe80::1", "IPv6 is lowercased");
    check(ConnectionGuard::normalize_address("::ffff:abcd") == "::ffff:abcd",
          "a non-dotted ::ffff: address is left alone");
    check(ConnectionGuard::is_loopback("127.0.0.1"), "127.0.0.1 is loopback");
    check(ConnectionGuard::is_loopback("127.4.5.6"), "127/8 is loopback");
    check(ConnectionGuard::is_loopback("::1"), "::1 is loopback");
    check(!ConnectionGuard::is_loopback("192.0.2.5"), "a LAN address is not loopback");
    check(!ConnectionGuard::is_loopback(""), "an unreadable address is not loopback");
}

static void test_bounded_task_queue() {
    std::unique_ptr<httplib::TaskQueue> pool(lemon::detail::new_bounded_thread_pool(1, 2));
    std::mutex mu;
    std::condition_variable cv;
    bool release = false;
    std::atomic<bool> started{false};
    check(pool->enqueue([&] {
        started = true;
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return release; });
    }), "a connection goes to the idle worker");
    while (!started) {
        std::this_thread::sleep_for(1ms);
    }
    check(pool->enqueue([] {}) && pool->enqueue([] {}), "connections queue up to the bound");
    check(!pool->enqueue([] {}), "a connection past the queue bound is refused");
    {
        std::lock_guard<std::mutex> lock(mu);
        release = true;
    }
    cv.notify_all();
    pool->shutdown();
}

static void test_cap() {
    ConnectionGuard::Limits limits;
    limits.max_connections_per_client = 2;
    ConnectionGuard guard(limits, false);
    check(!guard.deadline_enabled(), "cap-only guard has no deadline");

    ConnectionGuard::Ticket a1, a2, a3, b1;
    check(admitted(guard.admit(-1, "192.0.2.5", a1)), "first connection admitted");
    check(admitted(guard.admit(-1, "192.0.2.5", a2)), "second connection admitted");
    check(!admitted(guard.admit(-1, "192.0.2.5", a3)), "third connection from the same client refused");
    check(admitted(guard.admit(-1, "192.0.2.6", b1)), "another client is not affected");
    check(guard.connections_for("192.0.2.5") == 2, "refused connection is not counted");

    {
        ConnectionGuard::Ticket l1, l2, l3;
        check(admitted(guard.admit(-1, "127.0.0.1", l1)) &&
                  admitted(guard.admit(-1, "127.0.0.1", l2)) &&
                  admitted(guard.admit(-1, "127.0.0.1", l3)),
              "loopback is exempt from the cap");
    }
    check(guard.connections_for("127.0.0.1") == 0, "loopback tickets released on scope exit");

    a1 = ConnectionGuard::Ticket();
    check(guard.connections_for("192.0.2.5") == 1, "releasing a ticket frees a slot");
    check(admitted(guard.admit(-1, "192.0.2.5", a3)), "a freed slot can be reused");
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
    check(admitted(guard.admit(fds[0], "192.0.2.5", slow, t0)), "slow connection admitted");
    check(admitted(guard.admit(fds[0], "192.0.2.7", fast, t0)), "fast connection admitted");
    // admit() binds the serving thread to the most recent entry, as the front
    // server does for the connection it is about to serve.
    ConnectionGuard::mark_request_received();

    check(guard.collect_expired(t0 + 29s).empty(), "nothing expires before the deadline");
    auto expired = guard.collect_expired(t0 + 31s);
    check(expired.size() == 1 && expired[0]->client == "192.0.2.5",
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
        check(admitted(guard.admit(fds[0], "192.0.2.5", t)), "connection admitted");
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
    check(admitted(guard.admit(fds[0], "192.0.2.5", t)), "connection admitted");
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

static void test_untrackable_fails_closed() {
    ConnectionGuard::Limits limits;
    limits.receive_timeout = 30s;
    ConnectionGuard guard(limits, false);

    int fds[2];
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair");
    rlimit saved{};
    ::getrlimit(RLIMIT_NOFILE, &saved);
    rlimit low = saved;
    low.rlim_cur = 64;
    ::setrlimit(RLIMIT_NOFILE, &low);
    std::vector<int> filler;
    for (int fd; (fd = ::open("/dev/null", O_RDONLY)) >= 0;) {
        filler.push_back(fd);
    }
    ConnectionGuard::Ticket t;
    const auto a = guard.admit(fds[0], "192.0.2.5", t);
    for (int fd : filler) {
        ::close(fd);
    }
    ::setrlimit(RLIMIT_NOFILE, &saved);
    check(a == ConnectionGuard::Admission::untrackable,
          "a socket that cannot be duplicated is refused, not admitted without a deadline");
    check(guard.tracked() == 0 && guard.connections_for("192.0.2.5") == 0,
          "a refused connection leaves nothing tracked");
    ::close(fds[0]);
    ::close(fds[1]);
}

#ifdef __linux__
static int inheritable_descriptors() {
    int n = 0;
    for (int fd = 0; fd < 1024; ++fd) {
        const int flags = ::fcntl(fd, F_GETFD);
        if (flags >= 0 && !(flags & FD_CLOEXEC)) {
            ++n;
        }
    }
    return n;
}

static void test_watch_socket_is_close_on_exec() {
    ConnectionGuard::Limits limits;
    limits.receive_timeout = 30s;
    ConnectionGuard guard(limits);
    int fds[2];
    check(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) == 0, "socketpair");
    const int before = inheritable_descriptors();
    {
        ConnectionGuard::Ticket t;
        check(admitted(guard.admit(fds[0], "192.0.2.5", t)), "connection admitted");
        check(inheritable_descriptors() == before,
              "the watchdog's duplicate is not inherited by child processes");
    }
    ::close(fds[0]);
    ::close(fds[1]);
}
#endif

// A peer that connects and sends nothing, or stops mid-headers, must not hold
// the worker: the peek gives up and leaves the socket to httplib's timeouts.
static void check_peek_returns(const char* sent, const char* label) {
    int fds[2];
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair");
    if (*sent) {
        check(::send(fds[1], sent, std::strlen(sent), 0) > 0, "send partial request");
    }
    auto done = std::make_shared<std::atomic<bool>>(false);
    std::thread peek([fd = fds[0], done] {
        lemon::detail::peek_websocket_upgrade(fd);
        *done = true;
    });
    const auto t0 = std::chrono::steady_clock::now();
    while (!*done && std::chrono::steady_clock::now() - t0 < 3s) {
        std::this_thread::sleep_for(10ms);
    }
    check(*done, label);
    ::shutdown(fds[0], SHUT_RDWR);
    peek.join();
    ::close(fds[0]);
    ::close(fds[1]);
}

static void test_peek_is_bounded() {
    check_peek_returns("", "the upgrade peek returns for a silent peer");
    check_peek_returns("GET /realtime HTTP/1.1\r\nHost: x\r\n",
                       "the upgrade peek returns for headers that never finish");
}
#endif

int main() {
    std::puts("=== RUNNING CONNECTION GUARD TESTS ===");
    test_addresses();
    test_cap();
    test_bounded_task_queue();
#ifndef _WIN32
    test_deadline_logic();
    test_watchdog_shuts_down_socket();
    test_received_request_is_not_shut_down();
    test_untrackable_fails_closed();
    test_peek_is_bounded();
#ifdef __linux__
    test_watch_socket_is_close_on_exec();
#endif
#endif
    return report_results("connection guard");
}
