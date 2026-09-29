// Standalone test for lemon::utils process memory helpers.
// Build with: cmake --build --preset default --target test_process_memory
// Run with: ctest --test-dir build -R '^ProcessMemoryTest$' --output-on-failure

#include "lemon/utils/process_memory.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using lemon::utils::ProcessMemory;
using lemon::utils::current_process_id;
using lemon::utils::parse_proc_status_memory;
using lemon::utils::read_process_memory;
using lemon::utils::read_self_memory;
using lemon::utils::to_mib;

static int g_failures = 0;

static void check(bool ok, const char* name) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) {
        ++g_failures;
    }
}

static ProcessMemory parse(const std::string& text) {
    std::istringstream in(text);
    return parse_proc_status_memory(in);
}

static void test_parser() {
    const std::string kernel6 =
        "Name:\tllama-server\n"
        "State:\tS (sleeping)\n"
        "Pid:\t1234\n"
        "VmPeak:\t 1234567 kB\n"
        "VmSize:\t 1200000 kB\n"
        "VmHWM:\t  700000 kB\n"
        "VmRSS:\t  627088 kB\n"
        "RssAnon:\t   90214 kB\n"
        "RssFile:\t  536874 kB\n"
        "RssShmem:\t       0 kB\n"
        "Threads:\t8\n";
    ProcessMemory m = parse(kernel6);
    check(m.has_rss && m.rss_kib == 627088, "kernel 6.x fixture: VmRSS");
    check(m.has_anon && m.anon_kib == 90214, "kernel 6.x fixture: RssAnon");

    m = parse("Name:\told\nVmRSS:\t  4096 kB\nThreads:\t1\n");
    check(m.has_rss && m.rss_kib == 4096 && !m.has_anon, "pre-4.5 fixture: rss only");

    m = parse("Name:\tzombie\nState:\tZ (zombie)\nPid:\t42\nThreads:\t1\n");
    check(!m.has_rss && !m.has_anon, "zombie fixture: neither");

    m = parse("");
    check(!m.has_rss && !m.has_anon, "empty stream: neither");

    m = parse("VmRSS:");
    check(!m.has_rss && !m.has_anon, "truncated VmRSS with no value: neither");

    m = parse("VmRSS:   \nRssAnon:\tkB\n");
    check(!m.has_rss && !m.has_anon, "keys with no digits: neither");

    m = parse("VmRSS:      512 kB\nRssAnon:\t\t256 kB\n");
    check(m.has_rss && m.rss_kib == 512, "multi-space padding parses");
    check(m.has_anon && m.anon_kib == 256, "multi-tab padding parses");

    m = parse("VmRSSx:\t100 kB\nRssAnonX:\t200 kB\n");
    check(!m.has_rss && !m.has_anon, "prefix match requires the colon");
}

static void test_to_mib() {
    check(to_mib(0) == 0.0, "to_mib(0) == 0.0");
    check(to_mib(1024) == 1.0, "to_mib(1024) == 1.0");
    check(to_mib(1075) == 1.0, "to_mib(1075) == 1.0");
    check(to_mib(1076) == 1.1, "to_mib(1076) == 1.1");
    check(to_mib(627088) == 612.4, "to_mib(627088) == 612.4");
}

#ifdef __linux__
static int nonexistent_pid() {
    std::ifstream in("/proc/sys/kernel/pid_max");
    long long pid_max = 0;
    if (in >> pid_max && pid_max > 0 && pid_max < INT_MAX) {
        return static_cast<int>(pid_max + 1);
    }
    return INT_MAX;
}

static void test_live_reads() {
    ProcessMemory self = read_self_memory();
    check(self.has_rss && self.rss_kib > 0, "read_self_memory: rss > 0");
    check(self.has_anon, "read_self_memory: has anon");

    ProcessMemory by_pid = read_process_memory(current_process_id());
    check(by_pid.has_rss && by_pid.rss_kib > 0, "read_process_memory(self pid): rss");

    ProcessMemory zero = read_process_memory(0);
    check(!zero.has_rss && !zero.has_anon, "read_process_memory(0): empty");

    ProcessMemory negative = read_process_memory(-1);
    check(!negative.has_rss && !negative.has_anon, "read_process_memory(-1): empty");

    ProcessMemory gone = read_process_memory(nonexistent_pid());
    check(!gone.has_rss && !gone.has_anon, "read_process_memory(pid_max+1): empty");
}
#endif

int main() {
    test_parser();
    test_to_mib();
    check(current_process_id() > 0, "current_process_id() > 0");
#ifdef __linux__
    test_live_reads();
#endif
    if (g_failures > 0) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All process memory tests passed\n");
    return 0;
}
