#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <string>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace lemon {
namespace utils {

struct ProcessMemory {
    bool has_rss = false;
    bool has_anon = false;
    uint64_t rss_kib = 0;
    uint64_t anon_kib = 0;
};

namespace detail {

inline bool parse_status_kib(const std::string& line, const char* key, uint64_t& out) {
    const std::string prefix = key;
    if (line.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    size_t pos = prefix.size();
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) {
        ++pos;
    }
    if (pos >= line.size() || line[pos] < '0' || line[pos] > '9') {
        return false;
    }
    out = std::strtoull(line.c_str() + pos, nullptr, 10);
    return true;
}

}  // namespace detail

// VmRSS includes mmapped model weights and shared libraries; RssAnon is the
// private anonymous share. A zombie has no Vm*/Rss* lines, so both stay unset.
inline ProcessMemory parse_proc_status_memory(std::istream& in) {
    ProcessMemory mem;
    std::string line;
    while (std::getline(in, line)) {
        uint64_t value = 0;
        if (!mem.has_rss && detail::parse_status_kib(line, "VmRSS:", value)) {
            mem.has_rss = true;
            mem.rss_kib = value;
        } else if (!mem.has_anon && detail::parse_status_kib(line, "RssAnon:", value)) {
            mem.has_anon = true;
            mem.anon_kib = value;
        }
        if (mem.has_rss && mem.has_anon) {
            break;
        }
    }
    return mem;
}

inline ProcessMemory read_process_memory(int pid) {
    if (pid <= 0) {
        return {};
    }
#ifdef __linux__
    std::ifstream in("/proc/" + std::to_string(pid) + "/status");
    if (!in.is_open()) {
        return {};
    }
    return parse_proc_status_memory(in);
#else
    (void)pid;
    return {};
#endif
}

inline ProcessMemory read_self_memory() {
#ifdef __linux__
    std::ifstream in("/proc/self/status");
    if (!in.is_open()) {
        return {};
    }
    return parse_proc_status_memory(in);
#else
    return {};
#endif
}

inline double to_mib(uint64_t kib) {
    return std::round(static_cast<double>(kib) / 1024.0 * 10.0) / 10.0;
}

inline int current_process_id() {
#ifdef _WIN32
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

}  // namespace utils
}  // namespace lemon
