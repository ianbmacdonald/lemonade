#pragma once

#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

namespace lemon {

// Counts backend crashes per key. After `threshold` crashes inside `window`,
// the key is refused for `cooldown`, so an input that keeps crashing a backend
// cannot keep it in a crash-and-reload loop.
class CrashBreaker {
public:
    using Clock = std::chrono::steady_clock;

    CrashBreaker(int threshold, Clock::duration window, Clock::duration cooldown)
        : threshold_(threshold), window_(window), cooldown_(cooldown) {}

    // Whole seconds left in the cool-down (at least 1), or 0 when not blocked.
    long long blocked_seconds(const std::string& key, Clock::time_point now) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = state_.find(key);
        if (it == state_.end() || now >= it->second.blocked_until) return 0;
        const auto left = it->second.blocked_until - now;
        const long long secs =
            std::chrono::duration_cast<std::chrono::seconds>(left + std::chrono::seconds(1) -
                                                             Clock::duration(1))
                .count();
        return secs < 1 ? 1 : secs;
    }

    // Returns true when this crash opens the cool-down.
    bool record_crash(const std::string& key, Clock::time_point now) {
        std::lock_guard<std::mutex> lock(mutex_);
        State& s = state_[key];
        while (!s.crashes.empty() && now - s.crashes.front() >= window_) s.crashes.pop_front();
        s.crashes.push_back(now);
        if (static_cast<int>(s.crashes.size()) < threshold_) return false;
        s.crashes.clear();
        s.blocked_until = now + cooldown_;
        return true;
    }

private:
    struct State {
        std::deque<Clock::time_point> crashes;
        Clock::time_point blocked_until{};
    };

    const int threshold_;
    const Clock::duration window_;
    const Clock::duration cooldown_;
    std::mutex mutex_;
    std::unordered_map<std::string, State> state_;
};

}  // namespace lemon
