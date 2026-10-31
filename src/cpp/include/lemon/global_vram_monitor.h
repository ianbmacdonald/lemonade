#pragma once

#include <atomic>
#include <condition_variable>
#include <thread>
#include <functional>
#include <mutex>
#include <string>

namespace lemon {

// Callback function type. Hardware samples are 0.0 to 1.0; negative values are
// reserved for explicit idle-only evaluations through the internal test/admin hook.
using VramPressureCallback = std::function<void(double)>;

class GlobalVramMonitor {
public:
    GlobalVramMonitor();
    ~GlobalVramMonitor();

    // Start the monitor thread
    void start(int poll_interval_ms = 2000);

    // Stop the monitor thread
    void stop();

    // Register a callback to be notified of the current VRAM usage
    void set_pressure_callback(VramPressureCallback callback);

    // Test/admin hook: synchronously fire the pressure callback. A negative pct
    // requests an idle-only eviction-engine evaluation without pressure eviction.
    void simulate_pressure(double pct) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        if (pressure_callback_) {
            pressure_callback_(pct);
        }
    }

private:
    void monitor_loop();
    double poll_vram_usage() const;

    std::atomic<bool> running_;
    int poll_interval_ms_;
    std::thread monitor_thread_;
    // stop() wakes the loop, so a shutdown does not wait out the interval.
    std::mutex wake_mutex_;
    std::condition_variable wake_cv_;

    std::mutex callback_mutex_;
    VramPressureCallback pressure_callback_;
};

} // namespace lemon
