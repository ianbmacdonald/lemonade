#pragma once

#include <vector>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace lemon {

class Router;
class GlobalVramMonitor;
class WrappedServer;

class EvictionEngine {
public:
    EvictionEngine(Router* router, GlobalVramMonitor* vram_monitor);
    ~EvictionEngine();

    void start(int interval_ms = 5000);
    void stop();

    // Triggered by GlobalVramMonitor when pressure breaches threshold
    void on_vram_pressure(double pct);

private:
    void evaluation_loop();
    void evaluate_servers(double current_vram_pct);

    Router* router_;
    GlobalVramMonitor* vram_monitor_;

    std::atomic<bool> running_;
    int interval_ms_;
    std::thread engine_thread_;
    // stop() wakes the loop, so a shutdown does not wait out the interval.
    std::mutex wake_mutex_;
    std::condition_variable wake_cv_;
};

} // namespace lemon
