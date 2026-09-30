#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <lemon/cli_parser.h>
#include <lemon/config_file.h>
#include <lemon/runtime_config.h>
#include <lemon/utils/beacon_listener.h>

#include "test_config_helpers.h"

namespace fs = std::filesystem;
using json = nlohmann::json;
using lemon::ConfigFile;
using lemon::RuntimeConfig;
using test_helpers::check;

int main() {
    std::puts("=== RUNNING RUNTIME CONFIG DISCOVERY TESTS ===");

    json base_cfg = {
        {"config_version", 2},
        {"port", 13305},
        {"host", "localhost"},
        {"broadcast", true}
    };

    RuntimeConfig config(base_cfg);

    // 1. Test broadcast initial value and updates
    check(config.broadcast() == true, "initial broadcast is true");
    config.set({{"broadcast", false}});
    check(config.broadcast() == false, "broadcast updated to false");
    config.set({{"broadcast", true}});
    check(config.broadcast() == true, "broadcast updated to true");

    // 2. Test legacy no_broadcast alias mapping via set()
    config.set({{"no_broadcast", true}});
    check(config.broadcast() == false, "legacy no_broadcast: true sets broadcast to false");
    check(config.snapshot()["broadcast"] == false, "snapshot contains broadcast: false");
    check(!config.snapshot().contains("no_broadcast"), "snapshot does not contain legacy no_broadcast");

    config.set({{"no_broadcast", false}});
    check(config.broadcast() == true, "legacy no_broadcast: false sets broadcast to true");
    check(config.snapshot()["broadcast"] == true, "snapshot contains broadcast: true");
    check(!config.snapshot().contains("no_broadcast"), "snapshot does not contain legacy no_broadcast");

    // 3. Validation: rejects non-boolean values and conflicting keys
    bool threw_invalid_broadcast = false;
    try {
        config.set({{"broadcast", "not-a-bool"}});
    } catch (const std::invalid_argument& e) {
        threw_invalid_broadcast = true;
    }
    check(threw_invalid_broadcast, "rejects non-boolean broadcast");

    bool threw_invalid_no_broadcast = false;
    try {
        config.set({{"no_broadcast", 123}});
    } catch (const std::invalid_argument& e) {
        threw_invalid_no_broadcast = true;
    }
    check(threw_invalid_no_broadcast, "rejects non-boolean no_broadcast");

    bool threw_conflicting = false;
    try {
        config.set({{"broadcast", true}, {"no_broadcast", false}});
    } catch (const std::invalid_argument& e) {
        threw_conflicting = true;
    }
    check(threw_conflicting, "rejects conflicting broadcast and no_broadcast in set()");

    // 4. Test transient override does NOT leak into snapshot()
    config.set({{"broadcast", true}});
    config.set_broadcast_override(false);
    check(config.broadcast() == false, "broadcast_override(false) takes effect at runtime");
    check(config.snapshot()["broadcast"] == true, "broadcast_override does NOT mutate snapshot()");

    // 5. Override -> config set transition and side-effect callback firing
    // Case A: Persisted broadcast=false, startup override=true, user sets broadcast=false
    config.set({{"broadcast", false}});
    config.set_broadcast_override(true);
    check(config.broadcast() == true, "effective broadcast is true due to override");

    bool side_effect_fired_a = false;
    json side_effect_diff_a = json::object();
    config.set({{"broadcast", false}}, [&](const json& diff) {
        side_effect_fired_a = true;
        side_effect_diff_a = diff;
    });
    check(config.broadcast() == false, "broadcast becomes false after set()");
    check(side_effect_fired_a, "side-effect callback fires when effective broadcast changes (override cleared)");
    check(side_effect_diff_a.contains("broadcast") && side_effect_diff_a["broadcast"] == false,
          "applied_diff reflects broadcast change from override true -> false");

    // Case B: Persisted broadcast=true, startup override=false, user sets broadcast=true
    config.set({{"broadcast", true}});
    config.set_broadcast_override(false);
    check(config.broadcast() == false, "effective broadcast is false due to override");

    bool side_effect_fired_b = false;
    json side_effect_diff_b = json::object();
    config.set({{"broadcast", true}}, [&](const json& diff) {
        side_effect_fired_b = true;
        side_effect_diff_b = diff;
    });
    check(config.broadcast() == true, "broadcast becomes true after set()");
    check(side_effect_fired_b, "side-effect callback fires when effective broadcast changes (override cleared)");
    check(side_effect_diff_b.contains("broadcast") && side_effect_diff_b["broadcast"] == true,
          "applied_diff reflects broadcast change from override false -> true");

    // 6. Test Server CLI parser flags
    lemon::CLIParser server_parser_1;
    const char* argv1[] = {"lemond", "--no-broadcast"};
    check(server_parser_1.parse(2, const_cast<char**>(argv1)) == 0, "CLIParser parses --no-broadcast");
    check(server_parser_1.get_config().broadcast.has_value() && *server_parser_1.get_config().broadcast == false,
          "--no-broadcast sets broadcast to false");

    lemon::CLIParser server_parser_2;
    const char* argv2[] = {"lemond", "--broadcast"};
    check(server_parser_2.parse(2, const_cast<char**>(argv2)) == 0, "CLIParser parses --broadcast");
    check(server_parser_2.get_config().broadcast.has_value() && *server_parser_2.get_config().broadcast == true,
          "--broadcast sets broadcast to true");

    lemon::CLIParser server_parser_3;
    const char* argv3[] = {"lemond"};
    check(server_parser_3.parse(1, const_cast<char**>(argv3)) == 0, "CLIParser parses default invocation");
    check(!server_parser_3.get_config().broadcast.has_value(),
          "default invocation leaves broadcast unset (nullopt)");

    // 7. Test legacy constructor JSON migration
    json legacy_cfg_true = {
        {"config_version", 2},
        {"port", 13305},
        {"host", "localhost"},
        {"no_broadcast", true}
    };
    RuntimeConfig config_legacy_true(legacy_cfg_true);
    check(config_legacy_true.broadcast() == false, "legacy no_broadcast: true migrates to broadcast: false in ctor");
    check(config_legacy_true.snapshot()["broadcast"] == false, "snapshot has broadcast: false");
    check(!config_legacy_true.snapshot().contains("no_broadcast"), "legacy no_broadcast removed from snapshot");

    json legacy_cfg_false = {
        {"config_version", 2},
        {"port", 13305},
        {"host", "localhost"},
        {"no_broadcast", false}
    };
    RuntimeConfig config_legacy_false(legacy_cfg_false);
    check(config_legacy_false.broadcast() == true, "legacy no_broadcast: false migrates to broadcast: true in ctor");
    check(config_legacy_false.snapshot()["broadcast"] == true, "snapshot has broadcast: true");
    check(!config_legacy_false.snapshot().contains("no_broadcast"), "legacy no_broadcast removed from snapshot");

    // 8. Test real ConfigFile::load() and save() migration on disk
    fs::path temp_test_dir = fs::temp_directory_path() / ("lemonade_test_discovery_" + std::to_string(std::time(nullptr)));
    fs::create_directories(temp_test_dir);

    try {
        // Write initial config with legacy no_broadcast: true
        fs::path cfg_file = temp_test_dir / "config.json";
        {
            std::ofstream out(cfg_file);
            out << "{\"config_version\": 2, \"port\": 13305, \"host\": \"localhost\", \"no_broadcast\": true}\n";
        }

        json loaded = ConfigFile::load(temp_test_dir.string(), temp_test_dir.string());
        check(loaded["broadcast"] == false, "ConfigFile::load migrates legacy no_broadcast: true to broadcast: false");
        check(!loaded.contains("no_broadcast"), "ConfigFile::load removes legacy no_broadcast from in-memory object");

        // Verify disk file was saved with migrated keys
        std::ifstream saved_in(cfg_file);
        json saved_disk = json::parse(saved_in);
        check(saved_disk["broadcast"] == false, "saved disk config.json contains broadcast: false");
        check(!saved_disk.contains("no_broadcast"), "saved disk config.json does not contain no_broadcast");
    } catch (const std::exception& e) {
        check(false, (std::string("ConfigFile::load disk migration failed: ") + e.what()).c_str());
    }
    fs::remove_all(temp_test_dir);

    // 9. Test deployment defaults (LEMONADE_DEFAULTS_PATH) with legacy no_broadcast: true
    fs::path temp_defaults_dir = fs::temp_directory_path() / ("lemonade_test_defaults_" + std::to_string(std::time(nullptr)));
    fs::create_directories(temp_defaults_dir);

    try {
        fs::path defaults_file = temp_defaults_dir / "custom_defaults.json";
        {
            std::ofstream out(defaults_file);
            out << "{\"no_broadcast\": true}\n";
        }

#ifdef _WIN32
        _putenv_s("LEMONADE_DEFAULTS_PATH", defaults_file.string().c_str());
#else
        setenv("LEMONADE_DEFAULTS_PATH", defaults_file.string().c_str(), 1);
#endif

        json defaults = ConfigFile::get_defaults();
        check(defaults["broadcast"] == false, "LEMONADE_DEFAULTS_PATH legacy no_broadcast: true overrides broadcast to false");
        check(!defaults.contains("no_broadcast"), "LEMONADE_DEFAULTS_PATH removes legacy no_broadcast from defaults object");

        // When user creates a fresh config under these defaults, broadcast is false
        fs::path fresh_cache_dir = temp_defaults_dir / "cache";
        json fresh_loaded = ConfigFile::load(fresh_cache_dir.string(), fresh_cache_dir.string());
        check(fresh_loaded["broadcast"] == false, "fresh config inherited broadcast=false from deployment defaults");

#ifdef _WIN32
        _putenv_s("LEMONADE_DEFAULTS_PATH", "");
#else
        unsetenv("LEMONADE_DEFAULTS_PATH");
#endif
    } catch (const std::exception& e) {
        check(false, (std::string("LEMONADE_DEFAULTS_PATH test failed: ") + e.what()).c_str());
    }
    fs::remove_all(temp_defaults_dir);

    // 10. Test that malformed legacy values in config.json are rejected
    fs::path malformed_dir = fs::temp_directory_path() / ("lemonade_test_malformed_" + std::to_string(std::time(nullptr)));
    fs::create_directories(malformed_dir);

    try {
        fs::path cfg_file = malformed_dir / "config.json";
        {
            std::ofstream out(cfg_file);
            out << "{\"config_version\": 2, \"port\": 13305, \"no_broadcast\": \"true\"}\n";
        }

        bool threw_malformed = false;
        try {
            ConfigFile::load(malformed_dir.string(), malformed_dir.string());
        } catch (const std::invalid_argument&) {
            threw_malformed = true;
        }
        check(threw_malformed, "ConfigFile::load rejects malformed non-boolean no_broadcast string");
    } catch (const std::exception& e) {
        check(false, (std::string("malformed test failed: ") + e.what()).c_str());
    }
    fs::remove_all(malformed_dir);

    // 11. beacon_listen: default, round-trip, and rejection of non-booleans
    {
        RuntimeConfig beacon_cfg(base_cfg);
        check(beacon_cfg.beacon_listen() == false, "beacon_listen defaults to false");
        beacon_cfg.set({{"beacon_listen", true}});
        check(beacon_cfg.beacon_listen() == true, "beacon_listen set to true");
        check(beacon_cfg.snapshot()["beacon_listen"] == true, "snapshot reflects beacon_listen: true");
        beacon_cfg.set({{"beacon_listen", false}});
        check(beacon_cfg.beacon_listen() == false, "beacon_listen set back to false");

        bool threw_set = false;
        try {
            beacon_cfg.set({{"beacon_listen", "yes"}});
        } catch (const std::invalid_argument&) {
            threw_set = true;
        }
        check(threw_set, "set() rejects non-boolean beacon_listen");

        bool threw_ctor = false;
        try {
            json bad = base_cfg;
            bad["beacon_listen"] = "true";
            RuntimeConfig bad_cfg(bad);
        } catch (const std::invalid_argument&) {
            threw_ctor = true;
        }
        check(threw_ctor, "constructor rejects non-boolean beacon_listen");

        json defaults = ConfigFile::get_defaults();
        check(defaults.contains("beacon_listen") && defaults["beacon_listen"] == false,
              "ConfigFile defaults include beacon_listen: false");
    }

    // 12. beacon_listen_interfaces: default, round-trip, and rejection of bad shapes
    {
        RuntimeConfig iface_cfg(base_cfg);
        check(iface_cfg.beacon_listen_interfaces().empty(), "beacon_listen_interfaces defaults to empty");
        iface_cfg.set({{"beacon_listen_interfaces", json::array({"br-lan", "br-guest"})}});
        check(iface_cfg.beacon_listen_interfaces() == std::vector<std::string>({"br-lan", "br-guest"}),
              "beacon_listen_interfaces round-trips");
        check(iface_cfg.snapshot()["beacon_listen_interfaces"] == json::array({"br-lan", "br-guest"}),
              "snapshot reflects beacon_listen_interfaces");
        iface_cfg.set({{"beacon_listen_interfaces", json::array()}});
        check(iface_cfg.beacon_listen_interfaces().empty(), "beacon_listen_interfaces cleared");

        for (const json& bad : {json("br-lan"), json::array({1}), json::array({""}),
                                json::array({std::string(65, 'x')})}) {
            bool threw = false;
            try {
                iface_cfg.set({{"beacon_listen_interfaces", bad}});
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            check(threw, ("set() rejects beacon_listen_interfaces = " + bad.dump().substr(0, 20)).c_str());
        }

        bool threw_ctor = false;
        try {
            json bad = base_cfg;
            bad["beacon_listen_interfaces"] = "br-lan";
            RuntimeConfig bad_cfg(bad);
        } catch (const std::invalid_argument&) {
            threw_ctor = true;
        }
        check(threw_ctor, "constructor rejects a non-array beacon_listen_interfaces");

        json defaults = ConfigFile::get_defaults();
        check(defaults.contains("beacon_listen_interfaces") && defaults["beacon_listen_interfaces"] == json::array(),
              "ConfigFile defaults include beacon_listen_interfaces: []");
    }

    // 13. beacon_trusted_relays: default, round-trip, and rejection of bad shapes
    {
        RuntimeConfig relay_cfg(base_cfg);
        check(relay_cfg.beacon_trusted_relays() == json::array(), "beacon_trusted_relays defaults to []");
        json good = json::array({{{"source", "192.168.60.10"},
                                  {"allow_hosts", json::array({"192.168.79.20", "10.20.0.0/16"})}}});
        relay_cfg.set({{"beacon_trusted_relays", good}});
        check(relay_cfg.beacon_trusted_relays() == good, "beacon_trusted_relays round-trips");
        check(relay_cfg.snapshot()["beacon_trusted_relays"] == good, "snapshot reflects beacon_trusted_relays");
        auto parsed = lemon::parse_beacon_trusted_relays(good);
        check(parsed.size() == 1 && parsed[0].source == 0xC0A83C0Au && parsed[0].allow_hosts.size() == 2 &&
                  parsed[0].allow_hosts[0].prefix == 32 && parsed[0].allow_hosts[1].network == 0x0A140000u &&
                  parsed[0].allow_hosts[1].prefix == 16,
              "parse_beacon_trusted_relays decodes source, a bare host and a CIDR");
        check(lemon::beacon_trusted_relays_to_json(parsed) ==
                  json::array({{{"source", "192.168.60.10"},
                                {"allow_hosts", json::array({"192.168.79.20/32", "10.20.0.0/16"})}}}),
              "trusted relays serialise with canonical CIDRs");
        relay_cfg.set({{"beacon_trusted_relays", json::array()}});
        check(relay_cfg.beacon_trusted_relays() == json::array(), "beacon_trusted_relays cleared");

        auto relay = [](const json& source, const json& hosts) {
            return json::array({{{"source", source}, {"allow_hosts", hosts}}});
        };
        json too_many = json::array();
        for (int i = 0; i < 17; ++i) {
            too_many.push_back({{"source", "10.0.0." + std::to_string(i + 1)},
                                {"allow_hosts", json::array({"10.1.0.1"})}});
        }
        json duplicate = relay("10.0.0.1", json::array({"10.1.0.1"}));
        duplicate.push_back(duplicate[0]);
        std::vector<std::pair<std::string, json>> bad_values = {
            {"not an array", json({{"source", "10.0.0.1"}})},
            {"entry not an object", json::array({"10.0.0.1"})},
            {"missing source", json::array({{{"allow_hosts", json::array({"10.1.0.1"})}}})},
            {"missing allow_hosts", json::array({{{"source", "10.0.0.1"}}})},
            {"unknown field", json::array({{{"source", "10.0.0.1"},
                                            {"allow_hosts", json::array({"10.1.0.1"})},
                                            {"trust", true}}})},
            {"source not a string", relay(42, json::array({"10.1.0.1"}))},
            {"source hostname", relay("relay.lan", json::array({"10.1.0.1"}))},
            {"source leading zero", relay("192.168.060.10", json::array({"10.1.0.1"}))},
            {"source with prefix", relay("192.168.60.10/32", json::array({"10.1.0.1"}))},
            {"source public", relay("8.8.8.8", json::array({"10.1.0.1"}))},
            {"allow_hosts empty", relay("10.0.0.1", json::array())},
            {"allow_hosts not array", relay("10.0.0.1", "10.1.0.1")},
            {"allow_hosts entry not string", relay("10.0.0.1", json::array({1}))},
            {"allow_hosts garbage", relay("10.0.0.1", json::array({"ai4"}))},
            {"allow_hosts public host", relay("10.0.0.1", json::array({"8.8.8.8"}))},
            {"allow_hosts public cidr", relay("10.0.0.1", json::array({"0.0.0.0/0"}))},
            {"allow_hosts cidr wider than rfc1918", relay("10.0.0.1", json::array({"192.168.0.0/15"}))},
            {"allow_hosts cidr straddles rfc1918", relay("10.0.0.1", json::array({"172.0.0.0/8"}))},
            {"allow_hosts prefix out of range", relay("10.0.0.1", json::array({"10.1.0.0/33"}))},
            {"allow_hosts empty prefix", relay("10.0.0.1", json::array({"10.1.0.0/"}))},
            {"allow_hosts host bits set", relay("10.0.0.1", json::array({"10.1.0.5/24"}))},
            {"allow_hosts too many", relay("10.0.0.1", json(std::vector<std::string>(33, "10.1.0.1")))},
            {"too many relays", too_many},
            {"duplicate source", duplicate},
        };
        for (const auto& [label, bad] : bad_values) {
            bool threw = false;
            std::string message;
            try {
                relay_cfg.set({{"beacon_trusted_relays", bad}});
            } catch (const std::invalid_argument& e) {
                threw = true;
                message = e.what();
            }
            check(threw && message.find("beacon_trusted_relays") != std::string::npos,
                  ("set() rejects beacon_trusted_relays: " + label).c_str());
        }
        check(relay_cfg.beacon_trusted_relays() == json::array(), "a rejected set() leaves the relays unchanged");

        bool threw_ctor = false;
        try {
            json bad = base_cfg;
            bad["beacon_trusted_relays"] = relay("8.8.8.8", json::array({"10.1.0.1"}));
            RuntimeConfig bad_cfg(bad);
        } catch (const std::invalid_argument&) {
            threw_ctor = true;
        }
        check(threw_ctor, "constructor rejects a malformed beacon_trusted_relays");

        json defaults = ConfigFile::get_defaults();
        check(defaults.contains("beacon_trusted_relays") && defaults["beacon_trusted_relays"] == json::array(),
              "ConfigFile defaults include beacon_trusted_relays: []");
    }

    return test_helpers::report_results("C++ config/discovery");
}
