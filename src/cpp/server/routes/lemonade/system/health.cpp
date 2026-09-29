#include <string>
#include <vector>

#include "lemon/model_manager.h"
#include "lemon/router.h"
#include "lemon/runtime_config.h"
#include "lemon/server.h"
#include "lemon/server/api_route.h"
#include "lemon/server/http_listener.h"
#include "lemon/utils/process_memory.h"
#include "lemon/version.h"

namespace lemon {
namespace {

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

void add_memory_fields(json& obj, const utils::ProcessMemory& mem) {
    if (mem.has_rss) {
        obj["rss_mib"] = utils::to_mib(mem.rss_kib);
    }
    if (mem.has_anon) {
        obj["anon_mib"] = utils::to_mib(mem.anon_kib);
    }
}

ordered_json health_schema() {
    return ordered_json::parse(R"json({
        "type": "object",
        "required": ["status", "version", "model_loaded", "all_models_loaded", "max_models",
                     "pinned_models", "pinned_helper_models", "update_check_done"],
        "properties": {
            "status": {"const": "ok", "description": "Always `\"ok\"`."},
            "version": {"type": "string", "description": "Lemonade Server version."},
            "model_loaded": {"type": ["string", "null"], "description": "The most recently used loaded model, or `null` when none is loaded."},
            "update_check_done": {"type": "boolean", "description": "Whether the startup check for model updates has finished. Until it has, models' `update_available` can be incomplete."},
            "models_generation": {"type": "integer", "description": "Rises whenever the model list changes, including a model's `downloaded` or `update_available`. A client that polls health reads [`GET /v1/models`](./openai.md#get-v1models) again when it changes."},
            "model_sync": {
                "type": "object",
                "description": "The model sync, so a client that polls health can tell when one starts or finishes. Details are at [`GET /internal/models/sync/status`](./internal.md#get-internalmodelssyncstatus).",
                "required": ["status", "sync_id", "completed_sync_id"],
                "properties": {
                    "status": {"enum": ["idle", "in_progress", "success", "failed"], "description": "Status of the latest sync."},
                    "sync_id": {"type": "integer", "description": "The latest sync; `0` before any sync."},
                    "completed_sync_id": {"type": "integer", "description": "The most recent sync that finished."}
                }
            },
            "all_models_loaded": {
                "type": "array",
                "description": "Every loaded model.",
                "items": {
                    "type": "object",
                    "required": ["model_name", "checkpoint", "type", "device", "recipe",
                                 "recipe_options", "pinned", "pid", "backend_url", "last_use"],
                    "properties": {
                        "model_name": {"type": "string", "description": "Name of the loaded model"},
                        "checkpoint": {"type": "string", "description": "The model's main checkpoint."},
                        "last_use": {"type": "integer", "description": "Monotonic timestamp of last access (load or inference), in milliseconds"},
                        "type": {"type": "string", "description": "Model type: `llm`, `embedding`, `reranking`, `transcription`, `image`, `tts`, `audio-generation`, `classification` or `mesh`."},
                        "device": {"type": "string", "description": "Devices the model runs on, joined by `|`: `cpu`, `gpu` and `npu`, such as `gpu|npu`, or `none`."},
                        "pinned": {"type": "boolean", "description": "Whether the model is pinned, which exempts it from auto-eviction"},
                        "is_busy": {"type": "boolean", "description": "Whether the model has active requests or maintenance in progress"},
                        "is_streaming": {"type": "boolean", "description": "Whether the model is generating output tokens: `true` from the first chunk until every streaming request completes."},
                        "backend_url": {"type": "string", "description": "URL of the backend process serving the model."},
                        "pid": {"type": "integer", "description": "Process ID of the backend process serving the model."},
                        "rss_mib": {"type": "number", "description": "Resident memory of the backend process in MiB (`VmRSS`, rounded to one decimal). Includes memory-mapped model weights and shared libraries, so pages shared between backends can be counted more than once. Linux only; a best-effort sample, omitted when it can't be read. Only the direct backend process is measured, not its children."},
                        "anon_mib": {"type": "number", "description": "Private anonymous memory of the backend process in MiB (`RssAnon`, rounded to one decimal): the part of `rss_mib` not backed by files. Same availability as `rss_mib`."},
                        "launch_command": {"type": "array", "items": {"type": "string"}, "description": "The command that started the model's backend, program first, with the values actually used: an automatic `ctx_size` appears as a number, and flags Lemonade added are included. Absent for cloud models."},
                        "recipe": {"type": "string", "description": "Recipe used to load the model, e.g. `llamacpp` or `flm`."},
                        "recipe_options": {"type": "object", "description": "Options the model was loaded with, such as `ctx_size` or `llamacpp_args`."},
                        "residency_class": {"type": "string", "description": "`standard`, or `routing_helper` for a model a router policy keeps resident."},
                        "slot_pool": {"type": "string", "description": "The loaded-model limit the model counts against, or `unmetered`."},
                        "status": {"type": "string", "description": "Load state: `loading`, `ready`, `in_use`, `downsizing`, `downsized`, `evicting` or `unloaded`."},
                        "backend_alive": {"type": "boolean", "description": "Always `true` for a listed model."},
                        "backend_health": {"type": "string", "description": "`ready`, or `busy` while the backend has a request in flight."},
                        "loaded": {"type": "boolean", "description": "Always `true` for a listed model."},
                        "watchdog_reset": {"type": "boolean", "description": "Always `false` for a listed model, since a model whose backend the watchdog reset is left out."},
                        "watchdog_reset_reason": {"type": "string", "description": "Why the watchdog reset the backend; present after a reset."},
                        "max_context_window": {"type": "integer", "description": "Largest context the model supports, when known."},
                        "cost_input_per_million": {"type": "number", "description": "Cloud models only: USD per million input tokens, when the provider reports it."},
                        "cost_output_per_million": {"type": "number", "description": "Cloud models only: USD per million output tokens, when the provider reports it."}
                    }
                }
            },
            "pinned_models": {"type": "object", "additionalProperties": {"type": "integer", "description": "How many pinned models of this type are loaded."}, "description": "Loaded pinned models, counted per model type, such as `llm` or `embedding`."},
            "max_models": {
                "type": "object",
                "description": "Maximum number of models that can be loaded simultaneously per type (set via `max_loaded_models` in [Server Configuration](../guide/configuration/README.md))",
                "properties": {
                    "llm": {"type": "integer", "description": "Maximum LLM/chat models"},
                    "embedding": {"type": "integer", "description": "Maximum embedding models"},
                    "reranking": {"type": "integer", "description": "Maximum reranking models"},
                    "transcription": {"type": "integer", "description": "Maximum speech-to-text models"},
                    "image": {"type": "integer", "description": "Maximum image models"},
                    "tts": {"type": "integer", "description": "Maximum text-to-speech models"},
                    "classification": {"type": "integer", "description": "Maximum classification models"}
                },
                "additionalProperties": {"type": "integer"}
            },
            "websocket_port": {"type": "integer", "description": "Port of the dedicated WebSocket server for the [Realtime Audio Transcription API](./openai.md#ws-realtime) and [Log Streaming API](#log-streaming-api-websocket), present while it runs; the main port serves both APIs too. The port is OS-assigned, or set by the `websocket_port` setting in [Server Configuration](../guide/configuration/README.md)."},
            "telemetry": {
                "type": "object",
                "description": "Telemetry state.",
                "required": ["enabled"],
                "properties": {
                    "enabled": {"type": "boolean", "description": "Whether telemetry collection is active"},
                    "captures": {"type": "array", "items": {"enum": ["inputs", "outputs", "thinking"]}, "description": "The parts of each request telemetry captures, present when `enabled` is `true`."}
                }
            },
            "pinned_helper_models": {"type": "object", "additionalProperties": {"type": "integer", "description": "How many pinned routing-helper models of this type are loaded."}, "description": "Loaded pinned routing-helper models, counted per model type."},
            "beacon_listener": {
                "type": "object",
                "description": "State of the LAN beacon listener, controlled by the `beacon_listen` setting in [Server Configuration](../guide/configuration/README.md). The list is display-only: Lemonade never contacts a heard host, sends it credentials, or routes requests to it. When the listener is off, this is `{\"enabled\": false}`.",
                "required": ["enabled"],
                "properties": {
                    "enabled": {"type": "boolean", "description": "Whether the listener is running."},
                    "supported": {"type": "boolean", "description": "`false` on Windows, where the listener is not available in this release."},
                    "listening": {"type": "boolean", "description": "Whether at least one UDP socket is bound."},
                    "error": {"type": ["string", "null"], "description": "Platform-level error, or `null`. If the listener thread could not be started, `listening` is `false`, `sockets` is empty and `error` says why; toggling `beacon_listen` retries."},
                    "sockets": {"type": "array", "items": {"type": "object"}, "description": "One entry per bound broadcast address (`address`, `bound`, `error`). The listener binds each listened interface's subnet broadcast address (worked out from its address and netmask) plus `255.255.255.255`, never `0.0.0.0`, so it does not take the `127.0.0.1` beacons that `lemonade scan` relies on."},
                    "interfaces": {"type": "array", "items": {"type": "object"}, "description": "The interfaces being listened on: `name`, `address`, `netmask` and `broadcast`. By default this is every interface with an RFC1918 address. The `beacon_listen_interfaces` setting narrows it; on a gateway whose WAN side is also RFC1918, set it to the LAN bridges or WAN-side hosts are listed too."},
                    "interface_allowlist": {"type": "array", "items": {"type": "string"}, "description": "The `beacon_listen_interfaces` setting, or `[]` when every RFC1918 interface is heard."},
                    "self_instance_id": {"type": "string", "description": "This server's beacon `instance_id`, used to drop its own beacons."},
                    "self_port": {"type": "integer", "description": "This server's HTTP port, used to drop its own beacons."},
                    "ttl_seconds": {"type": "integer", "description": "Seconds a host stays listed after its last beacon."},
                    "max_hosts": {"type": "integer", "description": "Maximum listed hosts."},
                    "distinct_instances": {"type": "integer", "description": "Number of distinct servers heard; a multi-homed server has one row per interface."},
                    "hosts": {"type": "array", "items": {"type": "object"}, "description": "Heard servers, sorted by hostname: `hostname` (sanitised to `[A-Za-z0-9._-]`), `url`, `source_ip`, `instance_id` (or `null` for older servers), `first_seen` and `last_seen` (Unix seconds), `age_seconds`."},
                    "stats": {"type": "object", "description": "Counters per outcome: `accepted`, `refreshed`, `self`, `bad_source` (not RFC1918), `bad_payload`, `url_mismatch`, `rate_limited`, `table_full`, `wrong_interface` (arrived on an interface that is not being listened on), `evicted`."}
                }
            },
            "server_process": {
                "type": "object",
                "description": "The Lemonade Server process itself.",
                "required": ["pid"],
                "properties": {
                    "pid": {"type": "integer", "description": "Process ID of the server."},
                    "rss_mib": {"type": "number", "description": "Resident memory of the server in MiB, with the same meaning and availability as the per-model field."},
                    "anon_mib": {"type": "number", "description": "Private anonymous memory of the server in MiB, with the same meaning and availability as the per-model field."}
                }
            }
        }
    })json");
}

class HealthRoute : public ApiRoute {
public:
    using ApiRoute::ApiRoute;

    RouteSpec spec() const override {
        RouteSpec s;
        s.id = "lemonade.health";
        s.methods = {"GET"};
        s.paths = {"health"};
        s.summary = "Check the server's status and loaded models";
        s.description =
            "Reports the server's status, including the models it has loaded.";
        s.notes = {
            "`HEAD` returns `200 OK` with an empty body.",
        };

        RouteResponse response;
        response.format = ResponseFormat::Json;
        response.schema = health_schema();
        s.responses = {response};

        RouteExample example;
        example.format = ResponseFormat::Json;
        example.setup = {{"openai.chat_completions", "non-streaming"}};
        example.args = json::object();
        s.examples = {example};
        s.quiet_log = true;
        return s;
    }

    void handle(RouteRequest& req, httplib::Response& res) override {
        if (req.http.method == "HEAD") {
            res.status = 200;
            return;
        }

        RuntimeConfig& config = *ctx_.config;
        json response = {{"status", "ok"}};
        response["version"] = LEMON_VERSION_STRING;

        json telemetry_info = {{"enabled", config.telemetry_enabled()}};
        if (config.telemetry_enabled()) {
            std::vector<std::string> captures;
            if (!config.telemetry_hide_inputs()) {
                captures.push_back("inputs");
            }
            if (!config.telemetry_hide_outputs()) {
                captures.push_back("outputs");
            }
            if (!config.telemetry_hide_thinking()) {
                captures.push_back("thinking");
            }
            telemetry_info["captures"] = captures;
        }
        response["telemetry"] = telemetry_info;

        std::string loaded_model = ctx_.router->get_loaded_model();
        response["model_loaded"] = loaded_model.empty() ? json(nullptr) : json(loaded_model);
        response["all_models_loaded"] = ctx_.router->get_all_loaded_models();
        // Sampled after the router lock is released: a backend evicted in the gap
        // just loses its memory fields for this one response.
        for (auto& model : response["all_models_loaded"]) {
            const int pid = model.value("pid", 0);
            if (pid > 0) {
                add_memory_fields(model, utils::read_process_memory(pid));
            }
        }
        json server_process = {{"pid", utils::current_process_id()}};
        add_memory_fields(server_process, utils::read_self_memory());
        response["server_process"] = server_process;
        response["max_models"] = ctx_.router->get_max_model_limits();
        response["pinned_models"] = ctx_.router->get_pinned_model_counts();
        response["pinned_helper_models"] = ctx_.router->get_pinned_helper_counts();

        const int websocket_port = ctx_.listener->websocket_port();
        if (websocket_port > 0) {
            response["websocket_port"] = websocket_port;
        }

        response["update_check_done"] = ctx_.server->update_check_done();
        response["beacon_listener"] = ctx_.listener->beacon_listener_status();

        if (ctx_.model_manager) {
            const json sync = ctx_.model_manager->sync_status();
            response["model_sync"] = {
                {"status", sync.value("status", "idle")},
                {"sync_id", sync.value("sync_id", std::uint64_t{0})},
                {"completed_sync_id", sync.value("completed_sync_id", std::uint64_t{0})},
            };
            response["models_generation"] = ctx_.model_manager->models_generation();
        }

        res.set_content(response.dump(), "application/json");
    }
};

} // namespace

std::unique_ptr<ApiRoute> make_health_route(ServerContext& ctx) {
    return std::make_unique<HealthRoute>(ctx);
}

} // namespace lemon
