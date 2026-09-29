#include <algorithm>
#include <map>
#include <optional>
#include <string>

#include "lemon/error_types.h"
#include "lemon/model_manager.h"
#include "lemon/model_residency.h"
#include "lemon/route_decision_response.h"
#include "lemon/routing_classifier_services.h"
#include "lemon/routing_policy.h"
#include "lemon/routing_policy_parser.h"
#include "lemon/server/api_route.h"
#include "lemon/server/model_loader.h"
#include "lemon/server/router_dispatch.h"

namespace lemon {
namespace {

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

const RouteError kInvalidPolicy{400, "", "`policy` is invalid or internally inconsistent"};
const RouteError kPolicyOrModel{400, "", "Neither or both of `policy` and `model` are sent"};
const RouteError kRegisteredInvalid{
    400, "",
    "With `model`: the model is not a `collection.router`, `has_images`/`has_tools` are sent, "
    "`prompt` is not a string or an array of strings or token ids, `input` is not a string or "
    "array, or `messages` is not an array"};
const RouteError kRouterNotFound{404, "", "With `model`: no model by that name is registered"};
const RouteError kRouterEvaluationFailed{
    500, "",
    "With `model`: the router's registered policy failed to parse, or a helper model failed in a "
    "way its `on_error` policy does not absorb"};

const RouteArg kPolicy{"policy", ArgIn::JsonBody, {{"type", "object"}}, false, Support::Available,
                       "A `collection.router` policy document. `model_name` is accepted but is not required "
                       "for validation. See [Router Policies](../dev/router-policy.md). Required unless "
                       "`model` is sent, and refused alongside it."};
const RouteArg kModel{"model", ArgIn::JsonBody, {{"type", "string"}}, false, Support::Available,
                      "A registered `collection.router` model, by public or `user.` name, to classify "
                      "against instead of `policy`. The rest of the body is then read exactly as a chat, "
                      "completions or responses request to that router would be: `messages`, `prompt`, "
                      "`input`, `tools`, `metadata` and `max_tokens` build the routing context, and "
                      "`has_images`/`has_tools` are refused, since they come from the body itself."};
const RouteArg kPrompt{"prompt", ArgIn::JsonBody, {{"type", json::array({"string", "array"})}}, false, Support::Available,
                       "The prompt text to route. Defaults to `\"\"`. With `model`, an array of strings or "
                       "token ids is accepted too, as on completions."};
const RouteArg kHasImages{"has_images", ArgIn::JsonBody, {{"type", "boolean"}}, false, Support::Available,
                          "Simulate a request carrying image input. Defaults to `false`."};
const RouteArg kHasTools{"has_tools", ArgIn::JsonBody, {{"type", "boolean"}}, false, Support::Available,
                         "Simulate a request carrying tool definitions. Defaults to `false`."};
const RouteArg kMetadata{"metadata", ArgIn::JsonBody,
                         {{"type", "object"}, {"additionalProperties", {{"type", "string"}}}}, false, Support::Available,
                         "String-valued metadata pairs matched by `metadata` conditions."};

class RoutingValidateRoute : public ApiRoute {
public:
    using ApiRoute::ApiRoute;

    RouteSpec spec() const override {
        RouteSpec s;
        s.id = "router.routing_validate";
        s.methods = {"POST"};
        s.paths = {"routing/validate"};
        s.summary = "Test an ad-hoc or registered routing policy against a prompt without dispatching it";
        s.experimental = true;
        s.description =
            "Returns the decision a routing policy makes for a prompt, without registering the "
            "policy or sending the request to the chosen candidate. The Router Builder's **Test "
            "Prompt** tab uses it to iterate on a policy before attaching it to a "
            "`collection.router` model. Sent `model` instead of `policy`, it returns the decision a "
            "registered router would make for the request body, without dispatching it.";
        s.notes = {
            "### Classify Against a Registered Router\n\n"
            "Send `model` naming a registered `collection.router` model, instead of `policy`, to "
            "ask where that router would send a request without dispatching it. The router's "
            "registered policy is evaluated by the same code path dispatch uses, so the returned "
            "`decision` is the one a real request with the same body would get, and it equals the "
            "`x_lemonade_route` object a routed completion of that body returns with "
            "`route_trace: true`. The trace is always included, and the response carries the "
            "router's public name as `model` instead of `normalized_policy`.\n\n"
            "No candidate is loaded and nothing is forwarded to a backend. Deterministic rules are "
            "evaluated locally; model-backed conditions load and run their helper models as they "
            "do on dispatch, which can evict other models to make room. Classify-only requests are "
            "not counted in the routing metrics. Unlike dispatch, which falls back to the "
            "collection itself when routing fails, a failure here is an error.",
            "Validation is structural: every `candidates` entry, `default_model`, rule `route_to` "
            "and classifier model must be listed in `components`. Component names are not looked "
            "up in the model registry, so a policy can be tested before its candidates are "
            "downloaded, and registration's model-type checks, such as whether a "
            "`semantic_similarity` model can embed, do not run.",
            "Deterministic conditions (`keywords_any`, `regex`, `min_chars`, `metadata`, etc.) are "
            "evaluated locally. Model-backed conditions (`semantic_similarity`, `classifier`, "
            "`zero_shot` and `llm`, including `routing.router`) may load and run their models. A "
            "model failure follows the classifier's `on_error` policy (`match_false` by default), "
            "so routing continues to a later rule or to `default_model`. A residency conflict is "
            "the exception: when the classifier model cannot be resident alongside the models "
            "already loaded, the request returns 409 and no decision is made, because the policy "
            "cannot run on this machine in this combination.",
            "The example below uses explicit `routing.rules`, so its `normalized_policy` comes "
            "back unchanged. The `routing.router` shorthand is "
            "expanded into an `llm` classifier plus one rule per candidate, so a `routing` block "
            "written as:\n"
            "\n"
            "```json\n"
            "{\n"
            "  \"router\": {\n"
            "    \"type\": \"llm\",\n"
            "    \"model\": \"Qwen3-8B-GGUF\",\n"
            "    \"prompt\": \"Pick the best model for this request.\"\n"
            "  },\n"
            "  \"candidates\": [\"Qwen3-8B-GGUF\", \"vllm.qwen3-32b\"],\n"
            "  \"default_model\": \"Qwen3-8B-GGUF\"\n"
            "}\n"
            "```\n"
            "\n"
            "comes back with `router` replaced by these `classifiers` and `rules`:\n"
            "\n"
            "```json\n"
            "{\n"
            "  \"candidates\": [\"Qwen3-8B-GGUF\", \"vllm.qwen3-32b\"],\n"
            "  \"default_model\": \"Qwen3-8B-GGUF\",\n"
            "  \"classifiers\": [\n"
            "    {\n"
            "      \"id\": \"__router\",\n"
            "      \"type\": \"llm\",\n"
            "      \"model\": \"Qwen3-8B-GGUF\",\n"
            "      \"prompt\": \"Pick the best model for this request.\",\n"
            "      \"labels\": [\"Qwen3-8B-GGUF\", \"vllm.qwen3-32b\"]\n"
            "    }\n"
            "  ],\n"
            "  \"rules\": [\n"
            "    {\n"
            "      \"id\": \"__route_0\",\n"
            "      \"match\": {\"classifier\": \"__router\", \"label\": \"Qwen3-8B-GGUF\", \"min_score\": 1.0},\n"
            "      \"route_to\": \"Qwen3-8B-GGUF\"\n"
            "    },\n"
            "    {\n"
            "      \"id\": \"__route_1\",\n"
            "      \"match\": {\"classifier\": \"__router\", \"label\": \"vllm.qwen3-32b\", \"min_score\": 1.0},\n"
            "      \"route_to\": \"vllm.qwen3-32b\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            "\n"
            "Match `decision.matched_rule` against this document: a policy written with only "
            "`routing.router` has no rules of its own, only the generated `__route_0`, "
            "`__route_1`, etc.",
        };
        s.args = {kPolicy, kModel, kPrompt, kHasImages, kHasTools, kMetadata};

        const ordered_json decision = {
            {"type", "object"},
            {"description",
             "The routing decision for the prompt, with the per-condition `trace` included. See "
             "[Routing](#routing-collectionrouter) for the decision object."},
        };

        RouteResponse response;
        response.format = ResponseFormat::Json;
        response.schema = {
            {"type", "object"},
            {"required", ordered_json::array({"decision"})},
            {"properties", {
                {"decision", decision},
                {"normalized_policy", {{"type", "object"}, {"description", "The policy as evaluated, with any `routing.router` shorthand expanded. Present with `policy`."}}},
                {"model", {{"type", "string"}, {"description", "The router's public name. Present with `model`."}}},
            }},
        };
        s.responses = {response};

        RouteExample example;
        example.format = ResponseFormat::Json;
        example.args = ordered_json::parse(R"({
            "policy": {
                "version": "1",
                "recipe": "collection.router",
                "components": ["Qwen3-8B-GGUF", "vllm.qwen3-32b"],
                "routing": {
                    "candidates": ["Qwen3-8B-GGUF", "vllm.qwen3-32b"],
                    "default_model": "Qwen3-8B-GGUF",
                    "rules": [
                        {
                            "id": "code-to-big",
                            "match": {"keywords_any": ["def ", "function", "compile"]},
                            "route_to": "vllm.qwen3-32b"
                        }
                    ]
                }
            },
            "prompt": "please write a def to reverse a list"
        })");
        s.examples = {example};
        s.errors = {kBodyNotJson, kArgMissing, kArgInvalid, kInvalidPolicy, kPolicyOrModel,
                    kRegisteredInvalid, kRouterNotFound, kRouterEvaluationFailed,
                    ModelLoader::kRouterResidencyConflict};
        s.request_format = RequestFormat::Json;
        return s;
    }

    void handle(RouteRequest& req, httplib::Response& res) override {
        if (req.has(kModel)) {
            if (req.has(kPolicy)) {
                write_plain_error(res, kPolicyOrModel,
                    "'policy' and 'model' are mutually exclusive: send 'policy' to test an "
                    "ad-hoc policy, or 'model' to classify against a registered router");
                return;
            }
            handle_registered(req, res);
            return;
        }
        if (!req.has(kPolicy)) {
            write_plain_error(res, kPolicyOrModel, "'policy' must be a JSON object");
            return;
        }
        if (req.has(kPrompt) && !req.arg(kPrompt).is_string()) {
            write_plain_error(res, kArgInvalid, "'prompt' must be a string");
            return;
        }
        const std::string prompt = req.value(kPrompt, std::string());
        const bool has_images = req.value(kHasImages, false);
        const bool has_tools = req.value(kHasTools, false);

        std::map<std::string, std::string> metadata;
        if (req.has(kMetadata)) {
            const json& metadata_json = req.arg(kMetadata);
            for (auto it = metadata_json.begin(); it != metadata_json.end(); ++it) {
                metadata[it.key()] = it.value().get<std::string>();
            }
        }

        try {
            // The policy is not attached to a registered model, so any component name is
            // accepted as-is. require_declared_components keeps its default, so the
            // document must still be internally consistent, as registration requires.
            RoutingPolicyParseOptions options;
            options.resolve_component = [](const std::string& name) -> std::optional<std::string> {
                return name;
            };
            json normalized_routing;
            RoutePolicy policy = parse_route_policy_collection(
                req.arg(kPolicy), options, &normalized_routing);

            ModelLoader* loader = ctx_.model_loader;
            ClassifierServices services = make_router_classifier_services(
                *ctx_.router, [loader](const std::string& m) {
                    loader->ensure_loaded(m, json::object(), LoadPurpose::RoutingDependency);
                });
            RoutingPolicyEngine engine(std::move(policy), std::move(services));

            RouteContext route_context;
            route_context.input = prompt;
            route_context.params.chars = prompt.size();
            // A test prompt is a single turn with no history, so its whole-request
            // total is the prompt itself. Leaving this at 0 would make every
            // min_total_chars rule untestable here.
            route_context.params.total_chars = prompt.size();
            route_context.params.has_images = has_images;
            route_context.params.has_tools = has_tools;
            route_context.metadata = std::move(metadata);

            Decision decision = engine.route(route_context, /*want_trace=*/true);

            // Echo the policy actually evaluated (routing.router desugared into explicit
            // classifiers and rules), so a caller can match decision.matched_rule against
            // a rule that exists.
            json normalized_policy = req.arg(kPolicy);
            normalized_policy["routing"] = std::move(normalized_routing);

            json response = {
                {"decision", route_decision_to_json(decision)},
                {"normalized_policy", std::move(normalized_policy)},
            };
            res.set_content(response.dump(), "application/json");
        } catch (const RouterResidencyConflictException& e) {
            // The policy document is well-formed; the hardware just cannot host its
            // classifier right now. Reporting that as 400 would send the caller off
            // to debug a document that has nothing wrong with it.
            write_openai_error(res, ModelLoader::kRouterResidencyConflict, e.what(),
                               ErrorType::ROUTER_RESIDENCY_CONFLICT);
        } catch (const std::exception& e) {
            write_plain_error(res, kInvalidPolicy, std::string("Invalid routing policy: ") + e.what());
        }
    }

private:
    void handle_registered(RouteRequest& req, httplib::Response& res) {
        const json& body = req.body;
        // These only simulate request features for an ad-hoc policy; here the context is
        // derived from the body exactly as dispatch derives it, so an explicit flag would
        // silently disagree with what dispatch would see.
        for (const RouteArg* flag : {&kHasImages, &kHasTools}) {
            if (req.has(*flag)) {
                write_plain_error(res, kRegisteredInvalid,
                    "'" + flag->name + "' is not accepted with 'model'; it is derived from the "
                    "request's messages/tools as it is on dispatch");
                return;
            }
        }
        if (req.has(kPrompt)) {
            const json& prompt = req.arg(kPrompt);
            // Token ids and token-id arrays are valid completions prompts that dispatch
            // routes as empty text, so only other element types are rejected.
            const bool valid = prompt.is_string() ||
                std::all_of(prompt.begin(), prompt.end(), [](const json& part) {
                    return part.is_string() || part.is_number_integer() || part.is_array();
                });
            if (!valid) {
                write_plain_error(res, kRegisteredInvalid,
                                  "'prompt' must be a string or an array of strings or token ids");
                return;
            }
        }
        if (body.contains("input") && !body["input"].is_string() && !body["input"].is_array()) {
            write_plain_error(res, kRegisteredInvalid, "'input' must be a string or an array");
            return;
        }
        if (body.contains("messages") && !body["messages"].is_array()) {
            write_plain_error(res, kRegisteredInvalid, "'messages' must be an array");
            return;
        }

        try {
            const std::string requested_model =
                canonical_model_name(ctx_, req.arg(kModel).get<std::string>());
            json request = body;
            request[kModel.name] = requested_model;

            if (!ctx_.model_manager->model_exists(requested_model)) {
                write_plain_error(res, kRouterNotFound, "Model '" + requested_model + "' not found");
                return;
            }
            ModelInfo info = ctx_.model_manager->get_model_info(requested_model);
            if (!is_router_collection_recipe(info.recipe)) {
                write_plain_error(res, kRegisteredInvalid,
                    "Model '" + requested_model + "' is not a collection.router model "
                    "(recipe '" + info.recipe + "')");
                return;
            }

            std::optional<Decision> decision =
                evaluate_router_policy(ctx_, info, request, /*want_trace=*/true);
            if (!decision) {
                write_plain_error(res, kRouterEvaluationFailed,
                    "Router '" + requested_model + "' has no parsed routing policy");
                return;
            }

            json response = {
                {"model", ctx_.model_manager->get_public_model_name(info.model_name)},
                {"decision", route_decision_to_json(*decision)},
            };
            res.set_content(response.dump(), "application/json");
        } catch (const RouterResidencyConflictException& e) {
            write_openai_error(res, ModelLoader::kRouterResidencyConflict, e.what(),
                               ErrorType::ROUTER_RESIDENCY_CONFLICT);
        } catch (const std::exception& e) {
            write_plain_error(res, kRouterEvaluationFailed,
                              std::string("Routing evaluation failed: ") + e.what());
        }
    }
};

} // namespace

std::unique_ptr<ApiRoute> make_routing_validate_route(ServerContext& ctx) {
    return std::make_unique<RoutingValidateRoute>(ctx);
}

} // namespace lemon
