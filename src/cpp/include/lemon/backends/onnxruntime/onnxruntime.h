#pragma once

#include "lemon/backends/backend_descriptor.h"

#include <set>
#include <string>

namespace lemon {
namespace backends {
namespace onnxruntime {

// Arguments onnxruntime_args must not carry. onnxruntime_args comes from the
// /load request body, and --host would publish the engine on the LAN without
// lemond's API key.
inline const std::set<std::string>& reserved_custom_arg_flags() {
    static const std::set<std::string> flags = {"--model-path", "--port", "--host"};
    return flags;
}

// What an image model needs from ort-server. POST /classify/image exists only in
// the ianbmacdonald/ort-server line, from v0.2.0-musl; lemonade-sdk/ort-server,
// which is what lemonade downloads, has no such route. A version tag in cpu_bin
// installs from lemonade-sdk/ort-server, so only a path works.
inline constexpr const char* kImageRouteRequirement =
    "lemonade-sdk/ort-server, which lemonade downloads, has no image route; set "
    "onnxruntime.cpu_bin to the absolute path of an ort-server that serves /classify/image (for "
    "example ianbmacdonald/ort-server v0.2.0-musl or later), since a version tag there installs "
    "lemonade-sdk/ort-server";

// The ONNX Runtime backend descriptor (plain data). A generic ORT model server
// (see lemonade-sdk/ort-server); v1 serves text classification for the router's
// `classifier` condition type (issue #2592) on the CPU EP, and image models on
// /v1/images/classify with tflite-server's /classify/image contract. The
// subprocess is a self-contained ort-server bundle, like moonshine-server.
// "classification" stays first: it is the default mode the generated docs
// describe.
inline const BackendDescriptor descriptor = {
    /*recipe*/          "onnxruntime",
    /*display_name*/    "ONNX Runtime",
    /*binary*/          "ort-server",
    /*config_section*/  "",  // defaults to recipe
    /*default_device*/  DEVICE_CPU,
    /*slot_policy*/     SlotPolicy::Standard,
    /*selectable_backend*/ false,
    /*uses_ctx_size*/   false,
    /*dynamic_models*/  false,
    /*options*/ {
        {"onnxruntime_args", "--onnxruntime-args", "", "ARGS",
         "Custom arguments to pass to ort-server", "ONNX Runtime Options"},
    },
    /*support*/ {
        {"cpu", BackendTier::Experimental, BackendFormat::Native, {"windows"}, {{"cpu", {"x86_64"}}}, "x86_64 CPU"},
        {"cpu", BackendTier::Experimental, BackendFormat::Native, {"linux"}, {{"cpu", {"x86_64", "arm64"}}}, "x86_64/arm64 CPU"},
        {"cpu", BackendTier::Experimental, BackendFormat::Native, {"macos"}, {{"cpu", {"arm64"}}}, "arm64 CPU"},
    },
    /*supported_modes*/ {"classification", "image-classification"},
    /*required_checkpoints*/ {"main"},
    /*default_capabilities*/ {},
    /*web_display_name*/ "",
    /*rocm_channels*/   {},
    /*exposes_prometheus_metrics*/ false,
    /*rocm_requires_cwsr_fix*/ false,
    /*version_policy*/  VersionPolicy::Exact,
    /*self_manages_downloads*/ false,
    /*takes_args*/      true,
    /*arg_variants*/    {"cpu"},
    /*bin_variants*/    {"cpu"},
    /*config_extra*/    nlohmann::json::object(),
};

}  // namespace onnxruntime
}  // namespace backends
}  // namespace lemon
