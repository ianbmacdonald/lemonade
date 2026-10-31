#pragma once

#include "lemon/backends/backend_descriptor.h"

#include <set>
#include <string>

namespace lemon {
namespace backends {
namespace tflite {

// Arguments tflite_args must not carry. tflite_args comes from the /load
// request body: --host would publish the engine on the LAN without lemond's
// API key, and --weight-cache would make the engine write a file at a path
// the caller chooses.
inline const std::set<std::string>& reserved_custom_arg_flags() {
    static const std::set<std::string> flags = {
        "--model-path", "--port", "--host", "--weight-cache",
    };
    return flags;
}

// The TFLite backend descriptor (plain data). Serves text classification on
// /v1/classify through a tflite-server subprocess: the ort-server /classify
// contract on a LiteRT engine, so the same model directory layout (plus
// model.tflite) and the same scores. Image models go to /v1/images/classify and
// need tflite-server >= 0.2.0. Like litert, the binary is a pre-built executable
// on PATH; there is nothing to download or install. "classification" stays first:
// it is the default mode the generated docs describe.
inline const BackendDescriptor descriptor = {
    /*recipe*/          "tflite",
    /*display_name*/    "TFLite",
    /*binary*/          "tflite-server",
    /*config_section*/  "",  // defaults to recipe
    /*default_device*/  DEVICE_CPU,
    /*slot_policy*/     SlotPolicy::Standard,
    /*selectable_backend*/ false,
    /*uses_ctx_size*/   false,
    /*dynamic_models*/  false,
    /*options*/ {
        {"tflite_args", "--tflite-args", "", "ARGS",
         "Custom arguments to pass to tflite-server", "TFLite Options"},
    },
    /*support*/ {
        {"system", BackendTier::Experimental, BackendFormat::Native, {"linux"}, {{"cpu", {"x86_64", "arm64"}}}, "x86_64/ARM64 CPU"},
    },
    /*supported_modes*/ {"classification", "image-classification"},
    /*required_checkpoints*/ {"main"},
    /*default_capabilities*/ {},
    /*web_display_name*/ "",
    /*rocm_channels*/   {},
    /*exposes_prometheus_metrics*/ false,
    /*rocm_requires_cwsr_fix*/ false,
    /*version_policy*/  VersionPolicy::AtLeast,
    /*self_manages_downloads*/ false,
    /*takes_args*/      true,
    /*arg_variants*/    {},
    /*bin_variants*/    {},
    /*config_extra*/    nlohmann::json::object(),
};

}  // namespace tflite
}  // namespace backends
}  // namespace lemon
