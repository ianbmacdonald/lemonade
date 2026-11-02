#pragma once

#include "lemon/backends/backend_descriptor.h"

#include <set>
#include <string>

namespace lemon {
namespace backends {
namespace executorch {

// Arguments executorch_args must not carry; same reasons as
// tflite::reserved_custom_arg_flags (the request body sets them).
inline const std::set<std::string>& reserved_custom_arg_flags() {
    static const std::set<std::string> flags = {
        "--model-path", "--port", "--host", "--weight-cache",
    };
    return flags;
}

// What an image model needs from et-server: v0.2.0 is the first release with
// POST /classify/image (src/main.cpp at v0.1.0 has no such route).
inline constexpr const char* kImageRouteRequirement =
    "image models need et-server v0.2.0 or later (ianbmacdonald/et-server)";

// Serves text classification on /v1/classify through an et-server subprocess,
// which speaks the same ort-server /classify contract as tflite-server over an
// exported ExecuTorch program (model.pte); image models go to
// /v1/images/classify with tflite-server's /classify/image contract. The binary
// is pre-built and found on PATH; et-server ships x86_64 and aarch64 musl
// builds. "classification" stays first: it is the default mode the generated
// docs describe.
inline const BackendDescriptor descriptor = {
    /*recipe*/          "executorch",
    /*display_name*/    "ExecuTorch",
    /*binary*/          "et-server",
    /*config_section*/  "",  // defaults to recipe
    /*default_device*/  DEVICE_CPU,
    /*slot_policy*/     SlotPolicy::Standard,
    /*selectable_backend*/ false,
    /*uses_ctx_size*/   false,
    /*dynamic_models*/  false,
    /*options*/ {
        {"executorch_args", "--executorch-args", "", "ARGS",
         "Custom arguments to pass to et-server", "ExecuTorch Options"},
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

}  // namespace executorch
}  // namespace backends
}  // namespace lemon
