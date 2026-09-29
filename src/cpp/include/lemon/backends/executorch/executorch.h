#pragma once

#include "lemon/backends/backend_descriptor.h"

namespace lemon {
namespace backends {
namespace executorch {

// Serves text classification on /v1/classify through an et-server subprocess,
// which speaks the same ort-server /classify contract as tflite-server over an
// exported ExecuTorch program (model.pte). The binary is pre-built and found on
// PATH; only an x86_64 (musl) ExecuTorch build exists, hence the support list.
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
        {"system", {"linux"}, {{"cpu", {"x86_64"}}}, "x86_64 CPU"},
    },
    /*supported_modes*/ {"classification"},
    /*required_checkpoints*/ {"main"},
    /*default_capabilities*/ {},
    /*experimental*/    true,
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
