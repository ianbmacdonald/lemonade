#pragma once

#include "lemon/backends/backend_descriptor.h"

namespace lemon {
namespace backends {
namespace litert {

// The LiteRT backend descriptor (plain data). Header-only `inline const` so it
// links into both the lemonade CLI and lemond without a separate source file.
// Runs Google's LiteRT-LM runtime as a litert-lm-server subprocess serving an
// OpenAI-compatible /v1/chat/completions endpoint over a `.litertlm` model on
// the CPU backend. Like llamacpp's "system" variant, the binary is a pre-built
// executable on PATH — there is nothing to download or install.
inline const BackendDescriptor descriptor = {
    /*recipe*/          "litert",
    /*display_name*/    "LiteRT",
    /*binary*/          "litert-lm-server",
    /*config_section*/  "",  // defaults to recipe
    /*default_device*/  DEVICE_CPU,
    /*slot_policy*/     SlotPolicy::Standard,
    /*selectable_backend*/ false,
    /*uses_ctx_size*/   false,
    /*dynamic_models*/  false,
    /*options*/ {
        {"litert_args", "--litert-args", "", "ARGS",
         "Custom arguments to pass to litert-lm-server", "LiteRT Options"},
    },
    /*support*/ {
        {"system", {"linux"}, {{"cpu", {"x86_64", "arm64"}}}, "x86_64/ARM64 CPU"},
    },
    /*default_labels*/  {},
    /*required_checkpoints*/ {"main"},
    /*modality*/        "Text generation",
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

}  // namespace litert
}  // namespace backends
}  // namespace lemon
