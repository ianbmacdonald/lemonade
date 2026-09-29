#pragma once

#include <string>
#include <vector>

namespace lemon_cli {

// Labels `lemonade pull --label` accepts for a manual user.* registration.
inline const std::vector<std::string>& pull_labels() {
    static const std::vector<std::string> labels = {
        "chat",
        "coding",
        "dflash",
        "embeddings",
        "hot",
        "image-classification",
        "mtp",
        "reasoning",
        "reranking",
        "tool-calling",
        "vision"
    };
    return labels;
}

} // namespace lemon_cli
