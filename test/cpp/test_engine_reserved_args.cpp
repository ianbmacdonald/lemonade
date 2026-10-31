#include <initializer_list>
#include <iostream>
#include <set>
#include <string>

#include "lemon/backends/executorch/executorch.h"
#include "lemon/backends/litert/litert.h"
#include "lemon/backends/onnxruntime/onnxruntime.h"
#include "lemon/backends/tflite/tflite.h"
#include "lemon/utils/custom_args.h"

namespace {

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) {
        std::cout << "PASS: " << label << std::endl;
    } else {
        std::cout << "FAIL: " << label << std::endl;
        ++failures;
    }
}

// The engine *_args come from the /load request body, so every flag that moves
// the engine's bind address, port, model or files it writes must be refused in
// both spellings.
void check_engine(const std::string& name, const std::set<std::string>& reserved,
                  std::initializer_list<const char*> must_refuse) {
    for (const char* flag : must_refuse) {
        expect(!lemon::utils::validate_custom_args(std::string(flag) + " x", reserved).empty(),
               name + " rejects " + flag);
        expect(!lemon::utils::validate_custom_args(std::string(flag) + "=x", reserved).empty(),
               name + " rejects " + flag + "=");
    }
    expect(!lemon::utils::validate_custom_args("--threads 2 --host 0.0.0.0", reserved).empty(),
           name + " rejects --host after an allowed flag");
    expect(lemon::utils::validate_custom_args("--threads 2 --verbose", reserved).empty(),
           name + " allows arguments Lemonade does not manage");
}

}  // namespace

int main() {
    check_engine("tflite", lemon::backends::tflite::reserved_custom_arg_flags(),
                 {"--model-path", "--port", "--host", "--weight-cache"});
    check_engine("executorch", lemon::backends::executorch::reserved_custom_arg_flags(),
                 {"--model-path", "--port", "--host", "--weight-cache"});
    check_engine("litert", lemon::backends::litert::reserved_custom_arg_flags(),
                 {"--model", "--port", "--host", "--cache-dir"});
    check_engine("onnxruntime", lemon::backends::onnxruntime::reserved_custom_arg_flags(),
                 {"--model-path", "--port", "--host"});

    if (failures == 0) {
        std::cout << "All engine reserved-argument tests passed" << std::endl;
        return 0;
    }
    std::cout << failures << " test(s) failed" << std::endl;
    return 1;
}
