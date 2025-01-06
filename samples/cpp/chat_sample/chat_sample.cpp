// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR>");
    }
    std::string prompt;
    std::string models_path = argv[1];

    std::string device = argv[2];
    ov::AnyMap properties;
    if (device == "NPU") {
        properties = {
                                    {"NPU_USE_NPUW", "YES"},
                                    {"NPUW_DEVICES", "CPU"},
                                    {"NPUW_ONLINE_PIPELINE", "NONE"},
                                    {"PREFILL_CONFIG", { }},
                                    {"GENERATE_CONFIG", { }}
                                };
    }
    ov::genai::LLMPipeline pipe(models_path, device, properties);
    
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    config.do_sample = false;
    //config.min_new_tokens = 1;
    //config.max_new_tokens = 50;
    //config.stop_strings = {"yellow."};
    //config.include_stop_str_in_output = false;
    //config.include_stop_str_in_output = true;

    std::function<bool(std::string)> streamer = [](std::string word) { 
        std::cout << word << std::flush;
        // Return flag corresponds whether generation should be stopped.
        // false means continue generation.
        return false; 
    };

    //pipe.start_chat();
    std::cout << "question:\n";
    while (std::getline(std::cin, prompt)) {
        pipe.generate(prompt, config, streamer);
        std::cout << "\n----------\n"
            "question:\n";
    }
    //pipe.finish_chat();
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
