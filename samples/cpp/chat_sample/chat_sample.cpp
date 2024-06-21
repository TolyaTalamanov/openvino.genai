// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    std::string accumulated_str = "";

    std::string model_path = argv[1];
    std::string device_name = argv[2];
    std::string prompt = argv[3];

    ov::genai::LLMPipeline pipe(model_path, device_name);
    ov::genai::GenerationConfig config = pipe.get_generation_config();

    config.temperature = 0.7;
    config.do_sample = false;
    config.top_p = 0.95;
    config.top_k = 40;
    config.max_new_tokens = 15;

    std::function<bool(std::string)> streamer = [](std::string word) { std::cout << word << std::flush; return false; };
    pipe.generate(prompt, config, streamer);
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
