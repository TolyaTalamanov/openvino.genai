// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (4 > argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<WAV_FILE_PATH>\" <DEVICE>");
    }

    std::string model_path = argv[1];
    std::string wav_file_path = argv[2];
    std::string device = argv[3];

    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);

    ov::genai::WhisperPipeline pipeline{model_path, device};

    ov::genai::WhisperGenerationConfig config{model_path + "/generation_config.json"};
    config.max_new_tokens = 100;
    // 'task' and 'language' parameters are supported for multilingual models only
    config.language = "<|en|>";
    config.task = "transcribe";

    auto streamer = [](std::string word) {
        std::cout << word;
        return false;
    };

    //pipeline.generate(raw_speech, config, streamer);
    auto ans = pipeline.generate(raw_speech, config);

    std::cout << ans << std::endl;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
