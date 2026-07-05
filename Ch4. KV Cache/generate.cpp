// Chapter 4 demo: the Chapter 3 story demo, now running on your KV cache
//
// This file is GIVEN, there is nothing to implement here. Once your
// GenerateWithKVCache passes the tests, build and run:
//
//   ./build/ch4_story                                # default prompt, temperature 0.9
//   ./build/ch4_story "One day a dragon" 0.7 200     # prompt, temperature, max tokens
//
// It prints a tokens/sec figure at the end. Run the Chapter 3 demo with the
// same arguments and compare; that difference is your cache

#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Model.h"
#include "Tokenizer.h"

#ifndef MODEL_BIN_PATH
#define MODEL_BIN_PATH "../models/stories15M.bin"
#endif
#ifndef TOKENIZER_BIN_PATH
#define TOKENIZER_BIN_PATH "../models/tokenizer.bin"
#endif

int main(int argc, char** argv) {
    const std::string prompt      = argc > 1 ? argv[1] : "Once upon a time";
    const float       temperature = argc > 2 ? std::stof(argv[2]) : 0.9f;
    const int         maxTokens   = argc > 3 ? std::stoi(argv[3]) : 200;

    Model model(MODEL_BIN_PATH);
    Tokenizer tokenizer(TOKENIZER_BIN_PATH, model.config.vocab_size);

    std::vector<int> promptTokens = tokenizer.Encode(prompt, /*addBos=*/true, /*addEos=*/false);
    std::mt19937 rng(std::random_device{}());

    std::cout << prompt << std::flush;
    int prev = promptTokens.back();
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> generated = model.GenerateWithKVCache(promptTokens, maxTokens, temperature, rng,
        [&](int next) {
            std::cout << tokenizer.Decode(prev, next) << std::flush;
            prev = next;
        });
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::endl;

    if (generated.empty()) {
        std::printf("(no tokens generated: is GenerateWithKVCache implemented yet?)\n");
        return 0;
    }

    const double seconds = std::chrono::duration<double>(end - start).count();
    std::printf("\n%zu tokens in %.1f s (%.1f tokens/sec)\n",
                generated.size(), seconds, generated.size() / seconds);
    return 0;
}
