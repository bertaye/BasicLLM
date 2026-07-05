// Chapter 3 demo: watch your Generate write a story, streamed token by token
//
// This file is GIVEN, there is nothing to implement here. Once your Generate
// passes the tests, build and run:
//
//   ./build/ch3_story                                # default prompt, temperature 0.9
//   ./build/ch3_story "One day a dragon" 0.7 80      # prompt, temperature, max tokens
//
// Temperature 0 always retells the same story (greedy); higher values branch.
// Each run at temperature > 0 uses a fresh random seed, so the story changes

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
    const int         maxTokens   = argc > 3 ? std::stoi(argv[3]) : 50;

    Model model(MODEL_BIN_PATH);
    Tokenizer tokenizer(TOKENIZER_BIN_PATH, model.config.vocab_size);

    std::vector<int> promptTokens = tokenizer.Encode(prompt, /*addBos=*/true, /*addEos=*/false);
    std::mt19937 rng(std::random_device{}());

    std::cout << prompt << std::flush;
    int prev = promptTokens.back();
    std::vector<int> generated = model.Generate(promptTokens, maxTokens, temperature, rng,
        [&](int next) {
            std::cout << tokenizer.Decode(prev, next) << std::flush;
            prev = next;
        });
    std::cout << std::endl;

    if (generated.empty()) {
        std::printf("(no tokens generated: is Generate implemented yet?)\n");
    }
    return 0;
}
