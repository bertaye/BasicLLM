#pragma once

#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "BinaryFileLoader.h"
#include "Tensor.h"
#include "TensorOps.h"

// Chapter 3: the model
//
// The loading code and the forward pass (AttentionBlock, FeedForwardBlock,
// Forward) are GIVEN: the forward pass is the Chapter 2 answers, and the
// constructor comments cover the file format.
//
// Your task is the one function at the bottom: Generate, the loop that turns a
// one-shot next-token predictor into a text generator

class Model {
public:
    // From karpathy's llama2.c. For stories15M the values are:
    // dim=288, hidden_dim=768, n_layers=6, n_heads=6, n_kv_heads=6,
    // vocab_size=32000, seq_len=256
    struct ModelConfig {
        int dim;        // transformer dimension (features per token)
        int hidden_dim; // feed-forward hidden dimension
        int n_layers;   // number of transformer layers
        int n_heads;    // number of query heads
        int n_kv_heads; // number of key/value heads (== n_heads for stories15M)
        int vocab_size; // vocabulary size
        int seq_len;    // maximum sequence length
    };

    // Every weight the forward pass needs, as zero-copy views into the loaded
    // file buffer. Per-layer weights are stacked along axis 2
    struct TransformerWeights {
        basicllm::Tensor token_embedding_table; // {dim, vocab_size}
        basicllm::Tensor rms_att_weight;        // {dim, n_layers}
        basicllm::Tensor rms_ffn_weight;        // {dim, n_layers}
        basicllm::Tensor wq;                    // {dim, dim, n_layers} query projection
        basicllm::Tensor wk;                    // {dim, dim, n_layers} key projection
        basicllm::Tensor wv;                    // {dim, dim, n_layers} value projection
        basicllm::Tensor wo;                    // {dim, dim, n_layers} attention output projection
        basicllm::Tensor w1;                    // {dim, hidden_dim, n_layers} feed-forward gate
        basicllm::Tensor w2;                    // {hidden_dim, dim, n_layers} feed-forward down
        basicllm::Tensor w3;                    // {dim, hidden_dim, n_layers} feed-forward up
        basicllm::Tensor rms_final_weight;      // {dim}
        basicllm::Tensor wcls;                  // {dim, vocab_size} classifier
    };

    explicit Model(std::filesystem::path modelBinPath) {
        modelBin = BinaryFileLoader::loadFile(modelBinPath);
        if (!modelBin) {
            throw std::runtime_error("Failed to load model binary file");
        }

        std::memcpy(&config, modelBin.get(), sizeof(ModelConfig));

        const int headSize = config.dim / config.n_heads;

        // The weights sit back to back after the config header. Walk a float
        // pointer through the buffer and wrap each block in a non-owning view
        float* weightFloats = reinterpret_cast<float*>(modelBin.get() + sizeof(ModelConfig));

        weights.token_embedding_table = basicllm::Tensor::View(config.dim, config.vocab_size, 1, 1, (uint8_t*)weightFloats);
        weightFloats += (int64_t)config.vocab_size * config.dim;

        weights.rms_att_weight = basicllm::Tensor::View(config.dim, config.n_layers, 1, 1, (uint8_t*)weightFloats);
        weightFloats += (int64_t)config.n_layers * config.dim;

        weights.wq = basicllm::Tensor::View(config.dim, config.n_heads * headSize, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += (int64_t)config.n_layers * config.dim * config.n_heads * headSize;

        weights.wk = basicllm::Tensor::View(config.dim, config.n_kv_heads * headSize, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += (int64_t)config.n_layers * config.dim * config.n_kv_heads * headSize;

        weights.wv = basicllm::Tensor::View(config.dim, config.n_kv_heads * headSize, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += (int64_t)config.n_layers * config.dim * config.n_kv_heads * headSize;

        weights.wo = basicllm::Tensor::View(config.n_heads * headSize, config.dim, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += (int64_t)config.n_layers * config.dim * config.n_heads * headSize;

        weights.rms_ffn_weight = basicllm::Tensor::View(config.dim, config.n_layers, 1, 1, (uint8_t*)weightFloats);
        weightFloats += (int64_t)config.n_layers * config.dim;

        weights.w1 = basicllm::Tensor::View(config.dim, config.hidden_dim, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += (int64_t)config.n_layers * config.hidden_dim * config.dim;

        weights.w2 = basicllm::Tensor::View(config.hidden_dim, config.dim, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += (int64_t)config.n_layers * config.dim * config.hidden_dim;

        weights.w3 = basicllm::Tensor::View(config.dim, config.hidden_dim, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += (int64_t)config.n_layers * config.hidden_dim * config.dim;

        weights.rms_final_weight = basicllm::Tensor::View(config.dim, 1, 1, 1, (uint8_t*)weightFloats);
        weightFloats += config.dim;

        weightFloats += (int64_t)config.seq_len * (headSize / 2) * 2; // skip freq_cis_real / freq_cis_imag (legacy fields)

        // stories15M shares the classifier with the embedding table
        weights.wcls = config.vocab_size > 0
            ? weights.token_embedding_table
            : basicllm::Tensor::View(config.dim, config.vocab_size, 1, 1, (uint8_t*)weightFloats);
    }

    // The Chapter 2 answers, given

    basicllm::Tensor AttentionBlock(const basicllm::Tensor& x, int layer) {
        basicllm::Tensor h = basicllm::RMSNorm(x, basicllm::Slice(weights.rms_att_weight, 1, layer, 1));
        basicllm::Tensor q = basicllm::ApplyRoPE(basicllm::Project(h, weights.wq, layer), config.n_heads);
        basicllm::Tensor k = basicllm::ApplyRoPE(basicllm::Project(h, weights.wk, layer), config.n_kv_heads);
        basicllm::Tensor v = basicllm::Project(h, weights.wv, layer);
        basicllm::Tensor a = basicllm::MultiHeadAttention(q, k, v, config.n_heads, /*causal=*/true);
        return basicllm::Add(x, basicllm::Project(a, weights.wo, layer));
    }

    basicllm::Tensor FeedForwardBlock(const basicllm::Tensor& x, int layer) {
        basicllm::Tensor h   = basicllm::RMSNorm(x, basicllm::Slice(weights.rms_ffn_weight, 1, layer, 1));
        basicllm::Tensor act = basicllm::Mul(basicllm::SiLU(basicllm::Project(h, weights.w1, layer)),
                                             basicllm::Project(h, weights.w3, layer));
        return basicllm::Add(x, basicllm::Project(act, weights.w2, layer));
    }

    basicllm::Tensor Forward(const std::vector<int>& tokens) {
        basicllm::Tensor x = basicllm::GetRows(weights.token_embedding_table, tokens.data(), (int64_t)tokens.size());
        for (int l = 0; l < config.n_layers; ++l) {
            x = AttentionBlock(x, l);
            x = FeedForwardBlock(x, l);
        }
        x = basicllm::RMSNorm(x, weights.rms_final_weight);
        return basicllm::Project(x, weights.wcls, 0);
    }

    // Your task

    // Generate: the autoregressive loop. Feed the model a prompt, then keep
    // feeding it its OWN predictions until it has said enough.
    //
    //   promptTokens: the prompt as token ids (BOS already included)
    //   maxNewTokens: stop after this many generated tokens
    //   temperature:  passed straight to Sample; <= 0 means greedy
    //   rng:          the random state for Sample
    //   onToken:      if not null, call onToken(next) for every token AS SOON as
    //                 it is sampled. This is what lets a caller stream text to
    //                 the screen while generation is still running
    //
    //   Returns ONLY the newly generated tokens, not the prompt.
    //
    //   The loop:
    //     1. tokens = a copy of promptTokens
    //     2. repeat up to maxNewTokens times:
    //          a. logits = Forward(tokens)                      the WHOLE sequence, every step
    //          b. next   = Sample(logits, tokens.size() - 1, temperature, rng)
    //          c. if next is Tokenizer's EOS (token id 2), stop: the model chose
    //             to end the text
    //          d. call onToken(next) if a callback was given
    //          e. append next to tokens (and to the result)
    //          f. if tokens.size() reached config.seq_len, stop: the model
    //             cannot see further back than its maximum sequence length
    //
    //   Step (a) is deliberately wasteful: every iteration recomputes the
    //   forward pass over the ENTIRE sequence just to read the last column of
    //   logits. Feel how it slows down as the text grows. Fixing this is all of
    //   Chapter 4
    std::vector<int> Generate(const std::vector<int>& promptTokens, int maxNewTokens,
                              float temperature, std::mt19937& rng,
                              const std::function<void(int)>& onToken = nullptr) {
        // TODO: implement (Chapter 3, Task 3)
        return {};
    }

    // Public so the tests (and you, when poking around) can read them directly
    ModelConfig config{};
    TransformerWeights weights{};

private:
    std::unique_ptr<char[]> modelBin; // the raw file; every weight view points into it
};
