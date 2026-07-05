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

// Chapter 4: the model
//
// The loading code, the forward pass and the naive Generate are GIVEN: they are
// the Chapter 2 and 3 answers. The constructor comments cover the file format.
//
// Your tasks are the three functions at the bottom: AttentionBlockWithKVCache,
// ForwardWithKVCache and GenerateWithKVCache

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

    // The Chapter 2 and 3 answers, given

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
        std::vector<int> tokens = promptTokens;
        std::vector<int> generated;
        for (int step = 0; step < maxNewTokens; ++step) {
            basicllm::Tensor logits = Forward(tokens);
            int next = (int)basicllm::Sample(logits, (int64_t)tokens.size() - 1, temperature, rng);
            if (next == 2) break; // EOS: the model chose to end the text
            if (onToken) onToken(next);
            tokens.push_back(next);
            generated.push_back(next);
            if ((int)tokens.size() >= config.seq_len) break;
        }
        return generated;
    }

    // Your tasks

    // One attention block of layer `layer` for ONE token at position `pos`,
    // reading and writing the KV cache. The cached twin of Chapter 2's
    // AttentionBlock; compare them line by line when you are done.
    //
    //   x is {dim, 1}: a single token's residual stream.
    //   keyCache / valueCache are the FULL caches, {kv_dim, seq_len, n_layers}.
    //
    //   Steps:
    //     1. h = RMSNorm(x, this layer's row of rms_att_weight)
    //     2. q = ApplyRoPE(Project(h, wq, layer), n_heads,    10000.0f, pos)
    //        k = ApplyRoPE(Project(h, wk, layer), n_kv_heads, 10000.0f, pos)
    //        v = Project(h, wv, layer)
    //        The positionOffset argument finally earns its keep: this tensor is
    //        ONE column, so row index 0 must mean position `pos`, not position 0
    //     3. slice THIS layer's cache planes (axis 2 is the layer axis):
    //          kCacheLayer = Slice(keyCache,   2, layer, 1)     {kv_dim, seq_len}
    //          vCacheLayer = Slice(valueCache, 2, layer, 1)
    //        These are zero-copy views, so when MultiHeadAttentionWithKVCache
    //        writes into them, it writes the real cache
    //     4. a = MultiHeadAttentionWithKVCache(q, k, v, kCacheLayer, vCacheLayer, pos, n_heads)
    //     5. return Add(x, Project(a, wo, layer))               the residual, as always
    basicllm::Tensor AttentionBlockWithKVCache(const basicllm::Tensor& x, int layer,
                                               basicllm::Tensor& keyCache, basicllm::Tensor& valueCache,
                                               int64_t pos) {
        // TODO: implement (Chapter 4, Task 2)
        return {};
    }

    // The forward pass for ONE token at position `pos`. The cached twin of
    // Chapter 2's Forward.
    //
    //     1. x = GetRows(token_embedding_table, &token, 1)      {dim, 1}
    //     2. for each layer l in [0, n_layers):
    //          x = AttentionBlockWithKVCache(x, l, keyCache, valueCache, pos)
    //          x = FeedForwardBlock(x, l)                       unchanged from Chapter 2:
    //                                                           it never looks at other tokens
    //     3. x = RMSNorm(x, rms_final_weight)
    //     4. return Project(x, wcls, 0)                         logits {vocab_size, 1}
    //
    //   Note what is NOT here: no loop over the sequence. One token in, one
    //   column of logits out, and the past is entirely in the cache
    basicllm::Tensor ForwardWithKVCache(int token, int64_t pos,
                                        basicllm::Tensor& keyCache, basicllm::Tensor& valueCache) {
        // TODO: implement (Chapter 4, Task 3)
        return {};
    }

    // Generate with the KV cache: same contract as Generate (same arguments,
    // same return, same onToken streaming), radically cheaper per token.
    //
    //     1. allocate the caches:
    //          kv_dim = (dim / n_heads) * n_kv_heads
    //          keyCache   = Tensor(kv_dim, seq_len, n_layers)
    //          valueCache = Tensor(kv_dim, seq_len, n_layers)
    //        Freshly allocated memory is uninitialized and that is fine: the
    //        attention only ever reads columns [0, pos], which are always
    //        written before they are read.
    //     2. PREFILL: feed the prompt one token at a time, just to warm the cache:
    //          for p in [0, promptTokens.size()):
    //              logits = ForwardWithKVCache(promptTokens[p], p, keyCache, valueCache)
    //        Only the LAST logits matter; the earlier ones are a byproduct.
    //     3. DECODE: pos = promptTokens.size(); repeat up to maxNewTokens times:
    //          a. next = Sample(logits, 0, temperature, rng)    logits is {vocab, 1},
    //                                                           so the position is 0
    //          b. if next is EOS (token id 2), stop
    //          c. call onToken(next) if given, append next to the result
    //          d. if pos >= config.seq_len, stop                context limit
    //          e. logits = ForwardWithKVCache(next, pos, keyCache, valueCache); ++pos
    //
    //   At temperature 0 this must produce EXACTLY the same tokens as the naive
    //   Generate. Same model, same math, same story; only the wasted work is gone
    std::vector<int> GenerateWithKVCache(const std::vector<int>& promptTokens, int maxNewTokens,
                                         float temperature, std::mt19937& rng,
                                         const std::function<void(int)>& onToken = nullptr) {
        // TODO: implement (Chapter 4, Task 4)
        return {};
    }

    // Public so the tests (and you, when poking around) can read them directly
    ModelConfig config{};
    TransformerWeights weights{};

private:
    std::unique_ptr<char[]> modelBin; // the raw file; every weight view points into it
};
