#pragma once

#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

#include "BinaryFileLoader.h"
#include "Tensor.h"
#include "TensorOps.h"

// Chapter 2: the model
//
// Everything down to the constructor is GIVEN: the config struct, the weight
// tensors, and the code that maps the stories15M.bin file into them. The format
// is llama2.c's, and the constructor comments walk through it if you are
// curious; for this chapter you can treat loading as a solved problem and just
// use the weights.
//
// Your tasks are the three functions at the bottom: AttentionBlock,
// FeedForwardBlock, and Forward

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

    // Every weight the forward pass needs. All of these are zero-copy views into
    // the loaded file buffer; no float is ever duplicated.
    // The per-layer weights are STACKED: axis 2 is the layer axis, so wq holds
    // n_layers matrices in one tensor and Slice/Project pick one out
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

        // stories15M shares the classifier with the embedding table (a positive
        // vocab_size marks that in the llama2.c format)
        weights.wcls = config.vocab_size > 0
            ? weights.token_embedding_table
            : basicllm::Tensor::View(config.dim, config.vocab_size, 1, 1, (uint8_t*)weightFloats);
    }

    // Your tasks. Same rules as always: the recipe is in the comment, the body is
    // empty, `return {};` keeps it compiling until you implement it

    // One attention block of layer `layer`, INCLUDING the residual add.
    //
    //   x is {dim, seqLen}, the residual stream. Steps:
    //     1. h = RMSNorm(x, this layer's row of rms_att_weight)   <- Slice along axis 1
    //     2. q = ApplyRoPE(Project(h, wq, layer), n_heads)
    //        k = ApplyRoPE(Project(h, wk, layer), n_kv_heads)
    //        v = Project(h, wv, layer)                            <- no RoPE on v
    //     3. a = MultiHeadAttention(q, k, v, n_heads, causal = true)
    //     4. return Add(x, Project(a, wo, layer))                 <- the residual
    //
    //   Note step 4 adds to x, the block INPUT, not to h. The normalization only
    //   feeds the attention; the stream itself flows around it untouched
    basicllm::Tensor AttentionBlock(const basicllm::Tensor& x, int layer) {
        // TODO: implement (Chapter 2, Task 6)
        return {};
    }

    // One feed-forward block of layer `layer` (SwiGLU), INCLUDING the residual add.
    //
    //   x is {dim, seqLen}. Steps:
    //     1. h    = RMSNorm(x, this layer's row of rms_ffn_weight)
    //     2. gate = Project(h, w1, layer)                         {hidden_dim, seqLen}
    //        up   = Project(h, w3, layer)                         {hidden_dim, seqLen}
    //     3. act  = Mul(SiLU(gate), up)                           <- this is SwiGLU
    //     4. return Add(x, Project(act, w2, layer))               <- back to {dim, seqLen}
    basicllm::Tensor FeedForwardBlock(const basicllm::Tensor& x, int layer) {
        // TODO: implement (Chapter 2, Task 7)
        return {};
    }

    // The full forward pass: token ids in, next-token logits out.
    //
    //     1. x = GetRows(token_embedding_table, tokens)           {dim, seqLen}
    //     2. for each layer l in [0, n_layers):
    //          x = AttentionBlock(x, l)
    //          x = FeedForwardBlock(x, l)
    //     3. x = RMSNorm(x, rms_final_weight)
    //     4. return Project(x, wcls, 0)                           logits {vocab_size, seqLen}
    //
    //   Column s of the result scores every vocabulary token as the successor of
    //   position s. The prediction for "what comes next" is the LAST column
    basicllm::Tensor Forward(const std::vector<int>& tokens) {
        // TODO: implement (Chapter 2, Task 8)
        return {};
    }

    // Public so the tests (and you, when poking around) can read them directly
    ModelConfig config{};
    TransformerWeights weights{};

private:
    std::unique_ptr<char[]> modelBin; // the raw file; every weight view points into it
};
