#ifndef MODEL_H
#define MODEL_H

#include "BinaryFileLoader.h"
#include <iostream>
#include "Tensor.h"
#include "Tokenizer.h"
#include "Logger.h"
#include "TensorOps.h"
#include <chrono>
#include <random>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <iomanip>
#include <mutex>
#include <condition_variable>

class Model{
    public:
    //From Karpathy's llama2.c
    struct ModelConfig{
        int dim; // transformer dimension
        int hidden_dim; // for ffn layers
        int n_layers; // number of layers
        int n_heads; // number of query heads
        int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
        int vocab_size; // vocabulary size, usually 256 (byte-level)
        int seq_len; // max sequence length
    };
    struct TransformerWeights{
        // token embedding table
        basicllm::Tensor token_embedding_table;    // (vocab_size, dim)
        // weights for rmsnorms
        basicllm::Tensor rms_att_weight; // (layer, dim) rmsnorm weights
        basicllm::Tensor rms_ffn_weight; // (layer, dim)
        // weights for matmuls
        basicllm::Tensor wq; // (layer, dim, dim)
        basicllm::Tensor wk; // (layer, dim, dim)
        basicllm::Tensor wv; // (layer, dim, dim)
        basicllm::Tensor wo; // (layer, dim, dim)
        // weights for ffn
        basicllm::Tensor w1; // (layer, hidden_dim, dim)
        basicllm::Tensor w2; // (layer, dim, hidden_dim)
        basicllm::Tensor w3; // (layer, hidden_dim, dim)
        // final rmsnorm
        basicllm::Tensor rms_final_weight; // (dim,)
        // freq_cis for RoPE relatively positional embeddings
        basicllm::Tensor freq_cis_real; // (seq_len, dim/2)
        basicllm::Tensor freq_cis_imag; // (seq_len, dim/2)
        // (optional) classifier weights for the logits, on the last layer
        basicllm::Tensor wcls;
    };

    struct Metrics
    {
        std::atomic<uint64_t> generatedTokens = 0;
        std::atomic<double> generateSeconds = 0.0;
        std::atomic<double> lastTokenGenerateMilliSeconds = 0.0;
        void clear()
        {
            generatedTokens = 0;
            generateSeconds = 0.0;
            lastTokenGenerateMilliSeconds = 0.0;
        }
    };

    explicit Model(std::filesystem::path modelBinPath, std::filesystem::path tokenizerBinPath)
    {
        modelBin = BinaryFileLoader::loadFile(modelBinPath);
        if(!modelBin)
        {
            throw std::runtime_error("Failed to load model binary file.");
        }

        memcpy(&config, modelBin.get(), sizeof(ModelConfig));

        Logger::LogDebug("Model config: dim="  + std::to_string(config.dim) +
                        ", hidden_dim=" + std::to_string(config.hidden_dim) +
                        ", n_layers=" + std::to_string(config.n_layers) +
                        ", n_heads=" + std::to_string(config.n_heads) +
                        ", n_kv_heads=" + std::to_string(config.n_kv_heads) +
                        ", vocab_size=" + std::to_string(config.vocab_size) +
                        ", seq_len=" + std::to_string(config.seq_len));

        int headSize = config.dim / config.n_heads; //width of a single head / size of one head

        char* weightBytesStart = modelBin.get() + sizeof(ModelConfig);
        float* weightFloats = reinterpret_cast<float*>(weightBytesStart);

        // Token Embedding Table is a matrix like this:
        /*
            [
                [0.1, 0.2, 0.3, ...], // embedding for token 0; size is (dim,)
                [0.4, 0.5, 0.6, ...], // embedding for token 1; size is (dim,)
                ...
            ]
        */
        weights.token_embedding_table = basicllm::Tensor::View(config.dim, config.vocab_size, 1, 1, (uint8_t*)weightFloats);
        weightFloats += config.vocab_size * config.dim;

        weights.rms_att_weight = basicllm::Tensor::View(config.dim, config.n_layers, 1, 1, (uint8_t*)weightFloats);
        weightFloats += config.n_layers * config.dim;

        weights.wq = basicllm::Tensor::View(config.dim, config.n_heads * headSize, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += config.n_layers * config.dim * config.n_heads * headSize;

        weights.wk = basicllm::Tensor::View(config.dim, config.n_kv_heads * headSize, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += config.n_layers * config.dim * config.n_kv_heads * headSize;

        weights.wv = basicllm::Tensor::View(config.dim, config.n_kv_heads * headSize, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += config.n_layers * config.dim * config.n_kv_heads * headSize;

        weights.wo = basicllm::Tensor::View(config.n_heads * headSize, config.dim, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += config.n_layers * config.dim * config.n_heads * headSize;

        weights.rms_ffn_weight = basicllm::Tensor::View(config.dim, config.n_layers, 1, 1, (uint8_t*)weightFloats);
        weightFloats += config.n_layers * config.dim;

        weights.w1 = basicllm::Tensor::View(config.dim, config.hidden_dim, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += config.n_layers * config.hidden_dim * config.dim;

        weights.w2 = basicllm::Tensor::View(config.hidden_dim, config.dim, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += config.n_layers * config.dim * config.hidden_dim;

        weights.w3 = basicllm::Tensor::View(config.dim, config.hidden_dim, config.n_layers, 1, (uint8_t*)weightFloats);
        weightFloats += config.n_layers * config.hidden_dim * config.dim;

        weights.rms_final_weight = basicllm::Tensor::View(config.dim, 1, 1, 1, (uint8_t*)weightFloats);
        weightFloats += config.dim;
        weightFloats += config.seq_len * (headSize / 2) * 2; // Skip freq_cis_real and freq_cis_imag
        weights.wcls = config.vocab_size > 0 ? weights.token_embedding_table : basicllm::Tensor::View(config.dim, config.vocab_size, 1, 1, (uint8_t*)weightFloats);

        // Some sketching on runtime tensor sizes, based on loaded config values:
        // 
        //int kv_dim = headSize * config.n_kv_heads; // dimension of the key/value vectors

        //x = basicllm::Tensor(config.dim,1,1,1);
        //xb = basicllm::Tensor(config.dim,1,1,1);
        //xb2 = basicllm::Tensor(config.dim,1,1,1);

        //hb = basicllm::Tensor(config.hidden_dim,1,1,1);
        //hb2 = basicllm::Tensor(config.hidden_dim,1,1,1);

        //q = basicllm::Tensor(config.dim,1,1,1);
        //
        ////How did we select the Tensor dimensions here?
        ////kv_dim = k_dim = v_dim = n_kv_heads * headSize
        ////key_cache stores each token's key vector. One row = one token, width kv_dim,
        ////holding ALL kv-heads concatenated:
        ///*
        //    [
        //         => token 0's key, size (kv_dim,) = [ head0 (headSize) | head1 | ... ]
        //        [ k0_h0_0..k0_h0_{hs-1},  k0_h1_0..k0_h1_{hs-1},  ... ],

        //         => token 1's key
        //        [ k1_h0_0..k1_h0_{hs-1},  k1_h1_0..k1_h1_{hs-1},  ... ],
        //        ...
        //    ]
        //    and this whole matrix repeats per layer (n_layers deep)
        //*/
        //key_cache = basicllm::Tensor(kv_dim, config.seq_len, config.n_layers); 
        //value_cache = basicllm::Tensor(kv_dim, config.seq_len, config.n_layers); 

        ////Here, note that; normally we would have a Tensor of size (seq_len, seq_len, n_heads) for RAW attention scores
        ////But we are doing autoregression. And we are using KV-Cache. So; we are only interested with a 'single' token at the inference time.
        ////Hence, the Tensor becomes (seq_len, n_heads). the contiguous dimension is seq_len. and each seq_len row repeated n_heads times.
        //att = basicllm::Tensor(config.seq_len, config.n_heads, 1, 1); //one score per head per token
        //logits = basicllm::Tensor(config.vocab_size,1,1,1); //one score per vocab token.

        ////Initialize the Tokenizer
        tokenizer = std::move(Tokenizer(tokenizerBinPath, config.vocab_size));
        Logger::LogDebug("Initialized tokenizer with vocab size: " + std::to_string(tokenizer.VocabSize()));
    }

    ~Model()
    {
        for(auto& thr : metricThreads)
        {
            if (thr.joinable())
            {
               thr.join();
            }
        }
    }

    std::string Generate(const std::string& prompt, int maxNewTokens, float temperature, std::ostream& outStream, bool stream = true)
    {
        metrics.clear();
        // Encode prompt: BOS prepended, no EOS.
        std::vector<int> tokens = tokenizer.Encode(prompt, /*addBos=*/true, /*addEos=*/false);

        std::string output;
        int prevToken = tokens.empty() ? Tokenizer::BosToken : tokens.back();
        std::cout << prompt << std::flush;
        auto startTime = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < maxNewTokens && step < config.seq_len; ++step)
        {
            auto singleTokenStartTime = std::chrono::high_resolution_clock::now();
            basicllm::Tensor logits = Forward(tokens);                       // {vocab, seqLen}
            int next = static_cast<int>(basicllm::Sample(logits, static_cast<int64_t>(tokens.size()) - 1, temperature, rng)); // temperature<=0 => greedy

            if (next == Tokenizer::EosToken) break;                        // model ended the sequence

            auto piece = tokenizer.Decode(prevToken, next);
            auto singleTokenEndTime = std::chrono::high_resolution_clock::now();

            // Serialize the token write + metric update with the live-metrics thread so
            // their terminal escape sequences never interleave (both take streamMutex).
            {
                const char* colors[3] = { "\033[31m", "\033[32m", "\033[34m" }; // red / green / blue
                std::lock_guard<std::mutex> lock(streamMutex);
                if (stream)
                {
                    outStream << colors[metrics.generatedTokens % 3] << piece << "\033[0m" << std::flush;
                }
                metrics.generatedTokens++;
                metrics.lastTokenGenerateMilliSeconds =
                    std::chrono::duration<double, std::milli>(singleTokenEndTime - singleTokenStartTime).count();
            }
            streamCV.notify_one();

            output += piece;
            prevToken = next;
            tokens.push_back(next);

            if (static_cast<int>(tokens.size()) >= config.seq_len) break;  // hit model's context limit
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        metrics.generateSeconds.store(std::chrono::duration<double>(endTime - startTime).count());

        printMetrics.store(false);
        streamCV.notify_all();
        for (auto& thr : metricThreads)
        {
            if (thr.joinable())
            {
                thr.join();
            }
        }
        metricThreads.clear();
        std::cout << std::endl << std::flush;

        return output;
    }

    std::string GenerateWithKVCache(const std::string& prompt, int maxNewTokens, float temperature, std::ostream& outStream, bool stream = true)
    {
        metrics.clear();
        std::vector<int> tokens = tokenizer.Encode(prompt, /*addBos=*/true, /*addEos=*/false);

        // Per-generation KV cache: input-specific, so it lives here, not as a member.
        const int headSize = config.dim / config.n_heads;
        const int kv_dim = headSize * config.n_kv_heads;
        basicllm::Tensor keyCache(kv_dim, config.seq_len, config.n_layers);
        basicllm::Tensor valueCache(kv_dim, config.seq_len, config.n_layers);

        std::string output;
        int prevToken = tokens.empty() ? Tokenizer::BosToken : tokens.back();
        std::cout << prompt << std::flush;

        auto startTime = std::chrono::high_resolution_clock::now();

        int token = tokens[0];   // start by feeding the first prompt token
        for (int step = 0; step < (maxNewTokens + tokens.size() - 1) && step < config.seq_len; ++step)
        {
            auto singleTokenStartTime = std::chrono::high_resolution_clock::now();
            basicllm::Tensor logits = ForwardWithKVCache(token, step, keyCache, valueCache); // {vocab, 1}

            int next;
            if (step < static_cast<int64_t>(tokens.size()) - 1)
            {
                // PREFILL: still inside the prompt -> we already know the next token, just feed it.
                next = tokens[step + 1];
            }
            else
            {
                // DECODE: past the prompt -> actually sample from the logits.
                next = static_cast<int>(basicllm::Sample(logits, 0, temperature, rng)); // col 0: only column
                if (next == Tokenizer::EosToken) break;

                auto piece = tokenizer.Decode(prevToken, next);
                auto singleTokenEndTime = std::chrono::high_resolution_clock::now();

                // Serialize the token write + metric update with the live-metrics thread so
               // their terminal escape sequences never interleave (both take streamMutex).
                {
                    const char* colors[3] = { "\033[31m", "\033[32m", "\033[34m" }; // red / green / blue
                    std::lock_guard<std::mutex> lock(streamMutex);
                    if (stream)
                    {
                        outStream << colors[metrics.generatedTokens % 3] << piece << "\033[0m" << std::flush;
                    }
                    metrics.generatedTokens++;
                    metrics.lastTokenGenerateMilliSeconds =
                        std::chrono::duration<double, std::milli>(singleTokenEndTime - singleTokenStartTime).count();
                }
                streamCV.notify_one();
                output += piece;
                prevToken = next;
            }

            token = next;
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        metrics.generateSeconds.store(std::chrono::duration<double>(endTime - startTime).count());
        printMetrics.store(false);
        streamCV.notify_all();
        for (auto& thr : metricThreads)
        {
            if (thr.joinable())
            {
                thr.join();
            }
        }
        metricThreads.clear();
        std::cout << std::endl << std::flush;
        return output;
    }

    basicllm::Tensor Forward(const std::vector<int>& tokens)
    {
        const int64_t seqLen = static_cast<int64_t>(tokens.size());

        // Embed the tokens: {dim, seqLen}
        basicllm::Tensor x = basicllm::GetRows(weights.token_embedding_table, tokens.data(), seqLen);

        for (int l = 0; l < config.n_layers; ++l)
        {
            // Attention block
            basicllm::Tensor h = basicllm::RMSNorm(x, basicllm::Slice(weights.rms_att_weight, 1, l, 1)); //hidden state

            basicllm::Tensor q = basicllm::ApplyRoPE(basicllm::Project(h, weights.wq, l), config.n_heads);
            basicllm::Tensor k = basicllm::ApplyRoPE(basicllm::Project(h, weights.wk, l), config.n_kv_heads);
            basicllm::Tensor v = basicllm::Project(h, weights.wv, l);

            basicllm::Tensor a = basicllm::MultiHeadAttention(q, k, v, config.n_heads, /*causal=*/true);
            x = basicllm::Add(x, basicllm::Project(a, weights.wo, l));   // residual
            basicllm::Tensor h2   = basicllm::RMSNorm(x, basicllm::Slice(weights.rms_ffn_weight, 1, l, 1));

            // Feed-forward block (SwiGLU)
            basicllm::Tensor gate = basicllm::Project(h2, weights.w1, l);
            basicllm::Tensor up   = basicllm::Project(h2, weights.w3, l);
            basicllm::Tensor act  = basicllm::Mul(basicllm::SiLU(gate), up);
            x = basicllm::Add(x, basicllm::Project(act, weights.w2, l)); // residual
        }

        x = basicllm::RMSNorm(x, weights.rms_final_weight);            // final norm
        return basicllm::Project(x, weights.wcls, 0);                  // logits {vocab, seqLen}
    }

    basicllm::Tensor ForwardWithKVCache(int token, int64_t pos, basicllm::Tensor& kCache, basicllm::Tensor& vCache)
    {

        basicllm::Tensor x = basicllm::GetRows(weights.token_embedding_table, &token, 1);

        for (int l = 0; l < config.n_layers; ++l)
        {
            // Attention block
            basicllm::Tensor h = basicllm::RMSNorm(x, basicllm::Slice(weights.rms_att_weight, 1, l, 1)); //hidden state

            basicllm::Tensor q = basicllm::ApplyRoPE(basicllm::Project(h, weights.wq, l), config.n_heads, 10000.0f, pos);
            basicllm::Tensor k = basicllm::ApplyRoPE(basicllm::Project(h, weights.wk, l), config.n_kv_heads, 10000.0f, pos);
            basicllm::Tensor v = basicllm::Project(h, weights.wv, l);

            //Get this layer's k-v caches
            basicllm::Tensor kCache_layer = basicllm::Slice(kCache, 2, l, 1);
            basicllm::Tensor vCache_layer = basicllm::Slice(vCache, 2, l, 1);

            basicllm::Tensor a = basicllm::MultiHeadAttentionWithKVCache(q, k, v, kCache_layer, vCache_layer, pos, config.n_heads);

            x = basicllm::Add(x, basicllm::Project(a, weights.wo, l));   // residual
            basicllm::Tensor h2   = basicllm::RMSNorm(x, basicllm::Slice(weights.rms_ffn_weight, 1, l, 1));

            // Feed-forward block (SwiGLU)
            basicllm::Tensor gate = basicllm::Project(h2, weights.w1, l);
            basicllm::Tensor up   = basicllm::Project(h2, weights.w3, l);
            basicllm::Tensor act  = basicllm::Mul(basicllm::SiLU(gate), up);
            x = basicllm::Add(x, basicllm::Project(act, weights.w2, l)); // residual
        }

        x = basicllm::RMSNorm(x, weights.rms_final_weight);            // final norm
        return basicllm::Project(x, weights.wcls, 0);                  // logits {vocab, 1}
    }

    void enableLiveMetrics()
    {
        printMetrics.store(true);
        std::cout << std::endl; //padding
        metricThreads.emplace_back([this]() {
            auto lastTokenCount = this->metrics.generatedTokens.load();
            while (this->printMetrics.load())
            {
                std::unique_lock lock(streamMutex);
                streamCV.wait(lock, [&lastTokenCount, this]() {return lastTokenCount != this->metrics.generatedTokens.load() || !this->printMetrics.load(); });
                lastTokenCount = this->metrics.generatedTokens.load();

                std::cout << "\033[s";
                std::cout << "\033[1;1H";
                std::cout << std::left << std::setw(15);
                std::cout << std::setprecision(3) << 1000.0/this->metrics.lastTokenGenerateMilliSeconds.load();
                std::cout << " tokens/sec";
                std::cout << "\033[u";
                std::cout.flush();
            }
            });
    }

    Metrics* getMetrics()
    {
        return &metrics;
    }
    private:
    
    ModelConfig config{};
    TransformerWeights weights{};
    //RunState runState{};
    std::unique_ptr<char[]> modelBin;
    Tokenizer tokenizer;
    Metrics metrics{};
    std::atomic<bool> printMetrics{false};
    std::mutex streamMutex;
    std::vector<std::thread> metricThreads;
    std::condition_variable streamCV;
    std::mt19937 rng{std::random_device{}()}; // temperature-sampling RNG; use {42} for reproducible runs
};

#endif //MODEL_H