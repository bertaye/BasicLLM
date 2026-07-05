// Chapter 4 test runner
//
// Implement MultiHeadAttentionWithKVCache in TensorOps.h (Task 1), then the
// three Model.h functions (Tasks 2-4), and build and run after each one:
//   [       OK ]  your implementation matches the expected result
//   [  FAILED  ]  it ran but produced the wrong numbers
//   [ SKIPPED  ]  still a stub, or an earlier task it needs is missing
//
// The theme of every test in this chapter: the cache path must produce the
// SAME numbers as the full-recompute path, it is only allowed to be faster.
// Tasks 2-4 need the model file; fetch it with `python models/download.py`
// from the repository root. Build Release, Task 4 times a real comparison

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "Tensor.h"
#include "TensorOps.h"
#include "Model.h"
#include "Tokenizer.h"

#ifndef MODEL_BIN_PATH
#define MODEL_BIN_PATH "../models/stories15M.bin"
#endif
#ifndef TOKENIZER_BIN_PATH
#define TOKENIZER_BIN_PATH "../models/tokenizer.bin"
#endif

using basicllm::Tensor;

namespace {

// A minimal test harness. Each test is a function taking a TestContext; the
// macros below record pass/fail/skip on it
struct TestContext {
    bool failed  = false;
    bool skipped = false;

    // operator<< returns void so a skip reads as `return ctx_.Skip() << "...";`
    struct Sink {};
    Sink Skip() { skipped = true; return {}; }

    void Fail(const char* what) {
        failed = true;
        std::printf("    check failed: %s\n", what);
    }
};

template <typename T>
void operator<<(TestContext::Sink, const T&) {}

// Mimicing google gtest; it is a relatively heavy dependency for the project so we just go with these:

#define TEST(suite, name) void suite##_##name(TestContext &ctx_)
#define SKIP_TEST()       return ctx_.Skip()
#define ASSERT_EQ(a, b)      do { if (!((a) == (b))) { ctx_.Fail(#a " == " #b); return; } } while (0)
#define EXPECT_NEAR(a, b, t) do { if (std::fabs((a) - (b)) > (t)) ctx_.Fail(#a " near " #b); } while (0)
#define EXPECT_FLOAT_EQ(a, b) do { if (std::fabs((a) - (b)) > 1e-5f) ctx_.Fail(#a " == " #b); } while (0)
#define EXPECT_TRUE(x)        do { if (!(x)) ctx_.Fail(#x); } while (0)

// An empty stub returns a Tensor with no data. We skip (not fail) those, so a test
// never calls At() on an unimplemented op, which would crash
bool implemented(const Tensor& t) { return t.data != nullptr; }

constexpr float kTol      = 1e-3f;
constexpr float kLogitTol = 1e-2f;

// "Once upon a time" encoded by the reference tokenizer, BOS token (1) prepended
const std::vector<int> kPromptTokens = {1, 9038, 2501, 263, 931};

// The stories15M model, loaded once. Returns nullptr if the file is not there
Model* TheModel() {
    static std::unique_ptr<Model> model = [] {
        if (!std::filesystem::exists(MODEL_BIN_PATH)) return std::unique_ptr<Model>();
        return std::make_unique<Model>(MODEL_BIN_PATH);
    }();
    return model.get();
}

// Probe whether Task 1 is implemented, without crashing on a stub
bool Task1Ready() {
    Tensor q(2, 1), k(2, 1), v(2, 1), kc(2, 4), vc(2, 4);
    q.At(0, 0) = q.At(1, 0) = k.At(0, 0) = k.At(1, 0) = v.At(0, 0) = v.At(1, 0) = 1.0f;
    return implemented(MultiHeadAttentionWithKVCache(q, k, v, kc, vc, 0, 1));
}

// Task 1
TEST(KVCache, MultiHeadAttentionWithKVCache) {
    // 2 heads of headDim=2, a 3-token sequence with arbitrary fixed values.
    // Feeding the tokens one at a time through the cache must reproduce, column
    // by column, what the full causal MultiHeadAttention computes in one shot
    const int64_t dim = 4, seqLen = 3;
    Tensor Q(dim, seqLen), K(dim, seqLen), V(dim, seqLen);
    for (int64_t r = 0; r < seqLen; ++r) {
        for (int64_t c = 0; c < dim; ++c) {
            Q.At(c, r) = 0.10f * (float)(c + 1) - 0.20f * (float)r;
            K.At(c, r) = 0.05f * (float)c + 0.15f * (float)r - 0.3f;
            V.At(c, r) = (float)c + 10.0f * (float)r;
        }
    }

    Tensor expected = MultiHeadAttention(Q, K, V, /*numHeads=*/2, /*causal=*/true);

    Tensor kCache(dim, seqLen), vCache(dim, seqLen);
    for (int64_t p = 0; p < seqLen; ++p) {
        Tensor qp = Slice(Q, 1, p, 1);
        Tensor kp = Slice(K, 1, p, 1);
        Tensor vp = Slice(V, 1, p, 1);

        Tensor out = MultiHeadAttentionWithKVCache(qp, kp, vp, kCache, vCache, p, 2);
        if (!implemented(out)) SKIP_TEST() << "not implemented yet";

        ASSERT_EQ(out.elementCounts[0], dim);
        ASSERT_EQ(out.elementCounts[1], 1);
        for (int64_t c = 0; c < dim; ++c) {
            EXPECT_NEAR(out.At(c, 0), expected.At(c, p), kTol); // column p of the one-shot result
            EXPECT_FLOAT_EQ(kCache.At(c, p), K.At(c, p));       // the cache was actually written
            EXPECT_FLOAT_EQ(vCache.At(c, p), V.At(c, p));
        }
    }
}

// Task 2
TEST(KVCache, AttentionBlockWithKVCache) {
    Model* model = TheModel();
    if (!model) SKIP_TEST() << "stories15M.bin not found, run models/download.py";
    if (!Task1Ready()) SKIP_TEST() << "needs MultiHeadAttentionWithKVCache (Task 1) first";

    const int kvDim = (model->config.dim / model->config.n_heads) * model->config.n_kv_heads;
    Tensor keyCache(kvDim, model->config.seq_len, model->config.n_layers);
    Tensor valueCache(kvDim, model->config.seq_len, model->config.n_layers);

    // feed the prompt's embeddings through layer 0's block one token at a time
    Tensor x = GetRows(model->weights.token_embedding_table,
                       kPromptTokens.data(), (int64_t)kPromptTokens.size());
    Tensor out;
    for (int64_t p = 0; p < (int64_t)kPromptTokens.size(); ++p) {
        out = model->AttentionBlockWithKVCache(Slice(x, 1, p, 1), 0, keyCache, valueCache, p);
        if (!implemented(out)) SKIP_TEST() << "not implemented yet";
    }

    // the last token must match the Chapter 2 golden values, which the full
    // (uncached) AttentionBlock produced for the same prompt
    ASSERT_EQ(out.elementCounts[0], model->config.dim);
    ASSERT_EQ(out.elementCounts[1], 1);
    EXPECT_NEAR(out.At(0, 0),  0.045301f, kTol);
    EXPECT_NEAR(out.At(1, 0), -0.023480f, kTol);
    EXPECT_NEAR(out.At(2, 0), -0.001121f, kTol);
    EXPECT_NEAR(out.At(3, 0),  0.006605f, kTol);
}

// Task 3
TEST(KVCache, ForwardWithKVCache) {
    Model* model = TheModel();
    if (!model) SKIP_TEST() << "stories15M.bin not found, run models/download.py";
    if (!Task1Ready()) SKIP_TEST() << "needs MultiHeadAttentionWithKVCache (Task 1) first";

    const int kvDim = (model->config.dim / model->config.n_heads) * model->config.n_kv_heads;
    Tensor keyCache(kvDim, model->config.seq_len, model->config.n_layers);
    Tensor valueCache(kvDim, model->config.seq_len, model->config.n_layers);

    // probe Task 2 with a throwaway call so a missing block skips instead of crashing
    {
        Tensor x0 = GetRows(model->weights.token_embedding_table, kPromptTokens.data(), 1);
        if (!implemented(model->AttentionBlockWithKVCache(x0, 0, keyCache, valueCache, 0)))
            SKIP_TEST() << "needs AttentionBlockWithKVCache (Task 2) first";
    }

    Tensor logits;
    for (int64_t p = 0; p < (int64_t)kPromptTokens.size(); ++p) {
        logits = model->ForwardWithKVCache(kPromptTokens[p], p, keyCache, valueCache);
        if (!implemented(logits)) SKIP_TEST() << "not implemented yet";
    }

    // after the whole prompt, the logits must match the Chapter 2 golden values
    // that the full Forward produced for the same prompt
    ASSERT_EQ(logits.elementCounts[0], model->config.vocab_size);
    ASSERT_EQ(logits.elementCounts[1], 1);
    EXPECT_NEAR(logits.At(0, 0), -10.420861f, kLogitTol);
    EXPECT_NEAR(logits.At(1, 0),   0.461928f, kLogitTol);
    EXPECT_NEAR(logits.At(2, 0), -10.420829f, kLogitTol);
    EXPECT_NEAR(logits.At(3, 0), -10.420976f, kLogitTol);
    ASSERT_EQ(basicllm::ArgMax(logits, 0), 29892); // still predicts ","
}

// Task 4
TEST(KVCache, GenerateWithKVCache) {
    Model* model = TheModel();
    if (!model) SKIP_TEST() << "stories15M.bin not found, run models/download.py";
    if (!Task1Ready()) SKIP_TEST() << "needs MultiHeadAttentionWithKVCache (Task 1) first";

    // probe Task 3 with a throwaway call
    {
        const int kvDim = (model->config.dim / model->config.n_heads) * model->config.n_kv_heads;
        Tensor kc(kvDim, model->config.seq_len, model->config.n_layers);
        Tensor vc(kvDim, model->config.seq_len, model->config.n_layers);
        if (!implemented(model->ForwardWithKVCache(kPromptTokens[0], 0, kc, vc)))
            SKIP_TEST() << "needs ForwardWithKVCache (Task 3) first";
    }

    std::mt19937 rng(42);
    const int newTokens = 8;

    // the naive path (Chapter 3, given), timed
    auto naiveStart = std::chrono::high_resolution_clock::now();
    std::vector<int> naive = model->Generate(kPromptTokens, newTokens, 0.0f, rng);
    auto naiveEnd = std::chrono::high_resolution_clock::now();

    // your cached path, timed
    std::vector<int> streamed;
    auto cachedStart = std::chrono::high_resolution_clock::now();
    std::vector<int> cached = model->GenerateWithKVCache(
        kPromptTokens, newTokens, 0.0f, rng,
        [&](int token) { streamed.push_back(token); });
    auto cachedEnd = std::chrono::high_resolution_clock::now();
    if (cached.empty()) SKIP_TEST() << "not implemented yet";

    // greedy is deterministic: both paths must tell the exact same story
    const std::vector<int> expected = {29892, 727, 471, 263, 2217, 7826, 4257, 365};
    ASSERT_EQ(cached.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        ASSERT_EQ(cached[i], expected[i]);
    ASSERT_EQ(naive.size(), cached.size());
    for (size_t i = 0; i < cached.size(); ++i)
        ASSERT_EQ(naive[i], cached[i]);
    ASSERT_EQ(streamed.size(), cached.size());
    for (size_t i = 0; i < cached.size(); ++i)
        ASSERT_EQ(streamed[i], cached[i]);

    const double naiveMs  = std::chrono::duration<double, std::milli>(naiveEnd - naiveStart).count();
    const double cachedMs = std::chrono::duration<double, std::milli>(cachedEnd - cachedStart).count();
    std::printf("    same %d tokens: naive %.0f ms, kv-cache %.0f ms (%.1fx)\n",
                newTokens, naiveMs, cachedMs, naiveMs / cachedMs);
}

struct TestCase {
    const char* name;
    void (*run)(TestContext&);
};

const TestCase kTests[] = {
    {"KVCache.MultiHeadAttentionWithKVCache", KVCache_MultiHeadAttentionWithKVCache},
    {"KVCache.AttentionBlockWithKVCache",     KVCache_AttentionBlockWithKVCache},
    {"KVCache.ForwardWithKVCache",            KVCache_ForwardWithKVCache},
    {"KVCache.GenerateWithKVCache",           KVCache_GenerateWithKVCache},
};

}  // namespace

int main() {
    int passed = 0, failed = 0, skipped = 0;

    for (const TestCase& tc : kTests) {
        TestContext ctx;
        tc.run(ctx);

        const char* status = ctx.skipped ? "[ SKIPPED  ]"
                           : ctx.failed  ? "[  FAILED  ]"
                                         : "[       OK ]";
        std::printf("%s %s\n", status, tc.name);

        if (ctx.skipped)     ++skipped;
        else if (ctx.failed) ++failed;
        else                 ++passed;
    }

    std::printf("\n%d passed, %d failed, %d skipped\n", passed, failed, skipped);
    return failed == 0 ? 0 : 1;
}
