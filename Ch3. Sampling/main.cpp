// Chapter 3 test runner
//
// Implement ArgMax and Sample in TensorOps.h (Tasks 1-2), then Generate in
// Model.h (Task 3), and build and run after each one. Reporting works like the
// earlier chapters:
//   [       OK ]  your implementation matches the expected result
//   [  FAILED  ]  it ran but produced the wrong numbers
//   [ SKIPPED  ]  still a stub, or an earlier task it needs is missing
//
// Task 3 runs the real stories15M weights and is compared against the exact
// token ids the reference implementation generates. If the model file is
// missing, it skips; fetch it with `python models/download.py` from the
// repository root

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

// ArgMax and Sample return -1 while unimplemented (no real token id is negative)
bool implemented(int64_t id) { return id >= 0; }

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

// Task 1
TEST(Sampling, ArgMax) {
    Tensor L(4, 2); // vocabSize=4, seqLen=2
    L.At(0, 0) = 1; L.At(1, 0) =  5; L.At(2, 0) = 2; L.At(3, 0) = 0; // max at 1
    L.At(0, 1) = 0; L.At(1, 1) = -1; L.At(2, 1) = 3; L.At(3, 1) = 2; // max at 2

    int64_t best0 = ArgMax(L, 0);
    if (!implemented(best0)) SKIP_TEST() << "not implemented yet";

    ASSERT_EQ(best0, 1);
    ASSERT_EQ(ArgMax(L, 1), 2); // `pos` must pick the column, not always column 0
}

// Task 2
TEST(Sampling, Sample) {
    // logits ln(1), ln(2), ln(4): softmax at temperature 1 gives exactly
    // probabilities 1/7, 2/7, 4/7
    Tensor L(3, 1);
    L.At(0, 0) = std::log(1.0f);
    L.At(1, 0) = std::log(2.0f);
    L.At(2, 0) = std::log(4.0f);

    std::mt19937 rng(42);
    if (!implemented(Sample(L, 0, 1.0f, rng))) SKIP_TEST() << "not implemented yet";
    if (!implemented(ArgMax(L, 0))) SKIP_TEST() << "needs ArgMax (Task 1) first";

    // temperature <= 0 must be exactly greedy
    ASSERT_EQ(Sample(L, 0, 0.0f, rng), ArgMax(L, 0));

    // temperature 1: draw many times, the empirical frequencies must match the
    // softmax probabilities (any correct sampler converges to these)
    const int draws = 20000;
    int counts[3] = {0, 0, 0};
    for (int i = 0; i < draws; ++i) {
        int64_t token = Sample(L, 0, 1.0f, rng);
        ASSERT_EQ(token >= 0 && token < 3, true);
        counts[token]++;
    }
    EXPECT_NEAR(counts[0] / (float)draws, 1.0f / 7.0f, 0.02f);
    EXPECT_NEAR(counts[1] / (float)draws, 2.0f / 7.0f, 0.02f);
    EXPECT_NEAR(counts[2] / (float)draws, 4.0f / 7.0f, 0.02f);

    // a very cold temperature concentrates (almost) all mass on the best token
    int bestCount = 0;
    for (int i = 0; i < 2000; ++i) {
        if (Sample(L, 0, 0.05f, rng) == 2) bestCount++;
    }
    EXPECT_TRUE(bestCount / 2000.0f > 0.99f);
}

// Task 3
TEST(Sampling, Generate) {
    Model* model = TheModel();
    if (!model) SKIP_TEST() << "stories15M.bin not found, run models/download.py";

    std::mt19937 rng(42);
    std::vector<int> streamed;
    std::vector<int> generated = model->Generate(
        kPromptTokens, /*maxNewTokens=*/8, /*temperature=*/0.0f, rng,
        [&](int token) { streamed.push_back(token); });
    if (generated.empty()) SKIP_TEST() << "not implemented yet (or Sample/ArgMax missing)";

    // greedy continuation of "Once upon a time", computed by the reference
    // implementation from the same weights
    const std::vector<int> expected = {29892, 727, 471, 263, 2217, 7826, 4257, 365};
    ASSERT_EQ(generated.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        ASSERT_EQ(generated[i], expected[i]);

    // the callback must stream exactly the tokens that are returned
    ASSERT_EQ(streamed.size(), generated.size());
    for (size_t i = 0; i < generated.size(); ++i)
        ASSERT_EQ(streamed[i], generated[i]);

    // show the payoff: decode the continuation
    if (std::filesystem::exists(TOKENIZER_BIN_PATH)) {
        Tokenizer tokenizer(TOKENIZER_BIN_PATH, model->config.vocab_size);
        std::string text;
        int prev = kPromptTokens.back();
        for (int token : generated) {
            text += tokenizer.Decode(prev, token);
            prev = token;
        }
        std::printf("    \"Once upon a time\" continues: \"%s\"\n", text.c_str());
    }
}

struct TestCase {
    const char* name;
    void (*run)(TestContext&);
};

const TestCase kTests[] = {
    {"Sampling.ArgMax",   Sampling_ArgMax},
    {"Sampling.Sample",   Sampling_Sample},
    {"Sampling.Generate", Sampling_Generate},
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
