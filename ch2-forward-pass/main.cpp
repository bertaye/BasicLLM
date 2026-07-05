// Chapter 2 test runner
//
// Implement the ops in TensorOps.h (Tasks 1-5), then the blocks in Model.h
// (Tasks 6-8), and build and run after each one. Reporting works like Chapter 1:
//   [       OK ]  your implementation matches the expected result
//   [  FAILED  ]  it ran but produced the wrong numbers
//   [ SKIPPED  ]  still an empty stub, or an earlier task it needs is missing
//
// Tasks 6-8 run against the real stories15M weights and compare your numbers to
// values computed by the reference implementation. If the model file is missing,
// they skip; fetch it with `python models/download.py` from the repository root

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "Tensor.h"
#include "TensorOps.h"
#include "Model.h"

#ifndef MODEL_BIN_PATH
#define MODEL_BIN_PATH "../models/stories15M.bin"
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
constexpr float kLogitTol = 1e-2f; // logits are sums over dim=288, allow a bit more slack

// "Once upon a time" encoded by the reference tokenizer, BOS token (1) prepended.
// Turning text into ids is the tokenizer's job (header/Tokenizer.h on master,
// short and commented); here they are just input
const std::vector<int> kPromptTokens = {1, 9038, 2501, 263, 931};

// The stories15M model, loaded once and shared by Tasks 6-8. Returns nullptr if
// the model file is not there
Model* TheModel() {
    static std::unique_ptr<Model> model = [] {
        if (!std::filesystem::exists(MODEL_BIN_PATH)) return std::unique_ptr<Model>();
        return std::make_unique<Model>(MODEL_BIN_PATH);
    }();
    return model.get();
}

// Task 1
TEST(ForwardPass, Slice) {
    Tensor A(4, 3); // 4 columns, 3 rows
    for (int64_t r = 0; r < 3; ++r)
        for (int64_t c = 0; c < 4; ++c)
            A.At(c, r) = static_cast<float>(c + 10 * r); // a unique value per cell

    Tensor S = Slice(A, 0, 1, 2); // keep columns 1 and 2
    if (!implemented(S)) SKIP_TEST() << "not implemented yet";

    ASSERT_EQ(S.elementCounts[0], 2);
    ASSERT_EQ(S.elementCounts[1], 3);
    for (int64_t r = 0; r < 3; ++r)
        for (int64_t c = 0; c < 2; ++c)
            EXPECT_FLOAT_EQ(S.At(c, r), A.At(c + 1, r)); // shifted by `start`

    // zero-copy check: writing through the original must show up in the view
    A.At(1, 0) = 99.0f;
    EXPECT_FLOAT_EQ(S.At(0, 0), 99.0f);

    Tensor R = Slice(A, 1, 2, 1); // keep only row 2
    ASSERT_EQ(R.elementCounts[0], 4);
    ASSERT_EQ(R.elementCounts[1], 1);
    for (int64_t c = 0; c < 4; ++c)
        EXPECT_FLOAT_EQ(R.At(c, 0), A.At(c, 2));
}

// Task 2
TEST(ForwardPass, GetRows) {
    Tensor table(3, 4); // embedDim=3, vocabSize=4
    for (int64_t t = 0; t < 4; ++t)
        for (int64_t f = 0; f < 3; ++f)
            table.At(f, t) = static_cast<float>(10 * t + f); // token t, feature f

    const int ids[2] = {2, 0};
    Tensor E = GetRows(table, ids, 2);
    if (!implemented(E)) SKIP_TEST() << "not implemented yet";

    ASSERT_EQ(E.elementCounts[0], 3);
    ASSERT_EQ(E.elementCounts[1], 2);
    for (int64_t f = 0; f < 3; ++f) {
        EXPECT_FLOAT_EQ(E.At(f, 0), table.At(f, 2)); // first id is token 2
        EXPECT_FLOAT_EQ(E.At(f, 1), table.At(f, 0)); // second id is token 0
    }
}

// Task 3
TEST(ForwardPass, Project) {
    // W stacks 2 layers of a {in=2, out=2} matrix; layer 0 is filled with junk
    // so the test only passes if you slice out the RIGHT layer
    Tensor W(2, 2, 2);
    for (int64_t o = 0; o < 2; ++o)
        for (int64_t i = 0; i < 2; ++i)
            W.At(i, o, 0) = 100.0f; // layer 0: junk
    W.At(0, 0, 1) = 1; W.At(1, 0, 1) = 2; // layer 1, output feature 0
    W.At(0, 1, 1) = 3; W.At(1, 1, 1) = 4; // layer 1, output feature 1

    Tensor x(2, 1); // one token with features [5, 6]
    x.At(0, 0) = 5; x.At(1, 0) = 6;

    Tensor R = Project(x, W, 1);
    if (!implemented(R)) SKIP_TEST() << "not implemented yet";

    ASSERT_EQ(R.elementCounts[0], 2);
    ASSERT_EQ(R.elementCounts[1], 1);
    // out[o] = sum over i of x[i] * W[i, o, layer]
    EXPECT_NEAR(R.At(0, 0), 5 * 1 + 6 * 2, kTol); // 17
    EXPECT_NEAR(R.At(1, 0), 5 * 3 + 6 * 4, kTol); // 39
}

// Task 4
TEST(ForwardPass, ApplyRoPE) {
    // 2 heads of headDim=2, seqLen=2. Head 0 reuses the Chapter 1 RoPE test
    // values; head 1 must be rotated by the SAME angles (the schedule restarts)
    Tensor X(4, 2);
    X.At(0, 0) = 1; X.At(1, 0) = 2; X.At(2, 0) = 3; X.At(3, 0) = 4; // position 0
    X.At(0, 1) = 1; X.At(1, 1) = 0; X.At(2, 1) = 0; X.At(3, 1) = 1; // position 1

    Tensor R = ApplyRoPE(X, /*numHeads=*/2);
    if (!implemented(R)) SKIP_TEST() << "not implemented yet";

    ASSERT_EQ(R.elementCounts[0], 4);
    ASSERT_EQ(R.elementCounts[1], 2);
    // position 0: angle 0, both heads unchanged
    EXPECT_NEAR(R.At(0, 0), 1.0f, kTol);
    EXPECT_NEAR(R.At(1, 0), 2.0f, kTol);
    EXPECT_NEAR(R.At(2, 0), 3.0f, kTol);
    EXPECT_NEAR(R.At(3, 0), 4.0f, kTol);
    // position 1: both heads rotate by 1 rad (frequency schedule restarted)
    // head 0: (1,0) -> [cos(1), sin(1)]
    EXPECT_NEAR(R.At(0, 1),  0.54030f, kTol);
    EXPECT_NEAR(R.At(1, 1),  0.84147f, kTol);
    // head 1: (0,1) -> [-sin(1), cos(1)]
    EXPECT_NEAR(R.At(2, 1), -0.84147f, kTol);
    EXPECT_NEAR(R.At(3, 1),  0.54030f, kTol);
}

// Task 5
TEST(ForwardPass, MultiHeadAttention) {
    // 2 heads of headDim=1, seqLen=2. Each head is exactly the Chapter 1
    // Attention test, with head 1's values scaled by 10
    Tensor Q(2, 2), K(2, 2), V(2, 2);
    Q.At(0, 0) = 1; Q.At(0, 1) = 1;  K.At(0, 0) = 1; K.At(0, 1) = 2;  V.At(0, 0) = 10;  V.At(0, 1) = 20;   // head 0
    Q.At(1, 0) = 1; Q.At(1, 1) = 1;  K.At(1, 0) = 1; K.At(1, 1) = 2;  V.At(1, 0) = 100; V.At(1, 1) = 200;  // head 1

    Tensor out = MultiHeadAttention(Q, K, V, /*numHeads=*/2, /*causal=*/false);
    if (!implemented(out)) SKIP_TEST() << "not implemented yet";

    ASSERT_EQ(out.elementCounts[0], 2);
    ASSERT_EQ(out.elementCounts[1], 2);
    // per head: scores [1, 2] -> softmax [0.2689, 0.7311] -> weighted V
    EXPECT_NEAR(out.At(0, 0),  17.311f, kTol);
    EXPECT_NEAR(out.At(0, 1),  17.311f, kTol);
    EXPECT_NEAR(out.At(1, 0), 173.11f,  kTol * 10);
    EXPECT_NEAR(out.At(1, 1), 173.11f,  kTol * 10);

    // causal: position 0 may only see key 0, so it returns V's first column
    Tensor causalOut = MultiHeadAttention(Q, K, V, /*numHeads=*/2, /*causal=*/true);
    EXPECT_NEAR(causalOut.At(0, 0),  10.0f, kTol);
    EXPECT_NEAR(causalOut.At(1, 0), 100.0f, kTol);
    EXPECT_NEAR(causalOut.At(0, 1),  17.311f, kTol); // position 1 sees everything
}

// Task 6. From here on we are on the real model: your numbers are compared with
// the reference implementation running the same weights
TEST(ForwardPass, AttentionBlock) {
    Model* model = TheModel();
    if (!model) SKIP_TEST() << "stories15M.bin not found, run models/download.py";

    Tensor x = GetRows(model->weights.token_embedding_table,
                       kPromptTokens.data(), (int64_t)kPromptTokens.size());
    if (!implemented(x)) SKIP_TEST() << "needs GetRows (Task 2) first";

    Tensor out = model->AttentionBlock(x, 0);
    if (!implemented(out)) SKIP_TEST() << "not implemented yet";

    ASSERT_EQ(out.elementCounts[0], model->config.dim);
    ASSERT_EQ(out.elementCounts[1], (int64_t)kPromptTokens.size());
    // reference values of the last token's first four features after layer 0's
    // attention block
    const int64_t last = (int64_t)kPromptTokens.size() - 1;
    EXPECT_NEAR(out.At(0, last),  0.045301f, kTol);
    EXPECT_NEAR(out.At(1, last), -0.023480f, kTol);
    EXPECT_NEAR(out.At(2, last), -0.001121f, kTol);
    EXPECT_NEAR(out.At(3, last),  0.006605f, kTol);
}

// Task 7
TEST(ForwardPass, FeedForwardBlock) {
    Model* model = TheModel();
    if (!model) SKIP_TEST() << "stories15M.bin not found, run models/download.py";

    Tensor x = GetRows(model->weights.token_embedding_table,
                       kPromptTokens.data(), (int64_t)kPromptTokens.size());
    if (!implemented(x)) SKIP_TEST() << "needs GetRows (Task 2) first";

    Tensor afterAttention = model->AttentionBlock(x, 0);
    if (!implemented(afterAttention)) SKIP_TEST() << "needs AttentionBlock (Task 6) first";

    Tensor out = model->FeedForwardBlock(afterAttention, 0);
    if (!implemented(out)) SKIP_TEST() << "not implemented yet";

    ASSERT_EQ(out.elementCounts[0], model->config.dim);
    ASSERT_EQ(out.elementCounts[1], (int64_t)kPromptTokens.size());
    // reference values of the last token's first four features after the whole
    // of layer 0 (attention block + feed-forward block)
    const int64_t last = (int64_t)kPromptTokens.size() - 1;
    EXPECT_NEAR(out.At(0, last),  0.102448f, kTol);
    EXPECT_NEAR(out.At(1, last), -0.158479f, kTol);
    EXPECT_NEAR(out.At(2, last), -0.004494f, kTol);
    EXPECT_NEAR(out.At(3, last),  0.003362f, kTol);
}

// Task 8
TEST(ForwardPass, Forward) {
    Model* model = TheModel();
    if (!model) SKIP_TEST() << "stories15M.bin not found, run models/download.py";

    Tensor logits = model->Forward(kPromptTokens);
    if (!implemented(logits)) SKIP_TEST() << "not implemented yet";

    ASSERT_EQ(logits.elementCounts[0], model->config.vocab_size);
    ASSERT_EQ(logits.elementCounts[1], (int64_t)kPromptTokens.size());

    const int64_t last = (int64_t)kPromptTokens.size() - 1;
    EXPECT_NEAR(logits.At(0, last), -10.420861f, kLogitTol);
    EXPECT_NEAR(logits.At(1, last),   0.461928f, kLogitTol);
    EXPECT_NEAR(logits.At(2, last), -10.420829f, kLogitTol);
    EXPECT_NEAR(logits.At(3, last), -10.420976f, kLogitTol);

    // the punchline: the highest logit after "Once upon a time" must be token
    // 29892, which decodes to "," (the model wants to continue the sentence)
    int64_t best = 0;
    for (int64_t v = 1; v < model->config.vocab_size; ++v)
        if (logits.At(v, last) > logits.At(best, last)) best = v;
    ASSERT_EQ(best, 29892);
    std::printf("    the model read \"Once upon a time\" and predicted \",\"\n");
}

struct TestCase {
    const char* name;
    void (*run)(TestContext&);
};

const TestCase kTests[] = {
    {"ForwardPass.Slice",              ForwardPass_Slice},
    {"ForwardPass.GetRows",            ForwardPass_GetRows},
    {"ForwardPass.Project",            ForwardPass_Project},
    {"ForwardPass.ApplyRoPE",          ForwardPass_ApplyRoPE},
    {"ForwardPass.MultiHeadAttention", ForwardPass_MultiHeadAttention},
    {"ForwardPass.AttentionBlock",     ForwardPass_AttentionBlock},
    {"ForwardPass.FeedForwardBlock",   ForwardPass_FeedForwardBlock},
    {"ForwardPass.Forward",            ForwardPass_Forward},
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
