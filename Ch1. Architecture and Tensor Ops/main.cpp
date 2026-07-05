// Chapter 1 test runner
//
// Implement the functions in TensorOps.h one at a time, then build and run.
// A test shows as:
//   [       OK ]  your implementation matches the expected result
//   [  FAILED  ]  it ran but produced the wrong numbers
//   [ SKIPPED  ]  still an empty stub (returns an empty tensor)
//
// The goal is to turn every SKIPPED/FAILED into OK

#include <cmath>
#include <cstdio>

#include "Tensor.h"
#include "TensorOps.h"

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

constexpr float kTol = 1e-3f;

// Task 1
TEST(TensorOps, Transpose) {
    Tensor A(2, 3); // 2 columns, 3 rows
    for (int64_t r = 0; r < 3; ++r)
        for (int64_t c = 0; c < 2; ++c)
            A.At(c, r) = static_cast<float>(c + 10 * r); // a unique value per cell

    Tensor T = Transpose(A);
    if (!implemented(T)) SKIP_TEST() << "not implemented yet";

    ASSERT_EQ(T.elementCounts[0], 3);
    ASSERT_EQ(T.elementCounts[1], 2);
    for (int64_t i0 = 0; i0 < 3; ++i0)
        for (int64_t i1 = 0; i1 < 2; ++i1)
            EXPECT_FLOAT_EQ(T.At(i0, i1), A.At(i1, i0)); // T.At(i,j) reads A.At(j,i)
}

// Task 2
TEST(TensorOps, Scale) {
    Tensor A(2, 2);
    A.At(0, 0) = 1; A.At(1, 0) = 2; A.At(0, 1) = 3; A.At(1, 1) = 4;

    Tensor R = Scale(A, 2.0f);
    if (!implemented(R)) SKIP_TEST() << "not implemented yet";
    EXPECT_NEAR(R.At(0, 0), 2, kTol);
    EXPECT_NEAR(R.At(1, 0), 4, kTol);
    EXPECT_NEAR(R.At(0, 1), 6, kTol);
    EXPECT_NEAR(R.At(1, 1), 8, kTol);
}

// Task 3
TEST(TensorOps, Add) {
    Tensor A(2, 1), B(2, 1);
    A.At(0, 0) = 1;  A.At(1, 0) = 2;
    B.At(0, 0) = 10; B.At(1, 0) = 20;

    Tensor R = Add(A, B);
    if (!implemented(R)) SKIP_TEST() << "not implemented yet";
    EXPECT_NEAR(R.At(0, 0), 11, kTol);
    EXPECT_NEAR(R.At(1, 0), 22, kTol);
}

// Task 4
TEST(TensorOps, Mul) {
    Tensor A(2, 1), B(2, 1);
    A.At(0, 0) = 1;  A.At(1, 0) = 2;
    B.At(0, 0) = 10; B.At(1, 0) = 20;

    Tensor R = Mul(A, B);
    if (!implemented(R)) SKIP_TEST() << "not implemented yet";
    EXPECT_NEAR(R.At(0, 0), 10, kTol);
    EXPECT_NEAR(R.At(1, 0), 40, kTol);
}

// Task 5
TEST(TensorOps, MatMul) {
    Tensor A(2, 2); // {K=2, M=2}
    A.At(0, 0) = 1; A.At(1, 0) = 2;   // row m=0
    A.At(0, 1) = 3; A.At(1, 1) = 4;   // row m=1
    Tensor B(2, 2); // {N=2, K=2}
    B.At(0, 0) = 5; B.At(0, 1) = 6;   // n=0, over k
    B.At(1, 0) = 7; B.At(1, 1) = 8;   // n=1, over k

    Tensor C = MatMul(A, B); // expected {N=2, M=2}
    if (!implemented(C)) SKIP_TEST() << "not implemented yet";
    ASSERT_EQ(C.elementCounts[0], 2);
    ASSERT_EQ(C.elementCounts[1], 2);
    // C[n,m] = sum_k A[k,m] * B[n,k]
    EXPECT_NEAR(C.At(0, 0), 17, kTol);
    EXPECT_NEAR(C.At(1, 0), 23, kTol);
    EXPECT_NEAR(C.At(0, 1), 39, kTol);
    EXPECT_NEAR(C.At(1, 1), 53, kTol);
}

// Task 6
TEST(TensorOps, Softmax) {
    Tensor A(3, 1);
    A.At(0, 0) = 1; A.At(1, 0) = 2; A.At(2, 0) = 3;

    Tensor R = Softmax(A);
    if (!implemented(R)) SKIP_TEST() << "not implemented yet";
    EXPECT_NEAR(R.At(0, 0), 0.09003f, kTol);
    EXPECT_NEAR(R.At(1, 0), 0.24473f, kTol);
    EXPECT_NEAR(R.At(2, 0), 0.66524f, kTol);
}

// Task 7
TEST(TensorOps, RMSNorm) {
    Tensor X(4, 1);
    X.At(0, 0) = 1; X.At(1, 0) = 2; X.At(2, 0) = 3; X.At(3, 0) = 4;
    Tensor weight(4); // one weight per column
    for (int64_t c = 0; c < 4; ++c) weight.At(c) = 1.0f;

    Tensor R = RMSNorm(X, weight);
    if (!implemented(R)) SKIP_TEST() << "not implemented yet";
    // meanSquare = 30/4 = 7.5 ; scale = 1/sqrt(7.5) = 0.365148
    EXPECT_NEAR(R.At(0, 0), 0.36515f, kTol);
    EXPECT_NEAR(R.At(1, 0), 0.73030f, kTol);
    EXPECT_NEAR(R.At(2, 0), 1.09545f, kTol);
    EXPECT_NEAR(R.At(3, 0), 1.46059f, kTol);
}

// Task 8
TEST(TensorOps, SiLU) {
    Tensor X(3, 1);
    X.At(0, 0) = 0; X.At(1, 0) = 1; X.At(2, 0) = -1;

    Tensor R = SiLU(X);
    if (!implemented(R)) SKIP_TEST() << "not implemented yet";
    EXPECT_NEAR(R.At(0, 0), 0.0f, kTol);
    EXPECT_NEAR(R.At(1, 0), 0.73106f, kTol);
    EXPECT_NEAR(R.At(2, 0), -0.26894f, kTol);
}

// Task 9
TEST(TensorOps, RoPE) {
    Tensor X(2, 2); // headDim=2, seqLen=2
    X.At(0, 0) = 1; X.At(1, 0) = 2;   // token 0 (position 0)
    X.At(0, 1) = 1; X.At(1, 1) = 0;   // token 1 (position 1)

    Tensor R = RoPE(X); // base=10000, positionOffset=0
    if (!implemented(R)) SKIP_TEST() << "not implemented yet";
    // position 0: angle 0, so the pair is unchanged -> [1, 2]
    // position 1: angle 1 rad -> rotate (1,0) -> [cos(1), sin(1)] = [0.5403, 0.8415]
    EXPECT_NEAR(R.At(0, 0), 1.0f, kTol);
    EXPECT_NEAR(R.At(1, 0), 2.0f, kTol);
    EXPECT_NEAR(R.At(0, 1), 0.54030f, kTol);
    EXPECT_NEAR(R.At(1, 1), 0.84147f, kTol);
}

// Task 10
TEST(TensorOps, MaskCausal) {
    Tensor s(2, 2); // {keys=2, queries=2}
    for (int64_t r = 0; r < 2; ++r)
        for (int64_t c = 0; c < 2; ++c)
            s.At(c, r) = 1.0f;

    MaskCausal(s);
    if (s.At(1, 0) == 1.0f) SKIP_TEST() << "not implemented yet"; // key 1 after query 0 is untouched
    EXPECT_TRUE(std::isinf(s.At(1, 0)) && s.At(1, 0) < 0); // masked to -inf
    EXPECT_NEAR(s.At(0, 0), 1.0f, kTol); // the rest are untouched
    EXPECT_NEAR(s.At(0, 1), 1.0f, kTol);
    EXPECT_NEAR(s.At(1, 1), 1.0f, kTol);
}

// Task 11
TEST(TensorOps, Attention) {
    Tensor Q(1, 2), K(1, 2), V(1, 2); // headDim=1, seqQ=seqK=2, headDimV=1
    Q.At(0, 0) = 1; Q.At(0, 1) = 1;
    K.At(0, 0) = 1; K.At(0, 1) = 2;
    V.At(0, 0) = 10; V.At(0, 1) = 20;

    Tensor out = Attention(Q, K, V, /*causal=*/false);
    if (!implemented(out)) SKIP_TEST() << "not implemented yet";
    ASSERT_EQ(out.elementCounts[0], 1);
    ASSERT_EQ(out.elementCounts[1], 2);
    // scores per query = [1, 2] -> softmax -> [0.2689, 0.7311]
    // output = 0.2689*10 + 0.7311*20 = 17.311 for each query
    EXPECT_NEAR(out.At(0, 0), 17.311f, kTol);
    EXPECT_NEAR(out.At(0, 1), 17.311f, kTol);
}

struct TestCase {
    const char* name;
    void (*run)(TestContext&);
};

const TestCase kTests[] = {
    {"TensorOps.Transpose",  TensorOps_Transpose},
    {"TensorOps.Scale",      TensorOps_Scale},
    {"TensorOps.Add",        TensorOps_Add},
    {"TensorOps.Mul",        TensorOps_Mul},
    {"TensorOps.MatMul",     TensorOps_MatMul},
    {"TensorOps.Softmax",    TensorOps_Softmax},
    {"TensorOps.RMSNorm",    TensorOps_RMSNorm},
    {"TensorOps.SiLU",       TensorOps_SiLU},
    {"TensorOps.RoPE",       TensorOps_RoPE},
    {"TensorOps.MaskCausal", TensorOps_MaskCausal},
    {"TensorOps.Attention",  TensorOps_Attention},
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
