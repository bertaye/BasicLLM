#pragma once

#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

#include "Tensor.h"

// Chapter 2: Tensor operations
//
// The first part of this file is the Chapter 1 ops, COMPLETED. They are given so
// this chapter builds on its own; if you have not done Chapter 1 yet, go do it
// first, these are the answers.
//
// Your tasks are the five new ops at the bottom (Slice, GetRows, Project,
// ApplyRoPE, MultiHeadAttention). Same rules as Chapter 1: the exact math is in
// the comment above each function, the body is empty, and a `return {};`
// placeholder keeps the file compiling until you implement it

namespace basicllm {

// Chapter 1 ops, completed. Read them if you want a refresher; the conventions
// have not changed: axis 0 is the column axis, axis 1 is the row axis, and
// A.At(column, row) reads one element

// Transpose: swap the column axis and the row axis. Zero-copy view
inline Tensor Transpose(const Tensor& A) {
    Tensor T = A;
    std::swap(T.elementCounts[0], T.elementCounts[1]);
    std::swap(T.byteStrides[0],   T.byteStrides[1]);
    return T;
}

// Scale: result[column, row] = A[column, row] * scalar
inline Tensor Scale(const Tensor& A, float scalar) {
    const int64_t cols = A.elementCounts[0];
    const int64_t rows = A.elementCounts[1];

    Tensor out(cols, rows);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            out.At(c, r) = A.At(c, r) * scalar;
        }
    }
    return out;
}

// Add: result[column, row] = A[column, row] + B[column, row]
inline Tensor Add(const Tensor& A, const Tensor& B) {
    const int64_t cols = A.elementCounts[0];
    const int64_t rows = A.elementCounts[1];
    assert(B.elementCounts[0] == cols && B.elementCounts[1] == rows);

    Tensor out(cols, rows);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            out.At(c, r) = A.At(c, r) + B.At(c, r);
        }
    }
    return out;
}

// Mul: result[column, row] = A[column, row] * B[column, row]
inline Tensor Mul(const Tensor& A, const Tensor& B) {
    const int64_t cols = A.elementCounts[0];
    const int64_t rows = A.elementCounts[1];
    assert(B.elementCounts[0] == cols && B.elementCounts[1] == rows);

    Tensor out(cols, rows);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            out.At(c, r) = A.At(c, r) * B.At(c, r);
        }
    }
    return out;
}

// MatMul: A is {K, M}, B is {N, K}, result C is {N, M}
//   C[n, m] = sum over k of A[k, m] * B[n, k]
inline Tensor MatMul(const Tensor& A, const Tensor& B) {
    const int64_t K = A.elementCounts[0];
    const int64_t M = A.elementCounts[1];
    const int64_t N = B.elementCounts[0];
    assert(B.elementCounts[1] == K);

    Tensor C(N, M);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A.At(k, m) * B.At(n, k);
            }
            C.At(n, m) = sum;
        }
    }
    return C;
}

// Softmax: per row, subtract the row max for numerical safety, exponentiate,
// divide by the row sum
inline Tensor Softmax(const Tensor& A) {
    const int64_t cols = A.elementCounts[0];
    const int64_t rows = A.elementCounts[1];

    Tensor out(cols, rows);
    for (int64_t r = 0; r < rows; ++r) {
        float maxVal = A.At(0, r);
        for (int64_t c = 1; c < cols; ++c) {
            if (A.At(c, r) > maxVal) {
                maxVal = A.At(c, r);
            }
        }
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            const float e = std::exp(A.At(c, r) - maxVal);
            out.At(c, r) = e;
            sum += e;
        }
        for (int64_t c = 0; c < cols; ++c) {
            out.At(c, r) /= sum;
        }
    }
    return out;
}

// RMSNorm: per row; scale = 1 / sqrt(meanSquare + eps), then multiply by the
// per-column weight
inline Tensor RMSNorm(const Tensor& X, const Tensor& weight, float eps = 1e-5f) {
    const int64_t cols = X.elementCounts[0];
    const int64_t rows = X.elementCounts[1];
    assert(weight.elementCounts[0] == cols);

    Tensor out(cols, rows);
    for (int64_t r = 0; r < rows; ++r) {
        float sumSq = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            const float v = X.At(c, r);
            sumSq += v * v;
        }
        const float invRms = 1.0f / std::sqrt(sumSq / (float)cols + eps);
        for (int64_t c = 0; c < cols; ++c) {
            out.At(c, r) = X.At(c, r) * invRms * weight.At(c);
        }
    }
    return out;
}

// SiLU: result = x * sigmoid(x), applied to every element
inline Tensor SiLU(const Tensor& X) {
    const int64_t cols = X.elementCounts[0];
    const int64_t rows = X.elementCounts[1];

    Tensor out(cols, rows);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            out.At(c, r) = X.At(c, r) * (1.0f / (1.0f + std::exp(-X.At(c, r))));
        }
    }
    return out;
}

// RoPE: rotate feature pairs by an angle that depends on the token's position.
// X is {headDim, seqLen}; token at row r has position (positionOffset + r)
inline Tensor RoPE(const Tensor& X, float base = 10000.0f, int64_t positionOffset = 0) {
    const int64_t headDim = X.elementCounts[0];
    const int64_t seqLen  = X.elementCounts[1];
    assert(headDim % 2 == 0);

    const int64_t numPairs = headDim / 2;
    Tensor out(headDim, seqLen);
    for (int64_t r = 0; r < seqLen; ++r) {
        const int64_t position = positionOffset + r;
        for (int64_t i = 0; i < numPairs; ++i) {
            const float freq  = std::pow(base, -2.0f * (float)i / (float)headDim);
            const float angle = (float)position * freq;
            const float cosA  = std::cos(angle);
            const float sinA  = std::sin(angle);

            const float pair0 = X.At(2 * i, r);
            const float pair1 = X.At(2 * i + 1, r);
            out.At(2 * i,     r) = pair0 * cosA - pair1 * sinA;
            out.At(2 * i + 1, r) = pair0 * sinA + pair1 * cosA;
        }
    }
    return out;
}

// MaskCausal: for every query row r, set every key column c with c > r to
// negative infinity. Modifies `scores` in place
inline void MaskCausal(Tensor& scores) {
    const int64_t cols = scores.elementCounts[0];
    const int64_t rows = scores.elementCounts[1];
    const float negInf = -std::numeric_limits<float>::infinity();
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = r + 1; c < cols; ++c) {
            scores.At(c, r) = negInf;
        }
    }
}

// Attention (single head): softmax(Q * K^T / sqrt(headDim)) * V
// Q is {headDim, seqQ}, K is {headDim, seqK}, V is {headDimV, seqK};
// output is {headDimV, seqQ}
inline Tensor Attention(const Tensor& Q, const Tensor& K, const Tensor& V, bool causal = false) {
    const int64_t headDim = Q.elementCounts[0];

    Tensor scores = MatMul(Q, Transpose(K));
    scores = Scale(scores, 1.0f / std::sqrt((float)headDim));
    if (causal) {
        MaskCausal(scores);
    }
    Tensor weights = Softmax(scores);
    return MatMul(weights, V);
}

// End of the Chapter 1 ops. Your Chapter 2 tasks start here

// Slice: keep only a range of ONE axis. A zero-copy view, exactly like Transpose:
// no floats are copied, we only edit the description of the memory.
//
//   result = the elements [start, start + length) along axis `dim`;
//            every other axis is unchanged
//
// Two edits produce the view:
//   1. point `data` at the first kept element:
//        data = X.data + start * X.byteStrides[dim]
//   2. shrink that axis's count to `length`
// The strides stay EXACTLY as they were. That is what makes it work: stepping
// through the view uses the original memory layout, we just start somewhere
// else and stop earlier
inline Tensor Slice(const Tensor& X, int dim, int64_t start, int64_t length) {
    // TODO: implement (Chapter 2, Task 1)
    return {};
}

// GetRows: the embedding lookup. `table` is {embedDim, vocabSize}: row t holds
// the embedding vector of token id t. Given a sequence of token ids, gather the
// matching rows into a fresh tensor.
//
//   result[feature, s] = table[feature, tokenIds[s]]
//
// Shape: result is {embedDim, seqLen}, one column-run of features per token in
// the sequence
inline Tensor GetRows(const Tensor& table, const int* tokenIds, int64_t seqLen) {
    // TODO: implement (Chapter 2, Task 2)
    return {};
}

// Project: multiply the input by ONE layer's weight matrix, picked out of a
// stacked weight tensor.
//
// W is {inFeatures, outFeatures, numLayers}: axis 2 stacks one matrix per layer.
// x is {inFeatures, seqLen}: one column-run of features per token.
//
//   step 1: Slice layer `layer` out of W along axis 2   -> {inFeatures, outFeatures}
//   step 2: Transpose it                                -> {outFeatures, inFeatures}
//   step 3: MatMul(x, that)                             -> {outFeatures, seqLen}
//
// Why the Transpose? MatMul(A, B) wants B as {N, K} with K matching A's axis 0.
// Here K = inFeatures and N = outFeatures, so B must be {outFeatures, inFeatures},
// which is the sliced matrix flipped. Both Slice and Transpose are free views,
// so the only real work is the MatMul
inline Tensor Project(const Tensor& x, const Tensor& W, int64_t layer) {
    // TODO: implement (Chapter 2, Task 3)
    return {};
}

// ApplyRoPE: apply RoPE per head. X is {dim, seqLen} with dim = numHeads * headDim.
//
// The model's q and k vectors hold ALL heads side by side: features
// [0, headDim) belong to head 0, [headDim, 2*headDim) to head 1, and so on.
// RoPE's frequency schedule is defined over ONE head, so it must restart at
// every head boundary. Rotating the whole dim-wide vector in one call is the
// classic mistake: the frequencies would keep decaying across head boundaries
// and every head after the first gets the wrong angles.
//
//   for each head h:
//     head    = Slice(X, 0, h * headDim, headDim)          a {headDim, seqLen} view
//     rotated = RoPE(head, base, positionOffset)
//     copy rotated into result at column offset h * headDim
inline Tensor ApplyRoPE(const Tensor& X, int numHeads, float base = 10000.0f, int64_t positionOffset = 0) {
    // TODO: implement (Chapter 2, Task 4)
    return {};
}

// MultiHeadAttention: split the features into numHeads heads, run the Chapter 1
// Attention on each head independently, and write the results back side by side.
//
// Q, K, V are {embedDim, seqLen} with embedDim = numHeads * headDim.
//
//   for each head h:
//     Qh = Slice(Q, 0, h * headDim, headDim)      and the same for Kh, Vh
//     Oh = Attention(Qh, Kh, Vh, causal)          {headDim, seqLen}
//     copy Oh into result at column offset h * headDim
//
// Shape: result is {embedDim, seqLen}, same as the inputs. Note the slices are
// zero-copy views, so splitting into heads moves no data at all
inline Tensor MultiHeadAttention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                 int numHeads, bool causal = false) {
    // TODO: implement (Chapter 2, Task 5)
    return {};
}

}  // namespace basicllm
