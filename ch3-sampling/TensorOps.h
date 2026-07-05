#pragma once

#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "Tensor.h"

// Chapter 3: Tensor operations
//
// The first part of this file is the Chapter 1 and Chapter 2 ops, COMPLETED.
// They are given so this chapter builds on its own; if you have not done those
// chapters yet, go do them first, these are the answers.
//
// Your tasks are the two new functions at the bottom: ArgMax and Sample. They
// return an int64_t instead of a Tensor, so the "not implemented" placeholder
// is `return -1;` (no real token id is negative)

namespace basicllm {

// Chapter 1 and 2 ops, completed. The conventions have not changed: axis 0 is
// the column axis, axis 1 the row axis, and A.At(column, row) reads one element

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

// Slice: keep only the elements [start, start + length) along axis `dim`.
// Zero-copy view: move the data pointer, shrink the count, keep the strides
inline Tensor Slice(const Tensor& X, int dim, int64_t start, int64_t length) {
    assert(dim >= 0 && dim < MAX_DIMS);
    assert(start >= 0 && start + length <= X.elementCounts[dim]);
    Tensor v = X;
    v.elementCounts[dim] = length;
    v.data = X.data + start * X.byteStrides[dim];
    return v;
}

// GetRows: the embedding lookup. result[feature, s] = table[feature, tokenIds[s]]
inline Tensor GetRows(const Tensor& table, const int* tokenIds, int64_t seqLen) {
    const int64_t embedDim = table.elementCounts[0];

    Tensor out(embedDim, seqLen);
    for (int64_t s = 0; s < seqLen; ++s) {
        for (int64_t f = 0; f < embedDim; ++f) {
            out.At(f, s) = table.At(f, tokenIds[s]);
        }
    }
    return out;
}

// Project: multiply the input by ONE layer's matrix out of a stacked weight tensor
inline Tensor Project(const Tensor& x, const Tensor& W, int64_t layer) {
    return MatMul(x, Transpose(Slice(W, 2, layer, 1)));
}

// ApplyRoPE: apply RoPE per head; the frequency schedule restarts at every head
inline Tensor ApplyRoPE(const Tensor& X, int numHeads, float base = 10000.0f, int64_t positionOffset = 0) {
    const int64_t dim     = X.elementCounts[0];
    const int64_t seqLen  = X.elementCounts[1];
    const int64_t headDim = dim / numHeads;
    assert(dim % numHeads == 0);

    Tensor out(dim, seqLen);
    for (int h = 0; h < numHeads; ++h) {
        const int64_t start = h * headDim;
        Tensor rotated = RoPE(Slice(X, 0, start, headDim), base, positionOffset);
        for (int64_t r = 0; r < seqLen; ++r) {
            for (int64_t c = 0; c < headDim; ++c) {
                out.At(start + c, r) = rotated.At(c, r);
            }
        }
    }
    return out;
}

// MultiHeadAttention: slice the features into heads, attend per head, concatenate
inline Tensor MultiHeadAttention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                 int numHeads, bool causal = false) {
    const int64_t embedDim = Q.elementCounts[0];
    const int64_t seqLen   = Q.elementCounts[1];
    const int64_t headDim  = embedDim / numHeads;
    assert(embedDim % numHeads == 0);

    Tensor out(embedDim, seqLen);
    for (int h = 0; h < numHeads; ++h) {
        const int64_t start = h * headDim;
        Tensor oh = Attention(Slice(Q, 0, start, headDim),
                              Slice(K, 0, start, headDim),
                              Slice(V, 0, start, headDim), causal);
        for (int64_t r = 0; r < seqLen; ++r) {
            for (int64_t c = 0; c < headDim; ++c) {
                out.At(start + c, r) = oh.At(c, r);
            }
        }
    }
    return out;
}

// End of the given ops. Your Chapter 3 tasks start here

// ArgMax: greedy sampling. Return the vocabulary index of the LARGEST logit at
// sequence position `pos`.
//
// logits is {vocabSize, seqLen}: column s scores every vocabulary token as the
// successor of position s. So this is a walk over logits.At(v, pos) for
// v in [0, vocabSize), tracking the index of the biggest value.
//
// The next-token prediction of the whole sequence sits at pos = seqLen - 1
inline int64_t ArgMax(const Tensor& logits, int64_t pos) {
    // TODO: implement (Chapter 3, Task 1)
    return -1;
}

// Sample: temperature sampling. Draw ONE random token from the distribution
// softmax(logits[:, pos] / temperature).
//
//   If temperature <= 0, fall back to ArgMax (greedy is the cold limit).
//
//   Otherwise:
//     1. m = the largest logit at `pos` (for numerical safety, like Softmax)
//     2. for every v: probs[v] = exp( (logits[v, pos] - m) / temperature ),
//        and sum them up. Note we do NOT divide by the sum
//     3. draw a random number r uniformly from [0, sum)
//     4. walk v from 0, adding probs[v] to a running total; the first v whose
//        running total exceeds r is the sampled token
//     5. if floating point rounding walks off the end, return the last token
//
//   Step 3 and 4 are inverse-CDF sampling: drawing r against the UNNORMALIZED
//   cumulative sum is the same as normalizing first, but skips a whole pass.
//   `rng` carries the random state across calls; use it via
//   std::uniform_real_distribution<float>
inline int64_t Sample(const Tensor& logits, int64_t pos, float temperature, std::mt19937& rng) {
    // TODO: implement (Chapter 3, Task 2)
    return -1;
}

}  // namespace basicllm
