#pragma once

#include <cassert>
#include <cmath>
#include <limits>
#include <utility>
#include <random>
#include <vector>

#include "Tensor.h"

namespace basicllm {

// Transpose: swap dims 0 and 1. Zero-copy view (shares bytes, swaps counts/strides).
inline Tensor Transpose(const Tensor& A) {
    Tensor T = A;
    std::swap(T.elementCounts[0], T.elementCounts[1]);
    std::swap(T.byteStrides[0],   T.byteStrides[1]);
    return T;
}

// out[r,c] = A[r,c] * s.
inline Tensor Scale(const Tensor& A, float s) {
    const int64_t cols = A.elementCounts[0];
    const int64_t rows = A.elementCounts[1];

    Tensor out(cols, rows);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            out.At(c, r) = A.At(c, r) * s;
        }
    }
    return out;
}

// C = A * B.  A: {K, M}, B: {N, K}, C: {N, M}.  C[m,n] = sum_k A[m,k] * B[k,n].
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

// Row-wise softmax: out[r,c] = exp(A[r,c] - max_c) / sum_c exp(A[r,c] - max_c).
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

// out[c,r] = x[c,r] / sqrt(sum_c x[c]^2 / N + eps) * weight[c]
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

// silu(x) = x · sigmoid(x) = x / (1 + exp(-x))
inline Tensor SiLU(const Tensor& X) {
    const int64_t cols = X.elementCounts[0];
    const int64_t rows = X.elementCounts[1];
    Tensor out(cols, rows);
    for (int64_t r = 0; r < rows; r++)
    {
        for (int64_t c = 0; c < cols; c++)
        {
            out.At(c, r) = X.At(c, r) * (1 / (1 + std::exp(-1 * X.At(c, r))));
        }
    }
    return out;
}

// out[c,r] = a[c,r] + b[c,r]
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

// out[c,r] = a[c,r] * b[c,r]
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


// Slice along `dim`: keep elements [start, start+length) on that axis. Zero-copy
// view; shares bytes and keeps the original strides, so the other dims still step
// over the full original extents (the resulting view may be non-contiguous).
// Slice is simply 'cutting' over the 'dim' for 'length' elements starting from 'start'.
inline Tensor Slice(const Tensor& X, int dim, int64_t start, int64_t length) {
    assert(dim >= 0 && dim < MAX_DIMS);
    assert(start >= 0 && start + length <= X.elementCounts[dim]);
    Tensor v = X;
    v.elementCounts[dim] = length;
    v.data = X.data + start * X.byteStrides[dim];
    return v;
}

inline Tensor RoPE(const Tensor& X, float base = 10000.0f, int64_t positionOffset = 0) {
    const int64_t headDim = X.elementCounts[0];
    const int64_t seqLen = X.elementCounts[1];
    assert(headDim % 2 == 0);

    const int64_t numPairs = headDim / 2;
    Tensor out(headDim, seqLen);
    for (int64_t r = 0; r < seqLen; ++r) {
        const int64_t position = positionOffset + r; //Token idx
        for (int64_t i = 0; i < numPairs; ++i) {
            const float freq = std::pow(base, -2.0f * (float)i / (float)headDim); //Pair index
            const float angle = (float)position * freq;
            const float cosA = std::cos(angle);
            const float sinA = std::sin(angle);

            const float pair_0 = X.At(2 * i, r);
            const float pair_1 = X.At(2 * i + 1, r);
            out.At(2 * i, r) = pair_0 * cosA - pair_1 * sinA;
            out.At(2 * i + 1, r) = pair_0 * sinA + pair_1 * cosA;
        }
    }
    return out;
}

// Table is a tensor of (embedDim, vocabSize) 
// and we will return the corresponding embeddings in the sequence of tokenIds
// so output tensor will have the size (embedDim, seqLen)
inline Tensor GetRows(const Tensor& table, const int* tokenIds, int64_t seqLen) {
    const int64_t embedDim = table.elementCounts[0];

    Tensor out(embedDim, seqLen);
    for (int64_t s = 0; s < seqLen; ++s) {
        const int64_t id = tokenIds[s];
        for (int64_t f = 0; f < embedDim; ++f) {
            out.At(f, s) = table.At(f, id);
        }
    }
    return out;
}

// Greedy sample: vocab index of the largest logit at sequence position `pos`.
// logits: {vocabSize, seqLen}.  The next-token prediction is at pos = seqLen - 1.
inline int64_t ArgMax(const Tensor& logits, int64_t pos) {
    const int64_t vocabSize = logits.elementCounts[0];

    int64_t best = 0;
    float   bestVal = logits.At(0, pos);
    for (int64_t v = 1; v < vocabSize; ++v) {
        const float val = logits.At(v, pos);
        if (val > bestVal) {
            bestVal = val;
            best = v;
        }
    }
    return best;
}

// Temperature sampling: draw a token from softmax(logits[:,pos] / temperature).
// temperature <= 0 falls back to greedy (ArgMax) -- the cold limit.
// logits: {vocabSize, seqLen}.  rng carries the sampler's state across calls.
inline int64_t Sample(const Tensor& logits, int64_t pos, float temperature, std::mt19937& rng) {
    const int64_t vocabSize = logits.elementCounts[0];

    if (temperature <= 0.0f) {
        return ArgMax(logits, pos);
    }

    // Numerically stable softmax of (z / temperature): subtract the max first so the
    // exponent is always <= 0 (the shift is a constant factor that cancels out).
    float maxLogit = logits.At(0, pos);
    for (int64_t v = 1; v < vocabSize; ++v) {
        maxLogit = std::max(maxLogit, logits.At(v, pos));
    }

    std::vector<float> probs(vocabSize);
    float sum = 0.0f;
    for (int64_t v = 0; v < vocabSize; ++v) {
        const float e = std::exp((logits.At(v, pos) - maxLogit) / temperature);
        probs[v] = e;
        sum += e;
    }

    // Inverse-CDF draw: pick r in [0, sum) and walk the (unnormalized) cumulative sum.
    // Drawing against sum avoids a separate normalization pass.
    std::uniform_real_distribution<float> dist(0.0f, sum);
    const float r = dist(rng);
    float cumulative = 0.0f;
    for (int64_t v = 0; v < vocabSize; ++v) {
        cumulative += probs[v];
        if (r < cumulative) {
            return v;
        }
    }
    return vocabSize - 1; // guard against floating-point rounding at the tail
}

// SwiGLU feed-forward (Llama):  out = Wdown · ( SiLU(Wgate·x) ⊙ (Wup·x) )
// X: {embedDim, seqLen}.  Weights oriented {outFeatures, inFeatures}, same as your
// attention projections:  Wgate, Wup = {hiddenDim, embedDim};  Wdown = {embedDim, hiddenDim}.
inline Tensor SwiGLU(const Tensor& X, const Tensor& Wgate, const Tensor& Wup, const Tensor& Wdown) {
    Tensor gate = MatMul(X, Wgate);     // {hiddenDim, seqLen}
    Tensor up = MatMul(X, Wup);       // {hiddenDim, seqLen}
    Tensor act = Mul(SiLU(gate), up);  // gate it: SiLU(gate) ⊙ up   {hiddenDim, seqLen}
    return MatMul(act, Wdown);          // back down to {embedDim, seqLen}
}

// Causal mask: mask the scores of tokens with their successors.
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

// Attention(Q, K, V) = softmax(Q*K^T / sqrt(headDim)) * V, optionally causal.
// Q: {headDim, seqQ}, K: {headDim, seqK}, V: {headDimV, seqK} -> out: {headDimV, seqQ}.
inline Tensor Attention(const Tensor& Q, const Tensor& K, const Tensor& V, bool causal = false) {
    const int64_t headDim = Q.elementCounts[0];
    // Note that  we are not breaking strided access here.
    // because inside MatMul we are accesing A.At(k,m) and B.at(n,k) where k is the loop variable. so it visits columns first in A; and rows first in B but in B rows are actually columns because
    // they are TRANSPOSED
    Tensor scores = MatMul(Q, Transpose(K));
    scores = Scale(scores, 1.0f / std::sqrt((float)headDim));
    if (causal) {
        MaskCausal(scores);
    }
    Tensor weights = Softmax(scores);
    return MatMul(weights, V);
}

// Multi-head attention: split the embedding into numHeads heads, attend per head,
// concatenate back. Q/K/V: {embedDim, seqLen}, embedDim = numHeads * headDim.
inline Tensor MultiHeadAttention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                 int numHeads, bool causal = false) {
    //Q is a tensor of shape {numberOfFeatures per token, numberOfTokens}
    //How about K and V?
	const int64_t embedDim = Q.elementCounts[0];
    const int64_t seqLen   = Q.elementCounts[1];
    const int64_t headDim  = embedDim / numHeads;
    assert(embedDim % numHeads == 0);

    Tensor out(embedDim, seqLen);
    for (int h = 0; h < numHeads; ++h) {
        const int64_t start = h * headDim;
        Tensor oh = Attention(Slice(Q, 0, start, headDim), Slice(K, 0, start, headDim), Slice(V, 0, start, headDim), causal);
        for (int64_t r = 0; r < seqLen; ++r) {
            for (int64_t c = 0; c < headDim; ++c) {
                out.At(h * headDim + c, r) = oh.At(c, r);
            }
        }
    }
    return out;
}

// One-token attention against the KV cache (llama2.c-style).
//   q:  {dim,1}     current token's query  (already RoPE'd)
//   k:  {kv_dim,1}  current token's key    (already RoPE'd)
//   v:  {kv_dim,1}  current token's value
//   kCache:   {kv_dim, seq_len}  THIS layer's cache slice; columns = positions
//   vCache: {kv_dim, seq_len}  THIS layer's cache slice; columns = positions
//   pos: current position -> write column `pos`, attend over columns 0..pos
//   numHeads: query heads. Assumes n_kv_heads == n_heads (stories15M); GQA is a later fix.
inline Tensor MultiHeadAttentionWithKVCache(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& kCache, Tensor& vCache, int64_t pos, int numHeads)
{
    const int64_t dim = q.elementCounts[0];
    const int64_t kvDim = kCache.elementCounts[0];
    const int64_t headDim = dim / numHeads;

    // (1) append this token's k,v into the cache at column `pos`
    for (int64_t c = 0; c < kvDim; ++c) {
        kCache.At(c, pos) = k.At(c, 0);
        vCache.At(c, pos) = v.At(c, 0);
    }

    Tensor out(dim, 1);
    for (int h = 0; h < numHeads; ++h) {
        const int64_t offset = h * headDim;

        Tensor qHead = Slice(q, 0, offset, headDim); //A simple trick to extract [offset, offset+headDim) portion of q. i.e. we are getting each head of q here.
        
        // Double slice. Inner Slice(dim 0) picks this head; same head-extraction as q, but on the cache.
        // Outer Slice(dim 1) picks columns [0, pos+1); every position cached so far (0..pos, including the token we just wrote). This is simply causal masking!
        // Result {headDim, pos+1} = this head's keys for every token seen up to now
        Tensor kHead = Slice(Slice(kCache, 0, offset, headDim), 1, 0, pos + 1);   // {headDim, pos+1}
        Tensor vHead = Slice(Slice(vCache, 0, offset, headDim), 1, 0, pos + 1);   // {headDim, pos+1}

        //causal = false because it is not needed. setting it =true would be pointless, causal masking is embedded to tensors already.
        Tensor oHead = Attention(qHead, kHead, vHead, /*causal=*/false);           // {headDim, 1}

        for (int64_t c = 0; c < headDim; ++c)
            out.At(offset + c, 0) = oHead.At(c, 0);
    }
    return out;
}

// Our current Ops are way too generic for direct use.
// 'Project' op will help us to project the input tensor on the weight tensor of a specific layer.
inline Tensor Project(const Tensor& x, const Tensor& W, int64_t layer) {
    return MatMul(x, Transpose(Slice(W, 2, layer, 1)));
    //                         └─ pick layer  ┘ => remember that our loaded weights has multiple layers
    //               └─ fix orientation ┘
    //     └─ project every token ┘
}

// Apply RoPE per head. X is {dim, seqLen} with dim = numHeads * headDim; each head's
// headDim-wide slice is rotated independently (RoPE's frequency schedule restarts per head). 
// Used on Q (numHeads = n_heads) and K (numHeads = n_kv_heads); V is untouched.
inline Tensor ApplyRoPE(const Tensor& X, int numHeads, float base = 10000.0f, int64_t positionOffset = 0) {
    const int64_t dim     = X.elementCounts[0];
    const int64_t seqLen  = X.elementCounts[1];
    const int64_t headDim = dim / numHeads;
    assert(dim % numHeads == 0);

    Tensor out(dim, seqLen);
    for (int h = 0; h < numHeads; ++h) {
        const int64_t start = h * headDim;
        Tensor head    = Slice(X, 0, start, headDim);          // {headDim, seqLen} strided view
        Tensor rotated = RoPE(head, base, positionOffset);     // rotate this head alone
        for (int64_t r = 0; r < seqLen; ++r) {                 // paste back into out's head-h columns
            for (int64_t c = 0; c < headDim; ++c) {
                out.At(start + c, r) = rotated.At(c, r);
            }
        }
    }
    return out;
}

}  // namespace basicllm
