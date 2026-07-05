#pragma once

#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

#include "Tensor.h"

// Chapter 1: Tensor operations
//
// Every function below has a comment on top describing exactly what it should
// compute, and an EMPTY body for you to fill in. Implement them one at a time,
// then build and run the tests in main.cpp to check your work.
//
// Naming convention used in the formulas:
//   "column" is axis 0 (elementCounts[0]); neighbours in memory.
//   "row"    is axis 1 (elementCounts[1]).
// So a 2D tensor is read as: value at (column, row) = A.At(column, row).
//
// A placeholder `return {};` is left in each function so the file compiles before
// you start. An unimplemented op returns an empty tensor, and the tests report it
// as "TODO" instead of crashing. Delete the placeholder as you go.

namespace basicllm {

// Transpose: swap the column axis and the row axis.
//
// This does NOT move any data. It returns a view over the same memory with axis 0
// and axis 1 (both their counts and their strides) swapped. After it,
// result.At(column, row) reads the same float as A.At(row, column).
//
// Shape:  A is {cols, rows}  ->  result is {rows, cols}
inline Tensor Transpose(const Tensor& A) {
    // TODO: implement (Chapter 1, Task 1)
    return {};
}

// Scale: multiply every element by a single number.
//
//   result[column, row] = A[column, row] * scalar
inline Tensor Scale(const Tensor& A, float scalar) {
    // TODO: implement (Chapter 1, Task 2)
    return {};
}

// Add: element-by-element addition. A and B must have the same shape.
//
//   result[column, row] = A[column, row] + B[column, row]
inline Tensor Add(const Tensor& A, const Tensor& B) {
    // TODO: implement (Chapter 1, Task 3)
    return {};
}

// Mul: element-by-element multiplication. A and B must have the same shape.
//
//   result[column, row] = A[column, row] * B[column, row]
inline Tensor Mul(const Tensor& A, const Tensor& B) {
    // TODO: implement (Chapter 1, Task 4)
    return {};
}

// MatMul: matrix multiplication.
//
//   A is {K, M}, B is {N, K}, result C is {N, M}.
//   For each output position:
//     C[n, m] = sum over k of  A[k, m] * B[n, k]
//   where k runs across the shared inner dimension K.
//
//   The number of columns of A (which is K) must equal the number of rows of B.
inline Tensor MatMul(const Tensor& A, const Tensor& B) {
    // TODO: implement (Chapter 1, Task 5)
    return {};
}

// Softmax: turn each ROW of numbers into a probability distribution.
// Applied independently to every row (one pass per row).
//
//   For a given row, let m = the largest value in that row. Then:
//     result[column, row] = exp(A[column, row] - m) / sum over c of exp(A[c, row] - m)
//
//   Subtracting the max (m) before exp() is only for numerical safety; it does not
//   change the result, it just prevents exp() from overflowing.
inline Tensor Softmax(const Tensor& A) {
    // TODO: implement (Chapter 1, Task 6)
    return {};
}

// RMSNorm: Root-Mean-Square normalization (the normalization Llama uses).
// Applied independently to every row. `weight` has one value per column.
//
//   For a given row with N columns:
//     meanSquare = ( sum over c of X[c, row]^2 ) / N
//     scale      = 1 / sqrt(meanSquare + eps)
//     result[column, row] = X[column, row] * scale * weight[column]
//
//   eps is a tiny number that avoids dividing by zero.
inline Tensor RMSNorm(const Tensor& X, const Tensor& weight, float eps = 1e-5f) {
    // TODO: implement (Chapter 1, Task 7)
    return {};
}

// SiLU: the activation function used in Llama's feed-forward block.
// Applied to every element. sigmoid(x) = 1 / (1 + exp(-x)).
//
//   result[column, row] = X[column, row] * sigmoid( X[column, row] )
inline Tensor SiLU(const Tensor& X) {
    // TODO: implement (Chapter 1, Task 8)
    return {};
}

// RoPE: Rotary Position Embedding. Encodes a token's POSITION by rotating pairs of
// its features. X is {headDim, seqLen}: each column is one token's vector, and
// headDim must be even (features are handled in pairs).
//
//   For token at row `r`, its position is (positionOffset + r).
//   For pair index i = 0, 1, ..., (headDim/2 - 1):
//       frequency = base ^ ( -2 * i / headDim )
//       angle     = position * frequency
//       x0 = X[2*i,     r]        (first element of the pair)
//       x1 = X[2*i + 1, r]        (second element of the pair)
//       result[2*i,     r] = x0 * cos(angle) - x1 * sin(angle)
//       result[2*i + 1, r] = x0 * sin(angle) + x1 * cos(angle)
//
//   (That last step is just rotating the 2D point (x0, x1) by `angle`.)
inline Tensor RoPE(const Tensor& X, float base = 10000.0f, int64_t positionOffset = 0) {
    // TODO: implement (Chapter 1, Task 9)
    return {};
}

// MaskCausal: enforce that a token cannot attend to tokens that come after it.
// `scores` is {keys, queries}: scores.At(keyColumn, queryRow) is how much query
// `queryRow` attends to key `keyColumn`. Modifies `scores` IN PLACE.
//
//   For every query row r, set every key column c with c > r to negative infinity.
//   (After softmax, negative infinity becomes a weight of zero, so those future
//    keys are ignored.)
//
//   Use: -std::numeric_limits<float>::infinity()
inline void MaskCausal(Tensor& scores) {
    // TODO: implement (Chapter 1, Task 10)
}

// Attention (single head). Build this from the ops above.
//   Q is {headDim, seqQ}, K is {headDim, seqK}, V is {headDimV, seqK}.
//   Output is {headDimV, seqQ}.
//
//   Steps:
//     1. scores = MatMul(Q, Transpose(K))          -> {seqK, seqQ}
//     2. scores = Scale(scores, 1 / sqrt(headDim))  (headDim = Q's column count)
//     3. if (causal) MaskCausal(scores)
//     4. weights = Softmax(scores)
//     5. return MatMul(weights, V)                  -> {headDimV, seqQ}
inline Tensor Attention(const Tensor& Q, const Tensor& K, const Tensor& V, bool causal = false) {
    // TODO: implement (Chapter 1, Task 11) using the ops you wrote above
    return {};
}

}  // namespace basicllm
