#pragma once

#include <cstdint>
#include <cstdlib>
#include <cassert>

namespace basicllm {

// The most dimensions a tensor is allowed to have. We use four, which is enough
// for everything in this project (features, sequence, layers, and a spare axis)
constexpr int MAX_DIMS = 4;

// A CPU-only tensor of 32-bit floats.
//
// A tensor here is nothing more than:
//   1. a flat block of floats in memory (`data`), plus
//   2. some numbers that describe how to read that flat block as if it had a shape.
//
// The four axes, from innermost (fastest changing in memory) to outermost:
//   count0: the column axis   (neighbouring elements sit next to each other in memory)
//   count1: the row axis
//   count2: the layer axis
//   count3: the batch axis    (the slowest changing, biggest jumps in memory)
struct Tensor
{
    // How many elements exist along each axis.
    int64_t  elementCounts[MAX_DIMS] = {1, 1, 1, 1};

    // How many BYTES you move in memory to step one element along each axis
    // This is the key idea that lets us reshape/transpose/slice without copying
    size_t   byteStrides[MAX_DIMS]   = {0, 0, 0, 0};

    uint8_t* data     = nullptr;  // the flat block of floats (as raw bytes)
    int*     refCount = nullptr;  // how many Tensors share `data` (nullptr = we borrow it)

    Tensor() = default;

    // Allocate a brand new, contiguous tensor of the given shape.
    // "Contiguous" means the strides are packed tightly with no gaps
    explicit Tensor(int64_t count0, int64_t count1 = 1, int64_t count2 = 1, int64_t count3 = 1) {
        elementCounts[0] = count0;
        elementCounts[1] = count1;
        elementCounts[2] = count2;
        elementCounts[3] = count3;
        // Step one column  = one float.
        // Step one row     = skip a whole column-run.
        // Step one layer   = skip a whole row-of-columns block. And so on.
        byteStrides[0] = sizeof(float);
        byteStrides[1] = byteStrides[0] * count0;
        byteStrides[2] = byteStrides[1] * count1;
        byteStrides[3] = byteStrides[2] * count2;
        data     = static_cast<uint8_t*>(std::malloc(TotalBytes()));
        refCount = new int(1);
    }

    // Wrap an existing block of memory as a tensor WITHOUT copying or owning it.
    // Used for model weights that already live in a loaded file
    static Tensor View(int64_t count0, int64_t count1, int64_t count2, int64_t count3, uint8_t* dataPtr) {
        Tensor t;
        t.elementCounts[0] = count0;
        t.elementCounts[1] = count1;
        t.elementCounts[2] = count2;
        t.elementCounts[3] = count3;
        t.byteStrides[0] = sizeof(float);
        t.byteStrides[1] = t.byteStrides[0] * count0;
        t.byteStrides[2] = t.byteStrides[1] * count1;
        t.byteStrides[3] = t.byteStrides[2] * count2;
        t.data     = dataPtr;
        t.refCount = nullptr; // we do not own this memory, so we never free it
        return t;
    }

    // Copy / move / destroy: reference-counted sharing of `data`.
    // You do not need to fully understand these to do Chapter 1. The short version:
    // copying a Tensor shares the same underlying memory and bumps a counter; the
    // memory is freed only when the last owner goes away

    Tensor(const Tensor& other) {
        for (int i = 0; i < MAX_DIMS; ++i) {
            elementCounts[i] = other.elementCounts[i];
            byteStrides[i]   = other.byteStrides[i];
        }
        data     = other.data;
        refCount = other.refCount;
        if (refCount) {
            ++*refCount;
        }
    }

    Tensor& operator=(const Tensor& other) {
        if (this == &other) {
            return *this;
        }
        if (other.refCount) {
            ++*other.refCount;
        }
        if (refCount && --*refCount == 0) {
            std::free(data);
            delete refCount;
        }
        for (int i = 0; i < MAX_DIMS; ++i) {
            elementCounts[i] = other.elementCounts[i];
            byteStrides[i]   = other.byteStrides[i];
        }
        data     = other.data;
        refCount = other.refCount;
        return *this;
    }

    Tensor(Tensor&& other) noexcept {
        for (int i = 0; i < MAX_DIMS; ++i) {
            elementCounts[i] = other.elementCounts[i];
            byteStrides[i]   = other.byteStrides[i];
        }
        data     = other.data;
        refCount = other.refCount;
        other.data     = nullptr;
        other.refCount = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (refCount && --*refCount == 0) {
            std::free(data);
            delete refCount;
        }
        for (int i = 0; i < MAX_DIMS; ++i) {
            elementCounts[i] = other.elementCounts[i];
            byteStrides[i]   = other.byteStrides[i];
        }
        data     = other.data;
        refCount = other.refCount;
        other.data     = nullptr;
        other.refCount = nullptr;
        return *this;
    }

    ~Tensor() {
        if (refCount && --*refCount == 0) {
            std::free(data);
            delete refCount;
        }
    }

    // Small helpers

    int64_t TotalElements() const {
        return elementCounts[0] * elementCounts[1] * elementCounts[2] * elementCounts[3];
    }

    size_t TotalBytes() const {
        return static_cast<size_t>(TotalElements()) * sizeof(float);
    }

    int RefCount() const {
        return refCount ? *refCount : 0;
    }

    // Read or write the element at a given position.
    // The position (i0, i1, i2, i3) is turned into a byte offset using the strides.
    // Because it uses strides (not a fixed formula), it works on views too.
    float& At(int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
        assert(data);
        return *reinterpret_cast<float*>(data + i0 * byteStrides[0] + i1 * byteStrides[1] + i2 * byteStrides[2] + i3 * byteStrides[3]);
    }

    float At(int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) const {
        assert(data);
        return *reinterpret_cast<const float*>(data + i0 * byteStrides[0] + i1 * byteStrides[1] + i2 * byteStrides[2] + i3 * byteStrides[3]);
    }

    // A tensor is contiguous when the strides are still the tightly-packed values
    // a fresh tensor would have. A transposed or sliced view is usually NOT.
    bool IsContiguous() const {
        return byteStrides[0] == sizeof(float)
            && byteStrides[1] == byteStrides[0] * static_cast<size_t>(elementCounts[0])
            && byteStrides[2] == byteStrides[1] * static_cast<size_t>(elementCounts[1])
            && byteStrides[3] == byteStrides[2] * static_cast<size_t>(elementCounts[2]);
    }
};

}  // namespace basicllm
