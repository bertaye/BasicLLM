#pragma once

#include <cstdint>
#include <cstdlib>
#include <cassert>

namespace basicllm {

// Max dimensions a tensor can have (cf. ggml's GGML_MAX_DIMS).
constexpr int MAX_DIMS = 4;

// CPU only; f32 tensor.
// count0: column dimension (innermost/contiguous axis)
// count1: row dimension
// count2: layer dimension
// count3: batch dimension (outermost axis)
struct Tensor
{
    int64_t  elementCounts[MAX_DIMS] = {1, 1, 1, 1};
    size_t   byteStrides[MAX_DIMS]   = {0, 0, 0, 0};
    uint8_t* data     = nullptr;
    int*     refCount = nullptr;

    Tensor() = default;

    explicit Tensor(int64_t count0, int64_t count1 = 1, int64_t count2 = 1, int64_t count3 = 1) {
        elementCounts[0] = count0;
        elementCounts[1] = count1;
        elementCounts[2] = count2;
        elementCounts[3] = count3;
        byteStrides[0] = sizeof(float);
        byteStrides[1] = byteStrides[0] * count0;
        byteStrides[2] = byteStrides[1] * count1;
        byteStrides[3] = byteStrides[2] * count2;
        data     = static_cast<uint8_t*>(std::malloc(TotalBytes()));
        refCount = new int(1);
    }

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
        t.refCount = nullptr; // No ownership of data
        return t;
    }

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

    int64_t TotalElements() const {
        return elementCounts[0] * elementCounts[1] * elementCounts[2] * elementCounts[3];
    }

    size_t TotalBytes() const {
        return static_cast<size_t>(TotalElements()) * sizeof(float);
    }

    int RefCount() const {
        return refCount ? *refCount : 0;
    }

    // Addressing is the same byte-stride math ggml uses; works on views too.
    float& At(int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
        assert(data);
        return *reinterpret_cast<float*>(data + i0 * byteStrides[0] + i1 * byteStrides[1] + i2 * byteStrides[2] + i3 * byteStrides[3]);
    }

    float At(int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) const {
        assert(data);
        return *reinterpret_cast<const float*>(data + i0 * byteStrides[0] + i1 * byteStrides[1] + i2 * byteStrides[2] + i3 * byteStrides[3]);
    }

    bool IsContiguous() const {
        return byteStrides[0] == sizeof(float)
            && byteStrides[1] == byteStrides[0] * static_cast<size_t>(elementCounts[0])
            && byteStrides[2] == byteStrides[1] * static_cast<size_t>(elementCounts[1])
            && byteStrides[3] == byteStrides[2] * static_cast<size_t>(elementCounts[2]);
    }
};

}  // namespace basicllm
