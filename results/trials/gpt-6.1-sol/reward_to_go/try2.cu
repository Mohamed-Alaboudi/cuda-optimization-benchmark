#include <cuda_runtime.h>
#include <cstdint>
#include <cub/block/block_scan.cuh>

namespace {

// An affine map x -> add + mul*x. Scan order is reverse time:
// 'later' is applied before 'earlier'.
struct Affine {
    float add;
    float mul;
};

struct Compose {
    __device__ __forceinline__ Affine operator()(const Affine& later,
                                                const Affine& earlier) const {
        return {fmaf(earlier.mul, later.add, earlier.add),
                earlier.mul * later.mul};
    }
};

template<int Items>
__global__ void scan_rows(const float* __restrict__ reward,
                          const bool* __restrict__ done,
                          float gamma, int T, float* __restrict__ output) {
    constexpr int Threads = 256;
    using Scan = cub::BlockScan<Affine, Threads>;
    __shared__ typename Scan::TempStorage storage;

    const int64_t row_start = int64_t(blockIdx.x) * T;
    const int first = T - 1 - int(threadIdx.x) * Items;
    float r[Items];
    float a[Items];
    Affine aggregate{0.0f, 1.0f};

    // Each thread summarizes a short, contiguous, reverse-time interval.
    #pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int t = first - j;
        r[j] = t >= 0 ? reward[row_start + t] : 0.0f;
        a[j] = (t >= 0 && !done[row_start + t]) ? gamma : 0.0f;
        aggregate = Compose{}(aggregate, Affine{r[j], a[j]});
    }

    // The exclusive prefix gives the return just after this thread's interval.
    Affine prefix;
    Scan(storage).ExclusiveScan(aggregate, prefix, Affine{0.0f, 1.0f}, Compose{});

    float value = prefix.add;
    #pragma unroll
    for (int j = 0; j < Items; ++j) {
        value = fmaf(a[j], value, r[j]);
        if (first - j >= 0)
            output[row_start + first - j] = value;
    }
}

// General-size fallback, also avoiding an upper bound on supported row lengths.
__global__ void serial_rows(const float* __restrict__ reward,
                            const bool* __restrict__ done,
                            float gamma, int B, int T,
                            float* __restrict__ output) {
    const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= B) return;
    const int64_t start = row * T;
    float value = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        value = fmaf(gamma, done[start + t] ? 0.0f : value, reward[start + t]);
        output[start + t] = value;
    }
}

} // namespace

void reference(const float* reward, const bool* done, float gamma,
               int32_t B, int32_t T, float* output) {
    if (B <= 0 || T <= 0) return;
    if (T <= 256)
        scan_rows<1><<<B, 256>>>(reward, done, gamma, T, output);
    else if (T <= 1024)
        scan_rows<4><<<B, 256>>>(reward, done, gamma, T, output);
    else if (T <= 8192)
        scan_rows<32><<<B, 256>>>(reward, done, gamma, T, output);
    else
        serial_rows<<<(int64_t(B) + 127) / 128, 128>>>(reward, done, gamma, B, T, output);
}
