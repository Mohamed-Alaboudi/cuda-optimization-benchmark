#include <cuda_runtime.h>
#include <stdint.h>
#include <cub/block/block_scan.cuh>

namespace {
// An affine map x -> b + a*x.  Items are ordered from the end
// of a row to its beginning, so composition is right-after-left.
struct Affine {
    float b, a;
};
struct Compose {
    __device__ __forceinline__ Affine operator()(const Affine& x, const Affine& y) const {
        return {fmaf(y.a, x.b, y.b), y.a * x.a};
    }
};

template<int ITEMS>
__global__ void reward_scan(const float* __restrict__ reward,
                            const bool* __restrict__ done,
                            float gamma, int T,
                            float* __restrict__ output) {
    constexpr int THREADS = 256;
    using Scan = cub::BlockScan<Affine, THREADS>;
    __shared__ typename Scan::TempStorage storage;
    Affine items[ITEMS];
    const size_t base = size_t(blockIdx.x) * T;
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        int t = T - 1 - (int(threadIdx.x) * ITEMS + i);
        items[i] = (t >= 0) ? Affine{reward[base+t], done[base+t] ? 0.0f : gamma} : Affine{0.0f, 1.0f};
    }
    // The additive component of each inclusive prefix is its reward-to-go.
    // A terminal has a == 0, stopping contributions across that boundary.
    Scan(storage).InclusiveScan(items, items, Compose{});
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        int t = T - 1 - (int(threadIdx.x) * ITEMS + i);
        if (t >= 0) output[base+t] = items[i].b;
    }
}

__global__ void reward_serial(const float* reward, const bool* done, float gamma, int B, int T, float* output) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    size_t base = size_t(b)*T;
    float next = 0.0f;
    for (int t = T-1; t >= 0; --t) {
        next = fmaf(gamma, done[base+t] ? 0.0f : next, reward[base+t]);
        output[base+t] = next;
    }
}
}

void reference(const float* reward, const bool* done, float gamma, int32_t B, int32_t T, float* output) {
    if (B <= 0 || T <= 0) return;
    if (T <= 256) reward_scan<1><<<B,256>>>(reward,done,gamma,T,output);
    else if (T <= 1024) reward_scan<4><<<B,256>>>(reward,done,gamma,T,output);
    else if (T <= 8192) reward_scan<32><<<B,256>>>(reward,done,gamma,T,output);
    else reward_serial<<<(B+127)/128,128>>>(reward,done,gamma,B,T,output);
}
