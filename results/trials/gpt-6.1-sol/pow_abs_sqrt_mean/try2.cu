#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace {
constexpr int max_partials = 1536;
__device__ float workspace[max_partials];
__device__ __forceinline__ float value(float x, float y) {
    // Squaring makes the explicit absolute value redundant.
    float d = __fsub_rn(x, y);
    return sqrtf(__fadd_rn(__fmul_rn(d, d), 1.0e-6f));
}

__device__ __forceinline__ float block_sum(float x) {
    __shared__ float warp_sums[32];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        x += __shfl_down_sync(0xffffffff, x, offset);
    if (lane == 0) warp_sums[warp] = x;
    __syncthreads();
    x = threadIdx.x < (blockDim.x >> 5) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

__global__ void small_reduce(const float* __restrict__ inp, const float* __restrict__ target,
                             int64_t n, float* output) {
    float sum = 0.f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
        sum += value(inp[i], target[i]);
    sum = block_sum(sum);
    if (threadIdx.x == 0) *output = sum / float(n);
}

template<bool Vectorized>
__global__ void partial_reduce(const float* __restrict__ inp, const float* __restrict__ target,
                               int64_t n) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    if constexpr (Vectorized) {
        const int64_t nv = n / 4;
        const auto* a = reinterpret_cast<const float4*>(inp);
        const auto* b = reinterpret_cast<const float4*>(target);
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
             i < nv; i += int64_t(gridDim.x) * blockDim.x) {
            float4 x = a[i], y = b[i];
            s0 += value(x.x, y.x);
            s1 += value(x.y, y.y);
            s2 += value(x.z, y.z);
            s3 += value(x.w, y.w);
        }
    } else {
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
             i < n; i += int64_t(gridDim.x) * blockDim.x)
            s0 += value(inp[i], target[i]);
    }
    // The tensor has four channels, so n is always a multiple of four.
    float sum = block_sum((s0 + s1) + (s2 + s3));
    if (threadIdx.x == 0) workspace[blockIdx.x] = sum;
}

__global__ void final_reduce(int count, int64_t n, float* output) {
    float sum = 0.f;
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        sum += workspace[i];
    sum = block_sum(sum);
    if (threadIdx.x == 0) *output = sum / float(n);
}
}

void reference(const float* inp, const float* target, int64_t batch_size,
               int64_t height, int64_t width, float* output) {
    const int64_t n = batch_size * 4 * height * width;
    if (n <= 4096) {
        const int threads = n <= 32 ? 32 : (n <= 512 ? 128 : 256);
        small_reduce<<<1, threads>>>(inp, target, n, output);
    } else {
        // The default stream orders both passes and successive calls. Every
        // workspace entry used by the final pass is overwritten in this call.
        int blocks = int((n + 1023) / 1024);
        if (blocks > max_partials) blocks = max_partials;
        if (((reinterpret_cast<uintptr_t>(inp) | reinterpret_cast<uintptr_t>(target)) & 15) == 0)
            partial_reduce<true><<<blocks, 256>>>(inp, target, n);
        else
            partial_reduce<false><<<blocks, 256>>>(inp, target, n);
        final_reduce<<<1, 256>>>(blocks, n, output);
    }
}
