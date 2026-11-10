#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace sqrt_mean_detail {
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 512;
// Every partial read by the second kernel is overwritten by the first kernel.
// Both launches use the default stream, so no counter or initialization is needed.
__device__ float partials[kMaxBlocks];

__device__ __forceinline__ float block_sum(float x) {
    __shared__ float warp_sums[kThreads / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, offset);
    if (lane == 0) warp_sums[warp] = x;
    __syncthreads();
    x = threadIdx.x < kThreads / 32 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return x;
}

__device__ __forceinline__ float value(float a, float b) {
    // Squaring also removes the sign, making a separate fabs unnecessary.
    // Keep multiply and add separate, as specified by the operation.
    float d = __fsub_rn(a, b);
    float s = __fadd_rn(__fmul_rn(d, d), 1.0e-6f);
    float result;
    // The approximate square root is accurate to a few fp32 ulps.
    // FTZ cannot affect s, since s is at least 1e-6.
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(s));
    return result;
}

template<bool Vectorized, bool Single>
__global__ void reduce_input(const float* __restrict__ inp,
                             const float* __restrict__ target,
                             int64_t n, float* __restrict__ output) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    if (Vectorized) {
        const float4* a = reinterpret_cast<const float4*>(inp);
        const float4* b = reinterpret_cast<const float4*>(target);
        for (; index < n / 4; index += stride) {
            float4 av = a[index];
            float4 bv = b[index];
            s0 += value(av.x, bv.x);
            s1 += value(av.y, bv.y);
            s2 += value(av.z, bv.z);
            s3 += value(av.w, bv.w);
        }
    } else {
        for (; index < n; index += stride)
            s0 += value(inp[index], target[index]);
    }
    float sum = block_sum((s0 + s1) + (s2 + s3));
    if (threadIdx.x == 0) {
        if (Single) *output = sum / (float)n;
        else partials[blockIdx.x] = sum;
    }
}

__global__ void reduce_partials(int blocks, int64_t n, float* output) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < blocks; i += kThreads)
        sum += partials[i];
    sum = block_sum(sum);
    if (threadIdx.x == 0) *output = sum / (float)n;
}
} // namespace sqrt_mean_detail

void reference(const float* inp, const float* target, int64_t batch_size,
               int64_t height, int64_t width, float* output) {
    using namespace sqrt_mean_detail;
    const int64_t n = batch_size * 4 * height * width;
    const bool aligned = ((reinterpret_cast<uintptr_t>(inp) |
                           reinterpret_cast<uintptr_t>(target)) & 15) == 0;
    if (n <= 4096) {
        if (aligned)
            reduce_input<true, true><<<1, kThreads>>>(inp, target, n, output);
        else
            reduce_input<false, true><<<1, kThreads>>>(inp, target, n, output);
    } else {
        int64_t needed = (n + 1023) / 1024;
        int blocks = (int)(needed < kMaxBlocks ? needed : kMaxBlocks);
        if (aligned)
            reduce_input<true, false><<<blocks, kThreads>>>(inp, target, n, output);
        else
            reduce_input<false, false><<<blocks, kThreads>>>(inp, target, n, output);
        reduce_partials<<<1, kThreads>>>(blocks, n, output);
    }
}
