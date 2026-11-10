#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 752;
// Each entry is overwritten by the first kernel before the second kernel reads it.
// Default-stream ordering also makes the workspace reusable between calls.
__device__ float partial_sums[kMaxBlocks];

__device__ __forceinline__ float element_value(float x, float y) {
    const float d = x - y;
    const float a = fabsf(d);
    const float p = __fmul_rn(a, a);
    const float s = __fadd_rn(p, 1.0e-6f);
    // s is never subnormal. The native square root is accurate to a few ulps.
    float v;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(v) : "f"(s));
    return v;
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, offset);
    return x;
}

template<int Threads>
__device__ __forceinline__ float block_sum(float x) {
    x = warp_sum(x);
    if constexpr (Threads > 32) {
        __shared__ float sums[Threads / 32];
        if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = x;
        __syncthreads();
        x = threadIdx.x < Threads / 32 ? sums[threadIdx.x] : 0.0f;
        if (threadIdx.x < 32) x = warp_sum(x);
    }
    return x;
}

template<int Threads>
__global__ void small_reduce(const float* __restrict__ inp,
                             const float* __restrict__ target,
                             int64_t n, float* output) {
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += Threads)
        sum += element_value(inp[i], target[i]);
    sum = block_sum<Threads>(sum);
    if (threadIdx.x == 0) *output = sum / static_cast<float>(n);
}

template<bool Aligned>
__global__ void main_reduce(const float* __restrict__ inp,
                            const float* __restrict__ target,
                            int64_t n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    const int64_t nv = n / 4; // The channel count is four, so there is no tail.
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
         i < nv; i += stride) {
        float4 x, y;
        if constexpr (Aligned) {
            x = reinterpret_cast<const float4*>(inp)[i];
            y = reinterpret_cast<const float4*>(target)[i];
        } else {
            x = make_float4(inp[4*i], inp[4*i+1], inp[4*i+2], inp[4*i+3]);
            y = make_float4(target[4*i], target[4*i+1], target[4*i+2], target[4*i+3]);
        }
        s0 += element_value(x.x, y.x);
        s1 += element_value(x.y, y.y);
        s2 += element_value(x.z, y.z);
        s3 += element_value(x.w, y.w);
    }
    const float sum = block_sum<kThreads>((s0 + s1) + (s2 + s3));
    if (threadIdx.x == 0) partial_sums[blockIdx.x] = sum;
}

__global__ void finish_reduce(int count, int64_t n, float* output) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < count; i += kThreads)
        sum += partial_sums[i];
    sum = block_sum<kThreads>(sum);
    if (threadIdx.x == 0) *output = sum / static_cast<float>(n);
}
} // namespace

void reference(const float* inp, const float* target, int64_t batch_size,
               int64_t height, int64_t width, float* output) {
    const int64_t n = batch_size * 4 * height * width;
    if (n <= 32) {
        small_reduce<32><<<1, 32>>>(inp, target, n, output);
    } else if (n <= 4096) {
        small_reduce<kThreads><<<1, kThreads>>>(inp, target, n, output);
    } else {
        const int64_t needed = (n + 4 * kThreads - 1) / (4 * kThreads);
        const int blocks = needed < kMaxBlocks ? static_cast<int>(needed) : kMaxBlocks;
        if (((reinterpret_cast<uintptr_t>(inp) | reinterpret_cast<uintptr_t>(target)) & 15) == 0)
            main_reduce<true><<<blocks, kThreads>>>(inp, target, n);
        else
            main_reduce<false><<<blocks, kThreads>>>(inp, target, n);
        finish_reduce<<<1, kThreads>>>(blocks, n, output);
    }
}
