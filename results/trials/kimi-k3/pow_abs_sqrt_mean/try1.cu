// pow_abs_sqrt_mean: mean over all elements of sqrt((inp-target)^2 + 1e-6), fp32.
// Single fused kernel: grid-stride float4 reduction, block reduce, last block
// (via atomic counter, classic threadfence reduction) writes the mean and
// resets the scratch state for the next call.

#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ double g_sum;        // zero-initialized at module load
__device__ unsigned int g_count; // zero-initialized at module load

__device__ __forceinline__ float term(float a, float b) {
    float d = a - b;
    return sqrtf(fmaf(d, d, 1e-6f));
}

__global__ void reduce_mean_kernel(const float* __restrict__ inp,
                                   const float* __restrict__ tgt,
                                   long long n,
                                   float* __restrict__ output) {
    float acc = 0.f;
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;

    const long long n4 = n >> 2;
    if ((((uintptr_t)inp | (uintptr_t)tgt) & 15) == 0) {
        const float4* __restrict__ i4 = reinterpret_cast<const float4*>(inp);
        const float4* __restrict__ t4 = reinterpret_cast<const float4*>(tgt);
        for (long long i = tid; i < n4; i += stride) {
            float4 a = i4[i];
            float4 b = t4[i];
            acc += term(a.x, b.x) + term(a.y, b.y) + term(a.z, b.z) + term(a.w, b.w);
        }
        // tail (at most 3 elements)
        for (long long i = (n4 << 2) + tid; i < n; i += stride)
            acc += term(inp[i], tgt[i]);
    } else {
        for (long long i = tid; i < n; i += stride)
            acc += term(inp[i], tgt[i]);
    }

    // block reduction
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float sh[32];
    __shared__ bool is_last;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        const int nwarp = (blockDim.x + 31) >> 5;
        float v = (lane < nwarp) ? sh[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) {
            atomicAdd(&g_sum, (double)v);
            __threadfence();
            is_last = (atomicAdd(&g_count, 1u) == (unsigned int)gridDim.x - 1u);
        }
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        output[0] = (float)(g_sum / (double)n);
        g_sum = 0.0;
        g_count = 0u;
    }
}

} // namespace

void reference(const float* inp, const float* target, int64_t batch_size,
               int64_t height, int64_t width, float* output) {
    const long long n = (long long)batch_size * 4LL * (long long)height * (long long)width;
    if (n <= 0) {
        cudaMemsetAsync(output, 0, sizeof(float));
        return;
    }
    const int threads = 256;
    const long long n4 = (n + 3) >> 2;
    long long want = (n4 + threads - 1) / threads;
    int blocks = (int)(want < 4096 ? want : 4096);
    if (blocks < 1) blocks = 1;
    reduce_mean_kernel<<<blocks, threads>>>(inp, target, n, output);
}
