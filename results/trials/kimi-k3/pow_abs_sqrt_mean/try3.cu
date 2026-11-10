#include <cuda_runtime.h>
#include <cstdint>

// Fused: mean of sqrt((inp-target)^2 + 1e-6) over all elements.
// Channel dim is always 4, so total element count is divisible by 4 -> float4 loads.

__global__ void fused_mean_kernel(const float4* __restrict__ inp,
                                  const float4* __restrict__ target,
                                  long long n4,
                                  float inv_n,
                                  float* __restrict__ out) {
    float acc = 0.0f;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) {
        float4 a = inp[i];
        float4 b = target[i];
        float d0 = fabsf(a.x - b.x);
        float d1 = fabsf(a.y - b.y);
        float d2 = fabsf(a.z - b.z);
        float d3 = fabsf(a.w - b.w);
        acc += sqrtf(d0 * d0 + 1e-6f);
        acc += sqrtf(d1 * d1 + 1e-6f);
        acc += sqrtf(d2 * d2 + 1e-6f);
        acc += sqrtf(d3 * d3 + 1e-6f);
    }
    // warp reduction
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float sh[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) >> 5;
        float v = (lane < nw) ? sh[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) atomicAdd(out, v * inv_n);
    }
}

void reference(const float* inp, const float* target, int64_t batch_size, int64_t height, int64_t width, float* output) {
    long long n = (long long)batch_size * 4 * height * width;
    long long n4 = n >> 2;  // n always divisible by 4
    cudaMemsetAsync(output, 0, sizeof(float), 0);
    if (n4 == 0) return;
    const int block = 256;
    long long want = (n4 + block - 1) / block;
    int blocks = (int)(want < 4096 ? want : 4096);
    float inv_n = 1.0f / (float)n;
    fused_mean_kernel<<<blocks, block, 0, 0>>>(
        reinterpret_cast<const float4*>(inp),
        reinterpret_cast<const float4*>(target),
        n4, inv_n, output);
}
