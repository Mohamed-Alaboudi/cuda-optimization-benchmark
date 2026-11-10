// Mean of sqrt((inp - target)^2 + 1e-6) over all elements.
// Fused single-pass reduction: vectorized float4 loads, warp + block reduce,
// one atomicAdd per block into the output (pre-scaled by 1/N).

#include <cuda_runtime.h>
#include <cstdint>

#define THREADS 256
#define EPS 1e-6f

__global__ void pow_abs_sqrt_mean_kernel(const float4* __restrict__ inp,
                                         const float4* __restrict__ target,
                                         long long n4, float inv_n,
                                         float* __restrict__ output) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    float acc = 0.0f;
    for (long long i = tid; i < n4; i += stride) {
        float4 a = inp[i];
        float4 b = target[i];
        float d0 = a.x - b.x;
        float d1 = a.y - b.y;
        float d2 = a.z - b.z;
        float d3 = a.w - b.w;
        acc += sqrtf(d0 * d0 + EPS);
        acc += sqrtf(d1 * d1 + EPS);
        acc += sqrtf(d2 * d2 + EPS);
        acc += sqrtf(d3 * d3 + EPS);
    }

    // warp reduce
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    __shared__ float shared[THREADS / 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = acc;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < THREADS / 32) ? shared[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0 && v != 0.0f)
            atomicAdd(output, v * inv_n);
    }
}

void reference(const float* inp, const float* target, int64_t batch_size,
               int64_t height, int64_t width, float* output) {
    long long n = (long long)batch_size * 4 * height * width;
    long long n4 = n >> 2;  // channel dim is 4, so n is always divisible by 4
    float inv_n = 1.0f / (float)n;

    cudaMemsetAsync(output, 0, sizeof(float), 0);

    int blocks = (int)((n4 + THREADS - 1) / THREADS);
    int max_blocks = 2048;  // grid-stride cap; plenty to saturate the GPU
    if (blocks > max_blocks) blocks = max_blocks;
    if (blocks < 1) blocks = 1;

    pow_abs_sqrt_mean_kernel<<<blocks, THREADS, 0, 0>>>(
        reinterpret_cast<const float4*>(inp),
        reinterpret_cast<const float4*>(target), n4, inv_n, output);
}
