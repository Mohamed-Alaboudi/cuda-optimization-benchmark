#include <cuda_runtime.h>
#include <stdint.h>

// Intermediate sums are published by each CTA, then consumed by the final CTA.
static __device__ volatile float partial[1024];
static __device__ unsigned int arrivals = 0;

static __device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, offset);
    return x;
}

static __device__ __forceinline__ float block_sum(float x, float* scratch) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    x = warp_sum(x);
    if (lane == 0) scratch[warp] = x;
    __syncthreads();
    if (warp == 0) {
        x = lane < (blockDim.x >> 5) ? scratch[lane] : 0.0f;
        x = warp_sum(x);
    }
    return x;
}

static __device__ __forceinline__ float element(float a, float b) {
    float d = a - b;
    // Do not contract the multiply and add: the operation specifies two
    // separate fp32 roundings.  The argument of sqrt is always >= 1e-6.
    float s = __fmul_rn(d, d) + 1.0e-6f;
    float result;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(s));
    return result;
}

// Common last-CTA reduction, no scratch allocation or extra kernel launch.
static __device__ __forceinline__ void finish(float x, float inv_n, float* output,
                                               float* scratch, bool* is_last) {
    x = block_sum(x, scratch);
    if (gridDim.x == 1) {
        if (threadIdx.x == 0) *output = x * inv_n;
        return;
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = x;
        __threadfence();
        *is_last = atomicAdd(&arrivals, 1) == gridDim.x - 1;
    }
    __syncthreads();
    if (*is_last) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
            sum += partial[i];
        sum = block_sum(sum, scratch);
        if (threadIdx.x == 0) {
            *output = sum * inv_n;
            arrivals = 0;
        }
    }
}

// Aligned vector loads process four adjacent elements per lane.
__global__ void vector_kernel(const float4* __restrict__ a,
                              const float4* __restrict__ b,
                              int n4, float inv_n, float* __restrict__ output) {
    __shared__ float scratch[8];
    __shared__ bool is_last;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (; i < n4; i += step) {
        float4 x = a[i];
        float4 y = b[i];
        s0 += element(x.x, y.x);
        s1 += element(x.y, y.y);
        s2 += element(x.z, y.z);
        s3 += element(x.w, y.w);
    }
    finish((s0 + s1) + (s2 + s3), inv_n, output, scratch, &is_last);
}

// Handles unaligned contiguous arrays and exceptionally large element counts.
__global__ void scalar_kernel(const float* __restrict__ a,
                              const float* __restrict__ b,
                              int64_t n, float inv_n, float* __restrict__ output) {
    __shared__ float scratch[8];
    __shared__ bool is_last;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = (int64_t)gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (; i < n; i += step) sum += element(a[i], b[i]);
    finish(sum, inv_n, output, scratch, &is_last);
}

__global__ void zero_kernel(float* output) { *output = 0.0f; }

void reference(const float* inp, const float* target, int64_t batch_size,
               int64_t height, int64_t width, float* output) {
    int64_t n = batch_size * 4 * height * width;
    if (n <= 0) {
        zero_kernel<<<1, 1>>>(output);
        return;
    }
    constexpr int threads = 256;
    int64_t vec_count = n / 4;
    int blocks = (int)((vec_count + threads - 1) / threads > 1024 ? 1024 :
                       (vec_count + threads - 1) / threads);
    if (vec_count <= threads * 4) blocks = 1;
    float inv_n = (float)(1.0 / (double)n);
    if ((((uintptr_t)inp | (uintptr_t)target) & 15) == 0 && vec_count < 1000000000) {
        vector_kernel<<<blocks, threads>>>((const float4*)inp, (const float4*)target,
                                           (int)vec_count, inv_n, output);
    } else {
        scalar_kernel<<<blocks, threads>>>(inp, target, n, inv_n, output);
    }
}
