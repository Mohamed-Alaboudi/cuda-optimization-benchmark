#include <cuda_runtime.h>
#include <stdint.h>

#define PSA_THREADS 256
#define PSA_MAX_BLOCKS 2048

__device__ double psa_partials[PSA_MAX_BLOCKS];
__device__ unsigned int psa_counter = 0;

__device__ __forceinline__ float psa_f(float x, float y) {
    float d = x - y;
    float a = fabsf(d);
    float p = a * a;
    float s = p + 1e-6f;
    return sqrtf(s);
}

__device__ __forceinline__ float4 ldg_stream(const float4* p) {
    return __ldg(p);
}

template <bool VEC>
__global__ void __launch_bounds__(PSA_THREADS)
psa_kernel(const float* __restrict__ inp, const float* __restrict__ tgt, int64_t n, float* out) {
    int64_t tid = (int64_t)blockIdx.x * PSA_THREADS + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * PSA_THREADS;
    float acc0 = 0.f, acc1 = 0.f;
    if (VEC) {
        int64_t n4 = n >> 2;
        const float4* a4 = reinterpret_cast<const float4*>(inp);
        const float4* b4 = reinterpret_cast<const float4*>(tgt);
        int64_t i = tid;
        for (; i + 3 * stride < n4; i += 4 * stride) {
            float4 x[4], y[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) { x[u] = ldg_stream(a4 + i + u * stride); y[u] = ldg_stream(b4 + i + u * stride); }
#pragma unroll
            for (int u = 0; u < 4; u += 2) {
                acc0 += psa_f(x[u].x, y[u].x) + psa_f(x[u].y, y[u].y) + psa_f(x[u].z, y[u].z) + psa_f(x[u].w, y[u].w);
                acc1 += psa_f(x[u+1].x, y[u+1].x) + psa_f(x[u+1].y, y[u+1].y) + psa_f(x[u+1].z, y[u+1].z) + psa_f(x[u+1].w, y[u+1].w);
            }
        }
        for (; i < n4; i += stride) {
            float4 x0 = ldg_stream(a4 + i), y0 = ldg_stream(b4 + i);
            acc0 += psa_f(x0.x, y0.x) + psa_f(x0.y, y0.y) + psa_f(x0.z, y0.z) + psa_f(x0.w, y0.w);
        }
        for (int64_t j = (n4 << 2) + tid; j < n; j += stride) acc1 += psa_f(inp[j], tgt[j]);
    } else {
        for (int64_t j = tid; j < n; j += stride) acc0 += psa_f(inp[j], tgt[j]);
    }
    double acc = (double)acc0 + (double)acc1;
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ double warp_s[PSA_THREADS / 32];
    __shared__ bool is_last;
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) warp_s[w] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        double s = 0.0;
        for (int k = 0; k < PSA_THREADS / 32; ++k) s += warp_s[k];
        psa_partials[blockIdx.x] = s;
        __threadfence();
        unsigned int prev = atomicAdd(&psa_counter, 1u);
        is_last = (prev == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last) {
        __threadfence();
        double s = 0.0;
        for (int k = threadIdx.x; k < (int)gridDim.x; k += PSA_THREADS)
            s += __ldcg(&psa_partials[k]);
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        __syncthreads();
        if (lane == 0) warp_s[w] = s;
        __syncthreads();
        if (threadIdx.x == 0) {
            double t = 0.0;
            for (int k = 0; k < PSA_THREADS / 32; ++k) t += warp_s[k];
            out[0] = (float)(t / (double)n);
            psa_counter = 0;
        }
    }
}

void reference(const float* inp, const float* target, int64_t batch_size, int64_t height, int64_t width, float* output) {
    static int num_sms = 0;
    if (num_sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
        if (num_sms <= 0) num_sms = 128;
    }
    int64_t n = batch_size * 4 * height * width;
    bool vec = ((((uintptr_t)inp) | ((uintptr_t)target)) & 15) == 0;
    int64_t work = vec ? (n + 3) / 4 : n;
    int64_t blocks = (work + PSA_THREADS - 1) / PSA_THREADS;
    int64_t maxb = (int64_t)num_sms * 4;
    if (maxb > PSA_MAX_BLOCKS) maxb = PSA_MAX_BLOCKS;
    if (blocks > maxb) blocks = maxb;
    if (blocks < 1) blocks = 1;
    if (vec) psa_kernel<true><<<(int)blocks, PSA_THREADS>>>(inp, target, n, output);
    else psa_kernel<false><<<(int)blocks, PSA_THREADS>>>(inp, target, n, output);
}
