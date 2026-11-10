#include <cuda_runtime.h>
#include <stdint.h>

#define PAS_THREADS 512
#define PAS_MAX_BLOCKS 2048

__device__ double g_pas_partials[PAS_MAX_BLOCKS];
__device__ unsigned int g_pas_counter = 0;

__device__ __forceinline__ float pas_val(float x, float y) {
    float d = x - y;
    float a = fabsf(d);
    float p = a * a;
    float s = p + 1e-6f;
    return sqrtf(s);
}

__device__ __forceinline__ float4 ldcs4(const float4* p) { return __ldcs(p); }

__global__ void __launch_bounds__(PAS_THREADS)
pas_kernel(const float* __restrict__ inp, const float* __restrict__ tgt, int64_t n, double inv_n, float* out) {
    const int64_t n4 = n >> 2;
    const float4* a4 = reinterpret_cast<const float4*>(inp);
    const float4* b4 = reinterpret_cast<const float4*>(tgt);
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float acc0 = 0.f, acc1 = 0.f;
    // unroll by 2 for ILP
    for (; i + stride < n4; i += 2 * stride) {
        float4 x0 = ldcs4(a4 + i), y0 = ldcs4(b4 + i);
        float4 x1 = ldcs4(a4 + i + stride), y1 = ldcs4(b4 + i + stride);
        acc0 += pas_val(x0.x, y0.x) + pas_val(x0.y, y0.y) + pas_val(x0.z, y0.z) + pas_val(x0.w, y0.w);
        acc1 += pas_val(x1.x, y1.x) + pas_val(x1.y, y1.y) + pas_val(x1.z, y1.z) + pas_val(x1.w, y1.w);
    }
    for (; i < n4; i += stride) {
        float4 x0 = ldcs4(a4 + i), y0 = ldcs4(b4 + i);
        acc0 += pas_val(x0.x, y0.x) + pas_val(x0.y, y0.y) + pas_val(x0.z, y0.z) + pas_val(x0.w, y0.w);
    }
    // tail (n not multiple of 4)
    int64_t t = (n4 << 2) + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; t < n; t += stride) acc0 += pas_val(inp[t], tgt[t]);

    double acc = (double)acc0 + (double)acc1;
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ double sm[PAS_THREADS / 32];
    __shared__ bool is_last;
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) sm[w] = acc;
    __syncthreads();
    if (w == 0) {
        acc = (lane < PAS_THREADS / 32) ? sm[lane] : 0.0;
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (lane == 0) {
            g_pas_partials[blockIdx.x] = acc;
            __threadfence();
            unsigned int prev = atomicAdd(&g_pas_counter, 1u);
            is_last = (prev == gridDim.x - 1);
        }
    }
    __syncthreads();
    if (is_last) {
        __threadfence();
        double s = 0.0;
        for (int k = threadIdx.x; k < (int)gridDim.x; k += blockDim.x)
            s += ((volatile double*)g_pas_partials)[k];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        __syncthreads();
        if (lane == 0) sm[w] = s;
        __syncthreads();
        if (w == 0) {
            s = (lane < PAS_THREADS / 32) ? sm[lane] : 0.0;
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
            if (lane == 0) {
                out[0] = (float)(s * inv_n);
                g_pas_counter = 0;
            }
        }
    }
}

void reference(const float* inp, const float* target, int64_t batch_size, int64_t height, int64_t width, float* output) {
    int64_t n = batch_size * 4 * height * width;
    static int nsm = 0;
    if (nsm == 0) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 132;
    }
    int64_t n4 = n / 4;
    int64_t want = (n4 + PAS_THREADS - 1) / PAS_THREADS;
    int64_t maxb = (int64_t)nsm * 4;
    if (maxb > PAS_MAX_BLOCKS) maxb = PAS_MAX_BLOCKS;
    if (want > maxb) want = maxb;
    if (want < 1) want = 1;
    double inv_n = n > 0 ? 1.0 / (double)n : __builtin_nan("");
    pas_kernel<<<(int)want, PAS_THREADS>>>(inp, target, n, inv_n, output);
}
