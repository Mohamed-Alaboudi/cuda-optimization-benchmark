#include <cuda_runtime.h>
#include <stdint.h>

#define PASM_THREADS 512
#define PASM_MAX_BLOCKS 1024

__device__ float g_pasm_partials[PASM_MAX_BLOCKS];
__device__ unsigned int g_pasm_counter = 0;

__device__ __forceinline__ float pasm_val(float x, float y) {
    float d = x - y;
    float a = fabsf(d);
    float p = a * a;
    float s = p + 1e-6f;
    return sqrtf(s);
}

__device__ __forceinline__ float pasm_block_sum(float v, float* sh) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float r = 0.f;
    if (w == 0) {
        int nw = blockDim.x >> 5;
        r = (lane < nw) ? sh[lane] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    }
    return r;  // valid in thread 0
}

__global__ void __launch_bounds__(PASM_THREADS)
pasm_kernel(const float* __restrict__ a, const float* __restrict__ b, int64_t n, bool vec, float* out) {
    __shared__ float sh[32];
    __shared__ bool is_last;
    float acc = 0.f;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    if (vec) {
        int64_t n4 = n >> 2;
        const float4* a4 = reinterpret_cast<const float4*>(a);
        const float4* b4 = reinterpret_cast<const float4*>(b);
        int64_t i = tid;
        for (; i + 3 * stride < n4; i += 4 * stride) {
            float4 x[4], y[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) { x[k] = __ldcs(a4 + i + k * stride); y[k] = __ldcs(b4 + i + k * stride); }
            #pragma unroll
            for (int k = 0; k < 4; k++)
                acc += (pasm_val(x[k].x, y[k].x) + pasm_val(x[k].y, y[k].y)) +
                       (pasm_val(x[k].z, y[k].z) + pasm_val(x[k].w, y[k].w));
        }
        for (; i < n4; i += stride) {
            float4 x = __ldcs(a4 + i), y = __ldcs(b4 + i);
            acc += (pasm_val(x.x, y.x) + pasm_val(x.y, y.y)) + (pasm_val(x.z, y.z) + pasm_val(x.w, y.w));
        }
        for (int64_t j = (n4 << 2) + tid; j < n; j += stride) acc += pasm_val(a[j], b[j]);
    } else {
        for (int64_t j = tid; j < n; j += stride) acc += pasm_val(a[j], b[j]);
    }
    float bs = pasm_block_sum(acc, sh);
    if (threadIdx.x == 0) {
        g_pasm_partials[blockIdx.x] = bs;
        __threadfence();
        unsigned int t = atomicAdd(&g_pasm_counter, 1u);
        is_last = (t == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last) {
        __threadfence();
        double s = 0.0;
        for (int k = threadIdx.x; k < gridDim.x; k += blockDim.x)
            s += (double)((volatile float*)g_pasm_partials)[k];
        // block reduce in double
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        __shared__ double shd[32];
        int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
        __syncthreads();
        if (lane == 0) shd[w] = s;
        __syncthreads();
        if (w == 0) {
            int nw = blockDim.x >> 5;
            double r = (lane < nw) ? shd[lane] : 0.0;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
            if (lane == 0) {
                out[0] = (float)(r / (double)n);
                g_pasm_counter = 0;
            }
        }
    }
}

void reference(const float* inp, const float* target, int64_t batch_size, int64_t height, int64_t width, float* output) {
    int64_t n = batch_size * 4 * height * width;
    static int num_sms = 0;
    if (num_sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
        if (num_sms <= 0) num_sms = 128;
    }
    bool vec = ((reinterpret_cast<uintptr_t>(inp) | reinterpret_cast<uintptr_t>(target)) & 15) == 0;
    int64_t work = vec ? (n + 3) / 4 : n;
    int64_t blocks = (work + PASM_THREADS - 1) / PASM_THREADS;
    int64_t maxb = (int64_t)num_sms * 4;
    if (maxb > PASM_MAX_BLOCKS) maxb = PASM_MAX_BLOCKS;
    if (blocks > maxb) blocks = maxb;
    if (blocks < 1) blocks = 1;
    pasm_kernel<<<(int)blocks, PASM_THREADS>>>(inp, target, n, vec, output);
}
