#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#define PPO_THREADS 512
#define PPO_MAX_BLOCKS 1024

__device__ double g_part[6][PPO_MAX_BLOCKS];
__device__ unsigned int g_counter = 0;

struct Acc {
    float gain, kl, s1, s2, mx;
    int clip;
};

__device__ __forceinline__ void proc(float c, float o, float a, double lo, double hi,
                                     float flo, float fhi, Acc& A) {
    double lwd = (double)c - (double)o;
    float lw = (float)lwd;
    bool clipped = (lwd < lo) || (lwd > hi);
    float r = expf(lw);
    float lwc = fminf(fmaxf(lw, flo), fhi);
    float rc = clipped ? expf(lwc) : r;
    float g = fminf(r * a, rc * a);
    A.gain += g;
    A.clip += clipped ? 1 : 0;
    A.kl += expm1f(lw) - lw;
    A.s1 += r;
    A.s2 += r * r;
    A.mx = fmaxf(A.mx, r);
}

__device__ __forceinline__ double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ double warp_max(double v) {
    for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}

__global__ void __launch_bounds__(PPO_THREADS)
ppo_kernel(const float* __restrict__ cur, const float* __restrict__ old,
           const float* __restrict__ adv, long long N, double lo, double hi,
           float flo, float fhi, float* __restrict__ out) {
    Acc A;
    A.gain = 0.f; A.kl = 0.f; A.s1 = 0.f; A.s2 = 0.f; A.mx = -INFINITY; A.clip = 0;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    bool aligned = ((((uintptr_t)cur) | ((uintptr_t)old) | ((uintptr_t)adv)) & 15) == 0;
    long long done_n = 0;
    if (aligned) {
        long long n4 = N >> 2;
        const float4* c4 = reinterpret_cast<const float4*>(cur);
        const float4* o4 = reinterpret_cast<const float4*>(old);
        const float4* a4 = reinterpret_cast<const float4*>(adv);
        for (long long i = tid; i < n4; i += stride) {
            float4 c = __ldg(c4 + i), o = __ldg(o4 + i), a = __ldg(a4 + i);
            proc(c.x, o.x, a.x, lo, hi, flo, fhi, A);
            proc(c.y, o.y, a.y, lo, hi, flo, fhi, A);
            proc(c.z, o.z, a.z, lo, hi, flo, fhi, A);
            proc(c.w, o.w, a.w, lo, hi, flo, fhi, A);
        }
        done_n = n4 << 2;
    }
    for (long long i = done_n + tid; i < N; i += stride)
        proc(cur[i], old[i], adv[i], lo, hi, flo, fhi, A);

    double v[6] = {(double)A.gain, (double)A.clip, (double)A.kl, (double)A.s1, (double)A.s2,
                   (double)A.mx};
    __shared__ double sh[6][PPO_THREADS / 32];
    __shared__ bool is_last;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int k = 0; k < 5; ++k) v[k] = warp_sum(v[k]);
    v[5] = warp_max(v[5]);
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < 6; ++k) sh[k][warp] = v[k];
    }
    __syncthreads();
    if (warp == 0) {
        const int nw = PPO_THREADS / 32;
#pragma unroll
        for (int k = 0; k < 6; ++k) v[k] = (lane < nw) ? sh[k][lane] : (k == 5 ? -INFINITY : 0.0);
#pragma unroll
        for (int k = 0; k < 5; ++k) v[k] = warp_sum(v[k]);
        v[5] = warp_max(v[5]);
        if (lane == 0) {
#pragma unroll
            for (int k = 0; k < 6; ++k) g_part[k][blockIdx.x] = v[k];
            __threadfence();
            unsigned int t = atomicAdd(&g_counter, 1u);
            is_last = (t == gridDim.x - 1);
        }
    }
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    // last block: deterministic reduction over block partials
    double r[6] = {0, 0, 0, 0, 0, -INFINITY};
    for (int b = threadIdx.x; b < (int)gridDim.x; b += blockDim.x) {
#pragma unroll
        for (int k = 0; k < 5; ++k) r[k] += __ldcg(&g_part[k][b]);
        r[5] = fmax(r[5], __ldcg(&g_part[5][b]));
    }
#pragma unroll
    for (int k = 0; k < 5; ++k) r[k] = warp_sum(r[k]);
    r[5] = warp_max(r[5]);
    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < 6; ++k) sh[k][warp] = r[k];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = PPO_THREADS / 32;
        double s[6] = {0, 0, 0, 0, 0, -INFINITY};
        for (int w = 0; w < nw; ++w) {
            for (int k = 0; k < 5; ++k) s[k] += sh[k][w];
            s[5] = fmax(s[5], sh[5][w]);
        }
        double n = (double)N;
        out[0] = (float)(-s[0] / n);
        out[1] = (float)(s[1] / n);
        out[2] = (float)(s[2] / n);
        out[3] = (float)((s[3] * s[3]) / s[4] / n);
        out[4] = (float)s[5];
        out[5] = (float)(s[3] / n);
        g_counter = 0;
    }
}

void reference(const int32_t B, const int32_t T, const int32_t H, const float* current_log_prob,
               const float* old_log_prob, const float* advantage, const float clip_epsilon,
               float* output) {
    long long N = (long long)B * T * H;
    double eps = (double)clip_epsilon;
    double lo = log1p(-eps), hi = log1p(eps);
    float flo = (float)lo, fhi = (float)hi;
    static int sms = 0;
    if (!sms) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 100;
    }
    long long need = (N / 4 + PPO_THREADS - 1) / PPO_THREADS;
    long long blocks = need;
    long long cap = (long long)sms * 3;
    if (cap > PPO_MAX_BLOCKS) cap = PPO_MAX_BLOCKS;
    if (blocks > cap) blocks = cap;
    if (blocks < 1) blocks = 1;
    ppo_kernel<<<(int)blocks, PPO_THREADS>>>(current_log_prob, old_log_prob, advantage, N, lo, hi,
                                             flo, fhi, output);
}
