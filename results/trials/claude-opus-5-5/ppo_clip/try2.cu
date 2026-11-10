#include <cstdint>
#include <cuda_runtime.h>
#include <math.h>

#define PPO_THREADS 256
#define PPO_MAXB 2048

__device__ double g_ppo_part[PPO_MAXB * 6];
__device__ unsigned int g_ppo_counter = 0;

struct PpoAcc {
    float gain, clip, kl, s1, s2, mx;
};

__device__ __forceinline__ void ppo_elem(float c, float o, float a, double lo, double hi,
                                         float rlo, float rhi, PpoAcc& acc) {
    double lwd = (double)c - (double)o;   // exact difference
    float lw = (float)lwd;
    float r = expf(lw);
    float rc = fminf(fmaxf(r, rlo), rhi);
    acc.gain += fminf(r * a, rc * a);
    acc.clip += (lwd < lo || lwd > hi) ? 1.f : 0.f;
    acc.kl += expm1f(lw) - lw;
    acc.s1 += r;
    acc.s2 += r * r;
    acc.mx = fmaxf(acc.mx, r);
}

__global__ void __launch_bounds__(PPO_THREADS)
ppo_kernel(const float* __restrict__ cur, const float* __restrict__ old, const float* __restrict__ adv,
           long long n, double lo, double hi, float rlo, float rhi, int vec, float* __restrict__ out) {
    PpoAcc acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    long long start_tail = 0;
    if (vec) {
        long long n4 = n >> 2;
        const float4* c4 = reinterpret_cast<const float4*>(cur);
        const float4* o4 = reinterpret_cast<const float4*>(old);
        const float4* a4 = reinterpret_cast<const float4*>(adv);
        for (long long i = tid; i < n4; i += stride) {
            float4 c = __ldg(c4 + i), o = __ldg(o4 + i), a = __ldg(a4 + i);
            ppo_elem(c.x, o.x, a.x, lo, hi, rlo, rhi, acc);
            ppo_elem(c.y, o.y, a.y, lo, hi, rlo, rhi, acc);
            ppo_elem(c.z, o.z, a.z, lo, hi, rlo, rhi, acc);
            ppo_elem(c.w, o.w, a.w, lo, hi, rlo, rhi, acc);
        }
        start_tail = n4 << 2;
    }
    for (long long i = start_tail + tid; i < n; i += stride)
        ppo_elem(cur[i], old[i], adv[i], lo, hi, rlo, rhi, acc);

    double v[6] = {(double)acc.gain, (double)acc.clip, (double)acc.kl, (double)acc.s1, (double)acc.s2,
                   (double)acc.mx};
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int k = 0; k < 5; k++) v[k] += __shfl_down_sync(0xffffffffu, v[k], off);
        v[5] = fmax(v[5], __shfl_down_sync(0xffffffffu, v[5], off));
    }
    __shared__ double sh[PPO_THREADS / 32][6];
    __shared__ bool is_last;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0)
        for (int k = 0; k < 6; k++) sh[warp][k] = v[k];
    __syncthreads();
    if (threadIdx.x == 0) {
        double t[6];
        for (int k = 0; k < 6; k++) t[k] = sh[0][k];
        for (int w = 1; w < PPO_THREADS / 32; w++) {
            for (int k = 0; k < 5; k++) t[k] += sh[w][k];
            t[5] = fmax(t[5], sh[w][5]);
        }
        for (int k = 0; k < 6; k++) g_ppo_part[blockIdx.x * 6 + k] = t[k];
        __threadfence();
        unsigned int prev = atomicAdd(&g_ppo_counter, 1u);
        is_last = (prev == gridDim.x - 1);
    }
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    // last block: fixed-order reduction of partials
    double u[6] = {0, 0, 0, 0, 0, 0};
    for (unsigned int b = threadIdx.x; b < gridDim.x; b += blockDim.x) {
        volatile double* p = g_ppo_part + b * 6;
        for (int k = 0; k < 5; k++) u[k] += p[k];
        u[5] = fmax(u[5], p[5]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int k = 0; k < 5; k++) u[k] += __shfl_down_sync(0xffffffffu, u[k], off);
        u[5] = fmax(u[5], __shfl_down_sync(0xffffffffu, u[5], off));
    }
    __syncthreads();
    if (lane == 0)
        for (int k = 0; k < 6; k++) sh[warp][k] = u[k];
    __syncthreads();
    if (threadIdx.x == 0) {
        double t[6];
        for (int k = 0; k < 6; k++) t[k] = sh[0][k];
        for (int w = 1; w < PPO_THREADS / 32; w++) {
            for (int k = 0; k < 5; k++) t[k] += sh[w][k];
            t[5] = fmax(t[5], sh[w][5]);
        }
        double N = (double)n;
        out[0] = (float)(-t[0] / N);
        out[1] = (float)(t[1] / N);
        out[2] = (float)(t[2] / N);
        out[3] = (float)((t[3] * t[3]) / t[4] / N);
        out[4] = (float)t[5];
        out[5] = (float)(t[3] / N);
        g_ppo_counter = 0;
    }
}

void reference(const int32_t B, const int32_t T, const int32_t H, const float* current_log_prob,
               const float* old_log_prob, const float* advantage, const float clip_epsilon, float* output) {
    long long n = (long long)B * T * H;
    double eps = (double)clip_epsilon;
    double lo = log1p(-eps), hi = log1p(eps);
    float rlo = (float)exp(lo), rhi = (float)exp(hi);
    int vec = ((((uintptr_t)current_log_prob) | ((uintptr_t)old_log_prob) | ((uintptr_t)advantage)) & 15) == 0;
    long long per_block = (long long)PPO_THREADS * 8;
    long long blocks = (n + per_block - 1) / per_block;
    if (blocks < 1) blocks = 1;
    if (blocks > PPO_MAXB) blocks = PPO_MAXB;
    ppo_kernel<<<(unsigned)blocks, PPO_THREADS>>>(current_log_prob, old_log_prob, advantage, n, lo, hi, rlo,
                                                  rhi, vec, output);
}
