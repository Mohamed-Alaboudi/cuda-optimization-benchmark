#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#define PPO_THREADS 256
#define PPO_MAXBLOCKS 2048

__device__ double g_part[5][PPO_MAXBLOCKS];
__device__ float g_pmax[PPO_MAXBLOCKS];
__device__ unsigned int g_counter = 0;

struct Acc {
    float gain, clip, kl, s1, s2, mx;
};

__device__ __forceinline__ void proc(float c, float o, float a, double lo_d, double hi_d,
                                     float lo_f, float hi_f, Acc& acc) {
    float lw = c - o;
    double lwd = (double)c - (double)o;
    bool clipped = (lwd < lo_d) || (lwd > hi_d);
    float r = expf(lw);
    float lwc = fminf(fmaxf(lw, lo_f), hi_f);
    float rc = clipped ? expf(lwc) : r;
    float g = fminf(r * a, rc * a);
    acc.gain += g;
    acc.clip += clipped ? 1.f : 0.f;
    acc.kl += expm1f(lw) - lw;
    acc.s1 += r;
    acc.s2 += r * r;
    acc.mx = fmaxf(acc.mx, r);
}

__global__ void __launch_bounds__(PPO_THREADS)
ppo_kernel(long long n, const float* __restrict__ cur, const float* __restrict__ old,
           const float* __restrict__ adv, double lo_d, double hi_d, float lo_f, float hi_f,
           float* __restrict__ out, int vec) {
    Acc acc = {0.f, 0.f, 0.f, 0.f, 0.f, -INFINITY};
    double dg = 0, dc = 0, dk = 0, d1 = 0, d2 = 0;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    if (vec) {
        long long n4 = n >> 2;
        const float4* c4 = (const float4*)cur;
        const float4* o4 = (const float4*)old;
        const float4* a4 = (const float4*)adv;
        int cnt = 0;
        for (long long i = tid; i < n4; i += stride) {
            float4 c = __ldg(c4 + i), o = __ldg(o4 + i), a = __ldg(a4 + i);
            proc(c.x, o.x, a.x, lo_d, hi_d, lo_f, hi_f, acc);
            proc(c.y, o.y, a.y, lo_d, hi_d, lo_f, hi_f, acc);
            proc(c.z, o.z, a.z, lo_d, hi_d, lo_f, hi_f, acc);
            proc(c.w, o.w, a.w, lo_d, hi_d, lo_f, hi_f, acc);
            if (++cnt == 4) {
                cnt = 0;
                dg += acc.gain; dc += acc.clip; dk += acc.kl; d1 += acc.s1; d2 += acc.s2;
                acc.gain = acc.clip = acc.kl = acc.s1 = acc.s2 = 0.f;
            }
        }
        for (long long i = (n4 << 2) + tid; i < n; i += stride)
            proc(cur[i], old[i], adv[i], lo_d, hi_d, lo_f, hi_f, acc);
    } else {
        int cnt = 0;
        for (long long i = tid; i < n; i += stride) {
            proc(cur[i], old[i], adv[i], lo_d, hi_d, lo_f, hi_f, acc);
            if (++cnt == 16) {
                cnt = 0;
                dg += acc.gain; dc += acc.clip; dk += acc.kl; d1 += acc.s1; d2 += acc.s2;
                acc.gain = acc.clip = acc.kl = acc.s1 = acc.s2 = 0.f;
            }
        }
    }
    dg += acc.gain; dc += acc.clip; dk += acc.kl; d1 += acc.s1; d2 += acc.s2;
    float mx = acc.mx;

    // warp reduce
    for (int off = 16; off > 0; off >>= 1) {
        dg += __shfl_down_sync(0xffffffffu, dg, off);
        dc += __shfl_down_sync(0xffffffffu, dc, off);
        dk += __shfl_down_sync(0xffffffffu, dk, off);
        d1 += __shfl_down_sync(0xffffffffu, d1, off);
        d2 += __shfl_down_sync(0xffffffffu, d2, off);
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
    }
    __shared__ double sd[5][PPO_THREADS / 32];
    __shared__ float sm[PPO_THREADS / 32];
    __shared__ bool amLast;
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) {
        sd[0][w] = dg; sd[1][w] = dc; sd[2][w] = dk; sd[3][w] = d1; sd[4][w] = d2; sm[w] = mx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double t[5] = {0, 0, 0, 0, 0};
        float m = -INFINITY;
        for (int k = 0; k < PPO_THREADS / 32; k++) {
            for (int j = 0; j < 5; j++) t[j] += sd[j][k];
            m = fmaxf(m, sm[k]);
        }
        for (int j = 0; j < 5; j++) g_part[j][blockIdx.x] = t[j];
        g_pmax[blockIdx.x] = m;
        __threadfence();
        unsigned int prev = atomicAdd(&g_counter, 1u);
        amLast = (prev == gridDim.x - 1);
    }
    __syncthreads();
    if (!amLast) return;
    __threadfence();
    // last block: reduce partials
    double t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    float m = -INFINITY;
    for (int b = threadIdx.x; b < (int)gridDim.x; b += blockDim.x) {
        t0 += ((volatile double*)g_part[0])[b];
        t1 += ((volatile double*)g_part[1])[b];
        t2 += ((volatile double*)g_part[2])[b];
        t3 += ((volatile double*)g_part[3])[b];
        t4 += ((volatile double*)g_part[4])[b];
        m = fmaxf(m, ((volatile float*)g_pmax)[b]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        t0 += __shfl_down_sync(0xffffffffu, t0, off);
        t1 += __shfl_down_sync(0xffffffffu, t1, off);
        t2 += __shfl_down_sync(0xffffffffu, t2, off);
        t3 += __shfl_down_sync(0xffffffffu, t3, off);
        t4 += __shfl_down_sync(0xffffffffu, t4, off);
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    }
    __syncthreads();
    if (lane == 0) {
        sd[0][w] = t0; sd[1][w] = t1; sd[2][w] = t2; sd[3][w] = t3; sd[4][w] = t4; sm[w] = m;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double t[5] = {0, 0, 0, 0, 0};
        float mm = -INFINITY;
        for (int k = 0; k < PPO_THREADS / 32; k++) {
            for (int j = 0; j < 5; j++) t[j] += sd[j][k];
            mm = fmaxf(mm, sm[k]);
        }
        double dn = (double)n;
        out[0] = (float)(-t[0] / dn);
        out[1] = (float)(t[1] / dn);
        out[2] = (float)(t[2] / dn);
        out[3] = (float)((t[3] * t[3] / t[4]) / dn);
        out[4] = mm;
        out[5] = (float)(t[3] / dn);
        g_counter = 0;
    }
}

void reference(const int32_t B, const int32_t T, const int32_t H, const float* current_log_prob,
               const float* old_log_prob, const float* advantage, const float clip_epsilon,
               float* output) {
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    long long n = (long long)B * T * H;
    double eps = (double)clip_epsilon;
    double lo_d = log1p(-eps), hi_d = log1p(eps);
    float lo_f = (float)lo_d, hi_f = (float)hi_d;
    int vec = ((((uintptr_t)current_log_prob) | ((uintptr_t)old_log_prob) |
                ((uintptr_t)advantage)) & 15) == 0;
    long long per_block = (long long)PPO_THREADS * (vec ? 16 : 4);
    long long blocks = (n + per_block - 1) / per_block;
    long long maxb = (long long)nsm * 4;
    if (maxb > PPO_MAXBLOCKS) maxb = PPO_MAXBLOCKS;
    if (blocks > maxb) blocks = maxb;
    if (blocks < 1) blocks = 1;
    ppo_kernel<<<(int)blocks, PPO_THREADS>>>(n, current_log_prob, old_log_prob, advantage, lo_d,
                                              hi_d, lo_f, hi_f, output, vec);
}
