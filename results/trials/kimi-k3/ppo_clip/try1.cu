#include <cuda_runtime.h>
#include <math.h>

#define TB 256

// Monotonic float <-> unsigned maps for atomicMax
static __device__ __forceinline__ unsigned fenc(float x) {
    unsigned u = __float_as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static __device__ __forceinline__ float fdec(unsigned u) {
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u);
}

// Pass 1: sums of gain, clip count, kl term, ratio; maxes of ratio and lw
__global__ void ppo_k1(const float* __restrict__ cur, const float* __restrict__ oldp,
                       const float* __restrict__ adv, long N, float lo, float hi,
                       double* acc, unsigned* mx) {
    double sg = 0, sc = 0, sk = 0, sr = 0;
    float rm = 0.f, lm = -3.0e38f;
    long stride = (long)gridDim.x * blockDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        float lw = cur[i] - oldp[i];
        float r = expf(lw);
        float lc = fminf(fmaxf(lw, lo), hi);
        float rc = expf(lc);
        float a = adv[i];
        sg += (double)fminf(r * a, rc * a);
        if (lc != lw) sc += 1.0;
        sk += (double)((r - 1.0f) - lw);
        sr += (double)r;
        rm = fmaxf(rm, r);
        lm = fmaxf(lm, lw);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sg += __shfl_down_sync(0xffffffffu, sg, o);
        sc += __shfl_down_sync(0xffffffffu, sc, o);
        sk += __shfl_down_sync(0xffffffffu, sk, o);
        sr += __shfl_down_sync(0xffffffffu, sr, o);
        rm = fmaxf(rm, __shfl_down_sync(0xffffffffu, rm, o));
        lm = fmaxf(lm, __shfl_down_sync(0xffffffffu, lm, o));
    }
    __shared__ double ssg[TB / 32], ssc[TB / 32], ssk[TB / 32], ssr[TB / 32];
    __shared__ float srm[TB / 32], slm[TB / 32];
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) { ssg[w] = sg; ssc[w] = sc; ssk[w] = sk; ssr[w] = sr; srm[w] = rm; slm[w] = lm; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int nw = blockDim.x >> 5;
        for (int i = 1; i < nw; i++) {
            ssg[0] += ssg[i]; ssc[0] += ssc[i]; ssk[0] += ssk[i]; ssr[0] += ssr[i];
            srm[0] = fmaxf(srm[0], srm[i]); slm[0] = fmaxf(slm[0], slm[i]);
        }
        atomicAdd(&acc[0], ssg[0]);
        atomicAdd(&acc[1], ssc[0]);
        atomicAdd(&acc[2], ssk[0]);
        atomicAdd(&acc[3], ssr[0]);
        atomicMax(&mx[0], fenc(srm[0]));
        atomicMax(&mx[1], fenc(slm[0]));
    }
}

// Pass 2: sum(exp(lw - m)) and sum(exp(2*(lw - m))) for the logsumexps
__global__ void ppo_k2(const float* __restrict__ cur, const float* __restrict__ oldp,
                       long N, const unsigned* mx, double* acc) {
    double m = (double)fdec(mx[1]);
    double s1 = 0, s2 = 0;
    long stride = (long)gridDim.x * blockDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        double d = (double)(cur[i] - oldp[i]) - m;
        s1 += exp(d);
        s2 += exp(2.0 * d);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s1 += __shfl_down_sync(0xffffffffu, s1, o);
        s2 += __shfl_down_sync(0xffffffffu, s2, o);
    }
    __shared__ double ss1[TB / 32], ss2[TB / 32];
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) { ss1[w] = s1; ss2[w] = s2; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int nw = blockDim.x >> 5;
        for (int i = 1; i < nw; i++) { ss1[0] += ss1[i]; ss2[0] += ss2[i]; }
        atomicAdd(&acc[4], ss1[0]);
        atomicAdd(&acc[5], ss2[0]);
    }
}

// Finalize
__global__ void ppo_k3(const double* acc, const unsigned* mx, double N, float* out) {
    double m = (double)fdec(mx[1]);
    double lse1 = m + log(acc[4]);
    double lse2 = 2.0 * m + log(acc[5]);
    double ess = exp(2.0 * lse1 - lse2) / N;
    out[0] = (float)(-acc[0] / N);
    out[1] = (float)(acc[1] / N);
    out[2] = (float)(acc[2] / N);
    out[3] = (float)ess;
    out[4] = fdec(mx[0]);
    out[5] = (float)(acc[3] / N);
}

void reference(const int32_t B, const int32_t T, const int32_t H,
               const float* current_log_prob, const float* old_log_prob,
               const float* advantage, const float clip_epsilon, float* output) {
    static double* acc = nullptr;
    static unsigned* mx = nullptr;
    if (!acc) {
        cudaMalloc(&acc, 8 * sizeof(double));
        mx = (unsigned*)(acc + 6);
    }
    long N = (long)B * T * H;
    cudaMemsetAsync(acc, 0, 8 * sizeof(double));
    float lo = log1pf(-clip_epsilon), hi = log1pf(clip_epsilon);
    long need = (N + TB - 1) / TB;
    int blocks = (int)(need < 1024 ? need : 1024);
    ppo_k1<<<blocks, TB>>>(current_log_prob, old_log_prob, advantage, N, lo, hi, acc, mx);
    ppo_k2<<<blocks, TB>>>(current_log_prob, old_log_prob, N, mx, acc);
    ppo_k3<<<1, 1>>>(acc, mx, (double)N, output);
}
