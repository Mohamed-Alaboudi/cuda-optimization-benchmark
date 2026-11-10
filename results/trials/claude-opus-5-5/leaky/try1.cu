#include <cuda_runtime.h>
#include <stdint.h>

#define UNR 8

__global__ void lif_kernel(const float* __restrict__ x, const float* __restrict__ mem0,
                           const float* __restrict__ beta, const float* __restrict__ th,
                           int T, int H, long long BH, float* __restrict__ spikes,
                           float* __restrict__ final_mem) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BH) return;
    int h = (int)(i % H);
    float b = beta[h];
    b = fminf(fmaxf(b, 0.0f), 1.0f);
    if (beta[h] != beta[h]) b = beta[h];  // keep NaN like torch.clamp
    float thr = th[h];
    float mem = mem0[i];
    const float* xp = x + i;
    float* sp = spikes + i;
    int t = 0;
    for (; t + UNR <= T; t += UNR) {
        float v[UNR];
#pragma unroll
        for (int k = 0; k < UNR; ++k) v[k] = __ldcs(xp + (long long)(t + k) * BH);
#pragma unroll
        for (int k = 0; k < UNR; ++k) {
            float reset = (mem >= thr) ? 1.0f : 0.0f;
            float m = __fadd_rn(__fmul_rn(b, mem), v[k]);
            mem = __fsub_rn(m, __fmul_rn(reset, thr));
            __stcs(sp + (long long)(t + k) * BH, (mem >= thr) ? 1.0f : 0.0f);
        }
    }
    for (; t < T; ++t) {
        float v = __ldcs(xp + (long long)t * BH);
        float reset = (mem >= thr) ? 1.0f : 0.0f;
        float m = __fadd_rn(__fmul_rn(b, mem), v);
        mem = __fsub_rn(m, __fmul_rn(reset, thr));
        __stcs(sp + (long long)t * BH, (mem >= thr) ? 1.0f : 0.0f);
    }
    final_mem[i] = mem;
}

void reference(const float* input_current, const float* mem0, const float* beta,
               const float* threshold, int32_t T, int32_t B, int32_t H, float* spikes,
               float* final_mem) {
    long long BH = (long long)B * H;
    if (BH <= 0) return;
    int threads = 256;
    long long blocks = (BH + threads - 1) / threads;
    lif_kernel<<<(unsigned)blocks, threads>>>(input_current, mem0, beta, threshold, T, H, BH,
                                              spikes, final_mem);
}
