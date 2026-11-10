#include <cstdint>
#include <cuda_runtime.h>

__global__ void lif_kernel(const float* __restrict__ x,
                           const float* __restrict__ mem0,
                           const float* __restrict__ beta,
                           const float* __restrict__ thr,
                           int T, int B, int H,
                           float* __restrict__ spikes,
                           float* __restrict__ final_mem) {
    int N = B * H;
    int bh = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= N) return;
    int h = bh % H;
    // clamp beta to [0, 1]
    float b = fminf(fmaxf(beta[h], 0.0f), 1.0f);
    float th = thr[h];
    float mem = mem0[bh];
    const float* xp = x + bh;
    float* sp = spikes + bh;
    for (int t = 0; t < T; ++t) {
        float reset = (mem >= th) ? 1.0f : 0.0f;
        // match eager fp32 op order exactly (no fma contraction):
        // mem = ((beta * mem) + x[t]) - (reset * threshold)
        float r1 = __fmul_rn(b, mem);
        float r2 = __fadd_rn(r1, xp[(size_t)t * N]);
        float r3 = __fmul_rn(reset, th);
        mem = __fsub_rn(r2, r3);
        sp[(size_t)t * N] = (mem >= th) ? 1.0f : 0.0f;
    }
    final_mem[bh] = mem;
}

void reference(const float* input_current, const float* mem0, const float* beta,
               const float* threshold, int32_t T, int32_t B, int32_t H,
               float* spikes, float* final_mem) {
    int N = B * H;
    if (N <= 0 || T <= 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    lif_kernel<<<blocks, threads>>>(input_current, mem0, beta, threshold,
                                    T, B, H, spikes, final_mem);
}
