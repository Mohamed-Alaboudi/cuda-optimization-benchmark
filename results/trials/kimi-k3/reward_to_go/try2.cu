#include <cstdint>
#include <cuda_runtime.h>

// One thread per batch row: reverse sequential scan over time.
// out[t] = reward[t] + gamma * next, next = 0 if t == T-1 or done[t].
__global__ void r2g_kernel(const float* __restrict__ reward,
                           const bool* __restrict__ done,
                           float gamma, int B, int T,
                           float* __restrict__ output) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const long off = (long)b * T;
    const float* r = reward + off;
    const bool*  d = done + off;
    float* o = output + off;
    float next = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        next = d[t] ? r[t] : fmaf(gamma, next, r[t]);
        o[t] = next;
    }
}

void reference(const float* reward, const bool* done, float gamma,
               int32_t B, int32_t T, float* output) {
    if (B <= 0 || T <= 0) return;
    int threads = 128;
    int blocks = (B + threads - 1) / threads;
    r2g_kernel<<<blocks, threads>>>(reward, done, gamma, B, T, output);
}
