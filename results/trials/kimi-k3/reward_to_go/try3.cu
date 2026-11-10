// Batched discounted reward-to-go with done flags.
// One block per row; each thread owns a contiguous chunk of the row.
// Pass 1: per-chunk affine transform (A, C): out[chunk_start] = A * next_after_chunk + C.
// Shared-memory suffix scan composes transforms so each thread learns the value
// entering its chunk from the right.
// Pass 2: re-run the recurrence with the correct seed and write outputs.
// Internal math in fp64 for accuracy; output stored as fp32.

#include <cstdint>

__global__ void r2g_kernel(const float* __restrict__ reward,
                           const bool* __restrict__ done,
                           float gamma, int T,
                           float* __restrict__ output) {
    const int P = blockDim.x;
    const int tid = threadIdx.x;
    const long long row = blockIdx.x;
    const float* r = reward + row * (long long)T;
    const bool* d = done + row * (long long)T;
    float* o = output + row * (long long)T;

    const int K = (T + P - 1) / P;
    int start = tid * K;
    int end = start + K;
    if (end > T) end = T;
    if (start > T) start = T;

    const double gd = (double)gamma;
    double A = 1.0, C = 0.0;
    for (int t = end - 1; t >= start; --t) {
        double a = d[t] ? 0.0 : gd;
        C = (double)r[t] + a * C;
        A *= a;
    }

    __shared__ double sA[256];
    __shared__ double sC[256];
    sA[tid] = A;
    sC[tid] = C;
    __syncthreads();

    for (int off = 1; off < P; off <<= 1) {
        double a2 = 1.0, c2 = 0.0;
        if (tid + off < P) { a2 = sA[tid + off]; c2 = sC[tid + off]; }
        double a1 = sA[tid], c1 = sC[tid];
        __syncthreads();
        sA[tid] = a1 * a2;
        sC[tid] = a1 * c2 + c1;
        __syncthreads();
    }

    double next = (tid + 1 < P) ? sC[tid + 1] : 0.0;
    for (int t = end - 1; t >= start; --t) {
        double a = d[t] ? 0.0 : gd;
        next = (double)r[t] + a * next;
        o[t] = (float)next;
    }
}

void reference(const float* reward, const bool* done, float gamma, int32_t B, int32_t T, float* output) {
    if (B <= 0 || T <= 0) return;
    r2g_kernel<<<B, 256>>>(reward, done, gamma, T, output);
}
