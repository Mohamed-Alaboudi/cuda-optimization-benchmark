#include <cuda_runtime.h>
#include <cstdint>

// Block-per-row chunked affine scan.
// Each position is affine in the incoming value from the right: out[t] = r[t] + a[t]*x,
// with a[t] = done[t] ? 0 : gamma. Each thread reduces a contiguous chunk to (A, C)
// such that out[lo] = A * x_hi + C, a serial pass over chunk summaries gives each
// chunk its true incoming value, then each thread re-runs its chunk and writes out.
// All accumulation in double for accuracy against the fp64 reference.

#define NT 256

__global__ void rtg_block_kernel(const float* __restrict__ reward, const bool* __restrict__ done,
                                 float gamma, int T, float* __restrict__ output) {
    const int b = blockIdx.x;
    const float* r = reward + (size_t)b * T;
    const bool* d = done + (size_t)b * T;
    float* o = output + (size_t)b * T;

    __shared__ double sA[NT];
    __shared__ double sC[NT];

    const int tid = threadIdx.x;
    const double g = (double)gamma;
    const int C = (T + NT - 1) / NT;
    const int lo = tid * C;
    const int hi = min(lo + C, T);

    // Phase 1: reduce chunk to affine coefficients (A, C).
    double A = 1.0, Cc = 0.0;
    for (int t = hi - 1; t >= lo; --t) {
        const double a = d[t] ? 0.0 : g;
        A = a * A;
        Cc = (double)r[t] + a * Cc;
    }
    sA[tid] = A;
    sC[tid] = Cc;
    __syncthreads();

    // Phase 2: serial scan over chunk summaries (chunk k's incoming = value at (k+1)*C).
    if (tid == 0) {
        double x = 0.0;
        for (int k = NT - 1; k >= 0; --k) {
            const double Ak = sA[k], Ck = sC[k];
            sA[k] = x;              // reuse sA as the incoming-value array
            x = Ak * x + Ck;
        }
    }
    __syncthreads();

    // Phase 3: re-run chunk with the true incoming value and write output.
    double x = sA[tid];
    for (int t = hi - 1; t >= lo; --t) {
        x = (double)r[t] + (d[t] ? 0.0 : g * x);
        o[t] = (float)x;
    }
}

void reference(const float* reward, const bool* done, float gamma, int32_t B, int32_t T, float* output) {
    if (B <= 0 || T <= 0) return;
    rtg_block_kernel<<<(int)B, NT>>>(reward, done, gamma, (int)T, output);
}
