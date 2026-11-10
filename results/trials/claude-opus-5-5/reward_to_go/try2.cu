#include <cuda_runtime.h>
#include <cstdint>

#define NT 256
#define EPT 8
#define TILE (NT * EPT)

__global__ void __launch_bounds__(NT) rtg_kernel(const float* __restrict__ reward,
                                                 const bool* __restrict__ done,
                                                 float gamma, int T, float* __restrict__ output) {
    __shared__ float s_r[TILE];
    __shared__ unsigned char s_d[TILE];
    __shared__ double s_v[NT + 1];
    __shared__ double s_wc[NT / 32], s_wr[NT / 32];

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const long long rowoff = (long long)blockIdx.x * T;
    const float* rr = reward + rowoff;
    const unsigned char* dd = reinterpret_cast<const unsigned char*>(done) + rowoff;
    float* oo = output + rowoff;
    const double g = (double)gamma;

    double carry = 0.0;
    for (int end = T; end > 0; end -= TILE) {
        const int start = end - TILE;  // may be negative
        for (int i = tid; i < TILE; i += NT) {
            int t = start + i;
            if (t >= 0) { s_r[i] = rr[t]; s_d[i] = dd[t]; }
            else { s_r[i] = 0.f; s_d[i] = 0; }
        }
        __syncthreads();

        // local composition over this thread's chunk, right to left
        const int base = tid * EPT;
        double C = 1.0, R = 0.0;  // f(x) = R + C*x
#pragma unroll
        for (int k = EPT - 1; k >= 0; --k) {
            double c = s_d[base + k] ? 0.0 : g;
            double r = (double)s_r[base + k];
            // new = f_k o old : r + c*(R + C x)
            R = r + c * R;
            C = c * C;
        }
        // warp inclusive suffix scan
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            double C2 = __shfl_down_sync(0xffffffffu, C, off);
            double R2 = __shfl_down_sync(0xffffffffu, R, off);
            if (lane + off < 32) { R = R + C * R2; C = C * C2; }
        }
        if (lane == 0) { s_wc[warp] = C; s_wr[warp] = R; }
        __syncthreads();
        double x = carry;
        for (int w = NT / 32 - 1; w > warp; --w) x = s_wr[w] + s_wc[w] * x;
        s_v[tid] = R + C * x;
        if (tid == 0) s_v[NT] = carry;
        __syncthreads();
        double nxt = s_v[tid + 1];
        const double newcarry = s_v[0];
#pragma unroll
        for (int k = EPT - 1; k >= 0; --k) {
            double c = s_d[base + k] ? 0.0 : g;
            nxt = (double)s_r[base + k] + c * nxt;
            s_r[base + k] = (float)nxt;
        }
        __syncthreads();
        for (int i = tid; i < TILE; i += NT) {
            int t = start + i;
            if (t >= 0) oo[t] = s_r[i];
        }
        carry = newcarry;
        __syncthreads();
    }
}

void reference(const float* reward, const bool* done, float gamma, int32_t B, int32_t T, float* output) {
    if (B <= 0 || T <= 0) return;
    rtg_kernel<<<B, NT>>>(reward, done, gamma, T, output);
}
