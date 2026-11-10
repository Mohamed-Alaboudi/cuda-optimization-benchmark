#include <cuda_runtime.h>
#include <stdint.h>

#define R2G_THREADS 256
#define R2G_EPT 4
#define R2G_TILE (R2G_THREADS * R2G_EPT)

__global__ void __launch_bounds__(R2G_THREADS)
r2g_kernel(const float* __restrict__ reward, const bool* __restrict__ done, float gamma,
           int T, float* __restrict__ output)
{
    __shared__ float s_r[R2G_TILE];
    __shared__ unsigned char s_d[R2G_TILE];
    __shared__ float s_wA[R2G_THREADS / 32];
    __shared__ float s_wB[R2G_THREADS / 32];
    __shared__ float s_wY[R2G_THREADS / 32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const long long rowOff = (long long)blockIdx.x * T;
    const float* rr = reward + rowOff;
    const unsigned char* dd = reinterpret_cast<const unsigned char*>(done) + rowOff;
    float* oo = output + rowOff;

    float carry = 0.0f;  // value of output at the start of the following tile
    const int ntiles = (T + R2G_TILE - 1) / R2G_TILE;
    for (int tile = ntiles - 1; tile >= 0; --tile) {
        const int base = tile * R2G_TILE;
        const int len = min(R2G_TILE, T - base);
        __syncthreads();
        for (int i = tid; i < len; i += R2G_THREADS) {
            s_r[i] = rr[base + i];
            s_d[i] = dd[base + i];
        }
        __syncthreads();

        // Local chunk affine map: y_start = B + A * y_after
        const int s = tid * R2G_EPT;
        const int e = min(s + R2G_EPT, len);
        float A = 1.0f, Bv = 0.0f;
        for (int i = e - 1; i >= s; --i) {
            float a = s_d[i] ? 0.0f : gamma;
            Bv = s_r[i] + a * Bv;
            A = a * A;
        }
        // inclusive suffix scan within warp
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            float oA = __shfl_down_sync(0xffffffffu, A, off);
            float oB = __shfl_down_sync(0xffffffffu, Bv, off);
            if (lane + off < 32) {
                Bv = Bv + A * oB;
                A = A * oA;
            }
        }
        if (lane == 0) { s_wA[warp] = A; s_wB[warp] = Bv; }
        __syncthreads();
        if (tid == 0) {
            float y = carry;
            for (int w = R2G_THREADS / 32 - 1; w >= 0; --w) {
                s_wY[w] = y;  // value after warp w
                y = s_wB[w] + s_wA[w] * y;
            }
            s_wB[0] = y;  // reuse: tile start value (new carry)
        }
        __syncthreads();
        const float yAfterWarp = s_wY[warp];
        const float yStart = Bv + A * yAfterWarp;
        float yAfter = __shfl_down_sync(0xffffffffu, yStart, 1);
        if (lane == 31) yAfter = yAfterWarp;
        float y = yAfter;
        for (int i = e - 1; i >= s; --i) {
            float a = s_d[i] ? 0.0f : gamma;
            y = s_r[i] + a * y;
            s_r[i] = y;
        }
        carry = s_wB[0];
        __syncthreads();
        for (int i = tid; i < len; i += R2G_THREADS) oo[base + i] = s_r[i];
    }
}

void reference(const float* reward, const bool* done, float gamma, int32_t B, int32_t T, float* output)
{
    if (B <= 0 || T <= 0) return;
    r2g_kernel<<<B, R2G_THREADS>>>(reward, done, gamma, T, output);
}
