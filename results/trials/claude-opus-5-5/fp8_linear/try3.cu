#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>
using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 32
#define LDS (BK + 8)

__device__ __forceinline__ void cvt8(uint2 v, __half* dst) {
    // convert 8 fp8 e4m3 bytes to 8 halves (exact)
    __half2_raw h[4];
    unsigned short s[4] = {(unsigned short)(v.x & 0xFFFF), (unsigned short)(v.x >> 16),
                           (unsigned short)(v.y & 0xFFFF), (unsigned short)(v.y >> 16)};
#pragma unroll
    for (int i = 0; i < 4; i++) h[i] = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)s[i], __NV_E4M3);
    uint4 o;
    o.x = *(unsigned*)&h[0]; o.y = *(unsigned*)&h[1]; o.z = *(unsigned*)&h[2]; o.w = *(unsigned*)&h[3];
    *(uint4*)dst = o;
}

__global__ void __launch_bounds__(256) fp8_gemm(int M, int K, int N,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ W,
    float inv, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out)
{
    __shared__ __align__(32) __half As[2][BM * LDS];
    __shared__ __align__(32) __half Bs[2][BN * LDS];
    __shared__ __align__(32) float scratch[8][256];

    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    int wm = (warp >> 2) * 64;   // 2 warps along M
    int wn = (warp & 3) * 32;    // 4 warps along N

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.f);

    // each thread loads 2 chunks of 8 bytes for A and B: chunk id c = tid + 256*r; row = c/4, kc = (c%4)*8
    uint2 ra[2], rb[2];
    auto gload = [&](int k0) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            int c = tid + 256 * r;
            int row = c >> 2, kc = (c & 3) * 8;
            int gk = k0 + kc;
            int gm = m0 + row, gn = n0 + row;
            uint2 z = make_uint2(0, 0);
            ra[r] = (gm < M && gk < K) ? __ldg((const uint2*)(X + (size_t)gm * K + gk)) : z;
            rb[r] = (gn < N && gk < K) ? __ldg((const uint2*)(W + (size_t)gn * K + gk)) : z;
        }
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            int c = tid + 256 * r;
            int row = c >> 2, kc = (c & 3) * 8;
            cvt8(ra[r], &As[buf][row * LDS + kc]);
            cvt8(rb[r], &Bs[buf][row * LDS + kc]);
        }
    };

    int nk = (K + BK - 1) / BK;
    gload(0);
    sstore(0);
    __syncthreads();
    for (int t = 0; t < nk; t++) {
        int buf = t & 1;
        if (t + 1 < nk) gload((t + 1) * BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
#pragma unroll
            for (int i = 0; i < 4; i++) wmma::load_matrix_sync(fa[i], &As[buf][(wm + i * 16) * LDS + kk], LDS);
#pragma unroll
            for (int j = 0; j < 2; j++) wmma::load_matrix_sync(fb[j], &Bs[buf][(wn + j * 16) * LDS + kk], LDS);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        if (t + 1 < nk) sstore(buf ^ 1);
        __syncthreads();
    }

    float* sc = scratch[warp];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::store_matrix_sync(sc, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            int r = lane >> 1, c0 = (lane & 1) * 8;
            int gm = m0 + wm + i * 16 + r;
            int gnb = n0 + wn + j * 16 + c0;
            if (gm < M) {
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    int gn = gnb + e;
                    if (gn < N) {
                        float v = sc[r * 16 + c0 + e] * inv + __bfloat162float(bias[gn]);
                        out[(size_t)gm * N + gn] = __float2bfloat16(v);
                    }
                }
            }
            __syncwarp();
        }
}

void reference(const int32_t M, const int32_t K, const int32_t N, const __nv_fp8_e4m3* x_fp8, float scale_x,
               const __nv_fp8_e4m3* w_fp8, float scale_w, const __nv_bfloat16* bias, __nv_bfloat16* output)
{
    float inv = (float)(1.0 / ((double)scale_x * (double)scale_w));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    fp8_gemm<<<grid, 256>>>(M, K, N, (const uint8_t*)x_fp8, (const uint8_t*)w_fp8, inv, bias, output);
}
