#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

#define MAXNG 40

// A[b, i*(C+1) + 0] = silu(x), A[b, i*(C+1)+1+c] = bases
__global__ void kan_feat(const float* __restrict__ x, const float* __restrict__ grid,
                         int G, int K, int B, int I, float* __restrict__ A) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * I) return;
    int i = idx % I;
    int NG = G + 2 * K + 1;
    int C = G + K;
    float xv = x[idx];
    const float* g = grid + (size_t)i * NG;
    float gr[MAXNG];
    float bs[MAXNG];
    for (int j = 0; j < NG; ++j) gr[j] = g[j];
    for (int j = 0; j < NG - 1; ++j) bs[j] = (xv >= gr[j] && xv < gr[j + 1]) ? 1.f : 0.f;
    for (int k = 1; k <= K; ++k) {
        int n = NG - 1 - k;
        for (int j = 0; j < n; ++j) {
            float l = (xv - gr[j]) / (gr[j + k] - gr[j]) * bs[j];
            float r = (gr[j + k + 1] - xv) / (gr[j + k + 1] - gr[j + 1]) * bs[j + 1];
            bs[j] = l + r;
        }
    }
    float* a = A + (size_t)idx * (C + 1);
    a[0] = xv / (1.f + expf(-xv));
    for (int c = 0; c < C; ++c) a[1 + c] = bs[c];
}

__global__ void kan_weight(const float* __restrict__ bw, const float* __restrict__ sw,
                           const float* __restrict__ sc, int C, int O, int I, float* __restrict__ W) {
    size_t total = (size_t)O * I * (C + 1);
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < total; t += (size_t)gridDim.x * blockDim.x) {
        size_t oi = t / (C + 1);
        int c = (int)(t % (C + 1));
        W[t] = (c == 0) ? bw[oi] : sw[oi * C + (c - 1)] * sc[oi];
    }
}

static float* s_buf = nullptr;
static size_t s_cap = 0;
static cublasHandle_t s_h = nullptr;

void reference(const float* x, const float* grid, const float* base_weight, const float* spline_weight,
               const float* spline_scaler, int32_t grid_size, int32_t spline_order, int32_t B, int32_t I,
               int32_t O, float* out) {
    int G = grid_size, K = spline_order;
    int C = G + K;
    int Kd = I * (C + 1);
    size_t needA = (size_t)B * Kd, needW = (size_t)O * Kd;
    size_t need = needA + needW;
    if (need > s_cap) {
        if (s_buf) cudaFree(s_buf);
        cudaMalloc(&s_buf, need * sizeof(float));
        s_cap = need;
    }
    if (!s_h) {
        cublasCreate(&s_h);
        cublasSetMathMode(s_h, CUBLAS_PEDANTIC_MATH);
        cublasSetStream(s_h, 0);
    }
    float* A = s_buf;
    float* W = s_buf + needA;
    int n = B * I;
    kan_feat<<<(n + 127) / 128, 128>>>(x, grid, G, K, B, I, A);
    size_t tw = needW;
    int blocks = (int)((tw + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    kan_weight<<<blocks, 256>>>(base_weight, spline_weight, spline_scaler, C, O, I, W);
    float alpha = 1.f, beta = 0.f;
    cublasSgemm(s_h, CUBLAS_OP_T, CUBLAS_OP_N, O, B, Kd, &alpha, W, Kd, A, Kd, &beta, out, O);
}
