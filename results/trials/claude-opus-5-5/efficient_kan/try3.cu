#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

#define MAXK 64

__global__ void feat_kernel(const float* __restrict__ x, const float* __restrict__ grid,
                            int G, int K, int B, int I, float* __restrict__ A) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)B * I) return;
    int b = idx / I, i = idx % I;
    int C = G + K, NK = G + 2 * K + 1, Kt = I * (C + 1);
    float xv = x[idx];
    float* row = A + (long)b * Kt;
    row[i] = xv / (1.0f + expf(-xv));
    float g[MAXK], bs[MAXK];
    const float* gr = grid + (long)i * NK;
    for (int j = 0; j < NK; j++) g[j] = gr[j];
    for (int j = 0; j < NK - 1; j++) bs[j] = (xv >= g[j] && xv < g[j + 1]) ? 1.0f : 0.0f;
    for (int k = 1; k <= K; k++) {
        int n = NK - 1 - k;
        for (int j = 0; j < n; j++) {
            bs[j] = ((xv - g[j]) / (g[j + k] - g[j])) * bs[j] +
                    ((g[j + k + 1] - xv) / (g[j + k + 1] - g[j + 1])) * bs[j + 1];
        }
    }
    float* out = row + I + (long)i * C;
    for (int c = 0; c < C; c++) out[c] = bs[c];
}

__global__ void weight_kernel(const float* __restrict__ bw, const float* __restrict__ sw,
                              const float* __restrict__ sc, int C, int I, int O, float* __restrict__ W) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)O * I) return;
    int o = idx / I, i = idx % I;
    long Kt = (long)I * (C + 1);
    float* row = W + (long)o * Kt;
    row[i] = bw[idx];
    float s = sc[idx];
    const float* src = sw + idx * C;
    float* dst = row + I + (long)i * C;
    for (int c = 0; c < C; c++) dst[c] = src[c] * s;
}

static float* g_A = nullptr; static size_t g_Acap = 0;
static float* g_W = nullptr; static size_t g_Wcap = 0;
static cublasHandle_t g_h = nullptr;

static void ensure(float*& p, size_t& cap, size_t n) {
    if (n > cap) {
        if (p) cudaFree(p);
        cudaMalloc(&p, n * sizeof(float));
        cap = n;
    }
}

void reference(const float* x, const float* grid, const float* base_weight, const float* spline_weight,
               const float* spline_scaler, int32_t grid_size, int32_t spline_order, int32_t B, int32_t I,
               int32_t O, float* out) {
    int G = grid_size, K = spline_order, C = G + K;
    long Kt = (long)I * (C + 1);
    ensure(g_A, g_Acap, (size_t)B * Kt);
    ensure(g_W, g_Wcap, (size_t)O * Kt);
    if (!g_h) { cublasCreate(&g_h); cublasSetMathMode(g_h, CUBLAS_PEDANTIC_MATH); }
    cublasSetStream(g_h, 0);
    long n1 = (long)B * I, n2 = (long)O * I;
    feat_kernel<<<(n1 + 127) / 128, 128>>>(x, grid, G, K, B, I, g_A);
    weight_kernel<<<(n2 + 127) / 128, 128>>>(base_weight, spline_weight, spline_scaler, C, I, O, g_W);
    const float one = 1.0f, zero = 0.0f;
    cublasSgemm(g_h, CUBLAS_OP_T, CUBLAS_OP_N, O, B, (int)Kt, &one, g_W, (int)Kt, g_A, (int)Kt, &zero, out, O);
}
