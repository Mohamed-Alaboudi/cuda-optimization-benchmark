#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdint.h>

#define KAN_MAXG 64

// A row-major (B, Kd), Kd = I*(1+C): [silu(x) (I) | bases (I*C)]
__global__ void kan_build_A(const float* __restrict__ x, const float* __restrict__ grid,
                            int B, int I, int G, int K, float* __restrict__ A) {
    int C = G + K;
    int NG = G + 2 * K + 1;
    long long Kd = (long long)I * (1 + C);
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)B * I) return;
    int b = idx / I, i = idx % I;
    float xv = x[idx];
    const float* g = grid + (long long)i * NG;
    float gr[KAN_MAXG];
    float bs[KAN_MAXG];
    for (int j = 0; j < NG; j++) gr[j] = g[j];
    for (int j = 0; j < NG - 1; j++) bs[j] = (xv >= gr[j] && xv < gr[j + 1]) ? 1.f : 0.f;
    for (int k = 1; k <= K; k++) {
        int n = NG - 1 - k;
        for (int j = 0; j < n; j++) {
            float l = ((xv - gr[j]) / (gr[j + k] - gr[j])) * bs[j];
            float r = ((gr[j + k + 1] - xv) / (gr[j + k + 1] - gr[j + 1])) * bs[j + 1];
            bs[j] = l + r;
        }
    }
    float* row = A + (long long)b * Kd;
    row[i] = xv / (1.f + expf(-xv));
    float* br = row + I + (long long)i * C;
    for (int c = 0; c < C; c++) br[c] = bs[c];
}

// W row-major (O, Kd): [base_weight (I) | spline_weight*scaler (I*C)]
__global__ void kan_build_W(const float* __restrict__ bw, const float* __restrict__ sw,
                            const float* __restrict__ sc, int I, int O, int C, float* __restrict__ W) {
    long long Kd = (long long)I * (1 + C);
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)O * Kd) return;
    int o = idx / Kd;
    long long k = idx % Kd;
    float v;
    if (k < I) v = bw[(long long)o * I + k];
    else {
        long long t = k - I;  // = i*C + c
        long long i = t / C;
        v = sw[(long long)o * I * C + t] * sc[(long long)o * I + i];
    }
    W[idx] = v;
}

static cublasHandle_t g_h = nullptr;
static float* g_A = nullptr; static size_t g_Acap = 0;
static float* g_W = nullptr; static size_t g_Wcap = 0;

void reference(const float* x, const float* grid, const float* base_weight, const float* spline_weight,
               const float* spline_scaler, int32_t grid_size, int32_t spline_order, int32_t B, int32_t I,
               int32_t O, float* out) {
    int G = grid_size, K = spline_order, C = G + K;
    long long Kd = (long long)I * (1 + C);
    if (!g_h) {
        cublasCreate(&g_h);
        cublasSetMathMode(g_h, CUBLAS_PEDANTIC_MATH);
    }
    cublasSetStream(g_h, 0);
    size_t an = (size_t)B * Kd, wn = (size_t)O * Kd;
    if (an > g_Acap) { if (g_A) cudaFree(g_A); cudaMalloc(&g_A, an * sizeof(float)); g_Acap = an; }
    if (wn > g_Wcap) { if (g_W) cudaFree(g_W); cudaMalloc(&g_W, wn * sizeof(float)); g_Wcap = wn; }
    {
        long long n = (long long)B * I;
        int t = 128;
        kan_build_A<<<(unsigned)((n + t - 1) / t), t>>>(x, grid, B, I, G, K, g_A);
    }
    {
        long long n = (long long)O * Kd;
        int t = 256;
        kan_build_W<<<(unsigned)((n + t - 1) / t), t>>>(base_weight, spline_weight, spline_scaler, I, O, C, g_W);
    }
    float alpha = 1.f, beta = 0.f;
    cublasSgemm(g_h, CUBLAS_OP_T, CUBLAS_OP_N, O, B, (int)Kd, &alpha, g_W, (int)Kd, g_A, (int)Kd, &beta, out, O);
}
