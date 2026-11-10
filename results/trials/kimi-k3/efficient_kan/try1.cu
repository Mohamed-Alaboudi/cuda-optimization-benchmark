#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// Computes B-spline bases (B, I, C) via Cox-de Boor and silu(x) (B, I).
__global__ void bases_silu_kernel(const float* __restrict__ x,
                                  const float* __restrict__ grid,
                                  float* __restrict__ bases,
                                  float* __restrict__ silu,
                                  int B, int I, int G, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * I) return;
    int i = idx % I;
    float xv = x[idx];
    silu[idx] = xv / (1.0f + expf(-xv));  // x * sigmoid(x)

    const float* g = grid + (size_t)i * (G + 2 * K + 1);
    int n = G + 2 * K;  // number of indicator bases
    float cur[64];
    float nxt[64];
    for (int j = 0; j < n; j++) {
        cur[j] = (xv >= g[j] && xv < g[j + 1]) ? 1.0f : 0.0f;
    }
    for (int k = 1; k <= K; k++) {
        int m = n - k;
        for (int j = 0; j < m; j++) {
            float t1 = (xv - g[j]) / (g[j + k] - g[j]) * cur[j];
            float t2 = (g[j + k + 1] - xv) / (g[j + k + 1] - g[j + 1]) * cur[j + 1];
            nxt[j] = t1 + t2;
        }
        for (int j = 0; j < m; j++) cur[j] = nxt[j];
    }
    int C = G + K;
    float* out = bases + (size_t)idx * C;
    for (int j = 0; j < C; j++) out[j] = cur[j];
}

// scaled[o, i, c] = spline_weight[o, i, c] * spline_scaler[o, i]
__global__ void scale_kernel(const float* __restrict__ w,
                             const float* __restrict__ s,
                             float* __restrict__ out,
                             int total, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = w[idx] * s[idx / C];
}

static cublasHandle_t get_handle() {
    static cublasHandle_t h = nullptr;
    if (!h) {
        cublasCreate(&h);
        cublasSetStream(h, (cudaStream_t)0);
    }
    return h;
}

void reference(const float* x, const float* grid, const float* base_weight,
               const float* spline_weight, const float* spline_scaler,
               int32_t grid_size, int32_t spline_order,
               int32_t B, int32_t I, int32_t O, float* out) {
    int G = grid_size, K = spline_order;
    int C = G + K;
    size_t IC = (size_t)I * C;

    // cached workspace: bases (B*IC) | silu (B*I) | scaled weights (O*IC)
    static float* ws = nullptr;
    static size_t ws_sz = 0;
    size_t need = ((size_t)B * IC + (size_t)B * I + (size_t)O * IC) * sizeof(float);
    if (need > ws_sz) {
        if (ws) cudaFree(ws);
        cudaMalloc(&ws, need);
        ws_sz = need;
    }
    float* bases = ws;
    float* silu = bases + (size_t)B * IC;
    float* scaled = silu + (size_t)B * I;

    int t1 = B * I;
    bases_silu_kernel<<<(t1 + 255) / 256, 256>>>(x, grid, bases, silu, B, I, G, K);

    int t2 = (int)((size_t)O * IC);
    scale_kernel<<<(t2 + 255) / 256, 256>>>(spline_weight, spline_scaler, scaled, t2, C);

    cublasHandle_t h = get_handle();
    float alpha = 1.0f, beta0 = 0.0f, beta1 = 1.0f;
    // out (B x O, row-major) = silu (B x I) * base_weight^T (I x O)
    // col-major view: out^T (O x B) = base_weight (O x I) * silu^T (I x B)
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, O, B, I, &alpha,
                base_weight, I, silu, I, &beta0, out, O);
    // out += bases (B x IC) * scaled^T (IC x O)
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, O, B, (int)IC, &alpha,
                scaled, (int)IC, bases, (int)IC, &beta1, out, O);
}
