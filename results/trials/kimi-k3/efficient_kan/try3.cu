#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Kernel: for each (b, i) compute the C = G+K B-spline bases via Cox-de Boor
// starting from indicator bases, and also write silu(x) for the base term.
// ---------------------------------------------------------------------------
__global__ void bases_silu_kernel(const float* __restrict__ x,
                                  const float* __restrict__ grid,
                                  int G, int K, int B, int I,
                                  float* __restrict__ bases,   // (B, I*C)
                                  float* __restrict__ silu_x)  // (B, I)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * I) return;
    int i = idx % I;

    const int C = G + K;
    const int nknots = G + 2 * K + 1;
    const float* g = grid + (size_t)i * nknots;
    float xe = x[idx];

    float bas[64];
    int n = nknots - 1;  // number of indicator bases
    for (int j = 0; j < n; ++j)
        bas[j] = (xe >= g[j] && xe < g[j + 1]) ? 1.0f : 0.0f;

    for (int k = 1; k <= K; ++k) {
        int nn = n - 1;
        for (int j = 0; j < nn; ++j) {
            float d1 = g[j + k] - g[j];
            float d2 = g[j + k + 1] - g[j + 1];
            float t1 = (d1 != 0.0f) ? (xe - g[j]) / d1 * bas[j] : 0.0f;
            float t2 = (d2 != 0.0f) ? (g[j + k + 1] - xe) / d2 * bas[j + 1] : 0.0f;
            bas[j] = t1 + t2;
        }
        n = nn;
    }
    // n == C now
    float* out = bases + (size_t)idx * C;
    for (int j = 0; j < C; ++j) out[j] = bas[j];

    silu_x[idx] = xe / (1.0f + __expf(-xe));
}

// ---------------------------------------------------------------------------
// Kernel: scaled_spline_weight(o, i*C+c) = spline_weight(o,i,c) * scaler(o,i)
// ---------------------------------------------------------------------------
__global__ void scale_weight_kernel(const float* __restrict__ sw,   // (O,I,C)
                                    const float* __restrict__ sc,   // (O,I)
                                    int O, int I, int C,
                                    float* __restrict__ out)        // (O, I*C)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)O * I * C;
    if (idx >= total) return;
    size_t ic = idx % ((size_t)I * C);
    size_t o = idx / ((size_t)I * C);
    size_t i = ic / C;
    out[idx] = sw[idx] * sc[o * I + i];
}

// ---------------------------------------------------------------------------
// Workspace (cached across calls; process-lifetime, never freed).
// ---------------------------------------------------------------------------
static float*  g_ws = nullptr;
static size_t  g_ws_elems = 0;
static cublasHandle_t g_handle = nullptr;

static void ensure_workspace(size_t elems)
{
    if (elems > g_ws_elems) {
        if (g_ws) cudaFree(g_ws);
        size_t cap = elems + elems / 2;
        cudaMalloc(&g_ws, cap * sizeof(float));
        g_ws_elems = cap;
    }
}

void reference(const float* x, const float* grid, const float* base_weight,
               const float* spline_weight, const float* spline_scaler,
               int32_t grid_size, int32_t spline_order,
               int32_t B, int32_t I, int32_t O, float* out)
{
    const int G = grid_size, K = spline_order;
    const int C = G + K;
    const size_t BI = (size_t)B * I;
    const size_t M = (size_t)I * C;

    if (!g_handle) cublasCreate(&g_handle);

    // workspace layout: bases (B*I*C) | silu_x (B*I) | scaled_w (O*I*C)
    size_t need = (size_t)B * M + BI + (size_t)O * M;
    ensure_workspace(need);
    float* bases   = g_ws;
    float* silu_x  = bases + (size_t)B * M;
    float* sc_w    = silu_x + BI;

    // 1) bases + silu
    {
        int threads = 256;
        int blocks = (int)((BI + threads - 1) / threads);
        bases_silu_kernel<<<blocks, threads>>>(x, grid, G, K, B, I, bases, silu_x);
    }
    // 2) scaled spline weight
    {
        size_t total = (size_t)O * M;
        int threads = 256;
        size_t blocks = (total + threads - 1) / threads;
        scale_weight_kernel<<<(int)blocks, threads>>>(spline_weight, spline_scaler, O, I, C, sc_w);
    }

    // 3) out = bases @ sc_w^T   (row-major: via column-major C^T = W * A^T)
    const float one = 1.0f, zero = 0.0f;
    cublasSgemm(g_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                O, B, (int)M,
                &one,
                sc_w, (int)M,
                bases, (int)M,
                &zero,
                out, O);
    // 4) out += silu_x @ base_weight^T
    cublasSgemm(g_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                O, B, I,
                &one,
                base_weight, I,
                silu_x, I,
                &one,
                out, O);
}
