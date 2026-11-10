#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// KANLinear forward: out = silu(x) @ base_weight^T + bases(x) @ (spline_weight*scaler)^T
// G = grid_size, K = spline_order, C = G + K, knots per feature = G + 2K + 1.

__global__ void bases_silu_kernel(const float* __restrict__ x,
                                  const float* __restrict__ grid,
                                  float* __restrict__ silu_out,   // (B, I)
                                  float* __restrict__ bases_out,  // (B, I, C)
                                  int B, int I, int C, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * I;
    if (idx >= total) return;
    int i = idx % I;
    float xv = x[idx];
    // silu
    silu_out[idx] = xv / (1.0f + expf(-xv));
    // b-spline bases via Cox-de Boor
    const int nknots = C + K + 1;  // G + 2K + 1
    const float* g = grid + (size_t)i * nknots;
    float b[32];  // nknots-1 <= 32
    int n = nknots - 1;
    #pragma unroll 4
    for (int j = 0; j < n; ++j)
        b[j] = (xv >= g[j] && xv < g[j + 1]) ? 1.0f : 0.0f;
    for (int k = 1; k <= K; ++k) {
        int nn = n - 1;
        for (int j = 0; j < nn; ++j) {
            float t1 = (xv - g[j]) / (g[j + k] - g[j]) * b[j];
            float t2 = (g[j + k + 1] - xv) / (g[j + k + 1] - g[j + 1]) * b[j + 1];
            b[j] = t1 + t2;
        }
        n = nn;
    }
    float* out = bases_out + (size_t)idx * C;
    for (int j = 0; j < C; ++j) out[j] = b[j];
}

__global__ void scale_weight_kernel(const float* __restrict__ w,      // (O, I, C)
                                    const float* __restrict__ scaler, // (O, I)
                                    float* __restrict__ out,          // (O, I*C)
                                    size_t total, int C) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = w[idx] * scaler[idx / C];
}

void reference(const float* x, const float* grid, const float* base_weight,
               const float* spline_weight, const float* spline_scaler,
               int32_t grid_size, int32_t spline_order,
               int32_t B, int32_t I, int32_t O, float* out) {
    const int G = grid_size, K = spline_order;
    const int C = G + K;
    const size_t nBI = (size_t)B * I;
    const size_t nBIC = nBI * C;
    const size_t nOIC = (size_t)O * I * C;

    // cached workspace
    static float* buf = nullptr;
    static size_t cap = 0;
    size_t need = nBI + nBIC + nOIC;
    if (need > cap) {
        if (buf) cudaFree(buf);
        size_t newcap = need + need / 2;
        cudaMalloc(&buf, newcap * sizeof(float));
        cap = newcap;
    }
    float* silu = buf;
    float* bases = buf + nBI;
    float* wscaled = buf + nBI + nBIC;

    // 1) bases + silu
    {
        int threads = 256;
        int blocks = (int)((nBI + threads - 1) / threads);
        bases_silu_kernel<<<blocks, threads>>>(x, grid, silu, bases, B, I, C, K);
    }
    // 2) scaled spline weight
    {
        int threads = 256;
        int blocks = (int)((nOIC + threads - 1) / threads);
        scale_weight_kernel<<<blocks, threads>>>(spline_weight, spline_scaler, wscaled, nOIC, C);
    }
    // 3) GEMMs via cuBLAS (column-major view of row-major data).
    // out^T (O,B) = W (O,K) row-major ^T applied as OP_T ... see comments:
    // row-major W(O,K) == col-major (K,O); opT -> (O,K).
    // row-major A(B,K) == col-major (K,B); opN -> (K,B).
    // C = out row-major (B,O) == col-major (O,B), ldc = O.
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    const float one = 1.0f, zero = 0.0f;
    // base term: out^T = base_weight(OP_T) * silu
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                O, B, I, &one, base_weight, I, silu, I, &zero, out, O);
    // spline term, accumulate
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                O, B, I * C, &one, wscaled, I * C, bases, I * C, &one, out, O);
}
