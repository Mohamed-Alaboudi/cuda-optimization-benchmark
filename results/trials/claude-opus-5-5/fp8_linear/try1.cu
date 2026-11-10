#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <stdint.h>
#include <map>
#include <tuple>

namespace {

struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    cublasLtMatmulAlgo_t algo;
    bool ok = false;
};

cublasLtHandle_t g_lt = nullptr;
void* g_ws = nullptr;
const size_t WS_SIZE = 32u << 20;
__nv_bfloat16* g_conv = nullptr;
size_t g_conv_elems = 0;
std::map<std::tuple<int, int, int, int>, Plan> g_plans;

__global__ void fp8_to_bf16(const __nv_fp8_e4m3* __restrict__ in, __nv_bfloat16* __restrict__ out, size_t n8) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n8; i += stride) {
        uint2 v = reinterpret_cast<const uint2*>(in)[i];
        const __nv_fp8_e4m3* p = reinterpret_cast<const __nv_fp8_e4m3*>(&v);
        __align__(16) __nv_bfloat16 r[8];
#pragma unroll
        for (int j = 0; j < 8; j++) r[j] = __float2bfloat16(float(p[j]));
        reinterpret_cast<uint4*>(out)[i] = *reinterpret_cast<uint4*>(r);
    }
}

// Simple fallback kernel: one thread per output.
__global__ void naive_linear(int M, int K, int N, const __nv_fp8_e4m3* x, float sx, const __nv_fp8_e4m3* w,
                             float sw, const __nv_bfloat16* bias, __nv_bfloat16* out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;
    float acc = 0.f;
    for (int k = 0; k < K; k++) acc += float(x[(size_t)m * K + k]) * float(w[(size_t)n * K + k]);
    out[(size_t)m * N + n] = __float2bfloat16(acc / (sx * sw) + __bfloat162float(bias[n]));
}

Plan& get_plan(int M, int K, int N, bool fp8, const __nv_bfloat16* bias) {
    auto key = std::make_tuple(M, K, N, fp8 ? 1 : 0);
    auto it = g_plans.find(key);
    if (it != g_plans.end()) return it->second;
    Plan& p = g_plans[key];
    cudaDataType_t ab = fp8 ? CUDA_R_8F_E4M3 : CUDA_R_16BF;
    if (cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return p;
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
    cudaDataType_t bt = CUDA_R_16BF;
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    // A: W col-major K x N (transposed), B: X col-major K x M, D: N x M
    cublasLtMatrixLayoutCreate(&p.a, ab, K, N, K);
    cublasLtMatrixLayoutCreate(&p.b, ab, K, M, K);
    cublasLtMatrixLayoutCreate(&p.d, CUDA_R_16BF, N, M, N);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws = WS_SIZE;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t res[1];
    int nres = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(g_lt, p.op, p.a, p.b, p.d, p.d, pref, 1, res, &nres);
    cublasLtMatmulPreferenceDestroy(pref);
    if (st == CUBLAS_STATUS_SUCCESS && nres > 0) {
        p.algo = res[0].algo;
        p.ok = true;
    }
    return p;
}

bool run_lt(int M, int K, int N, const void* A, const void* B, float alpha, const __nv_bfloat16* bias,
            __nv_bfloat16* out, bool fp8) {
    Plan& p = get_plan(M, K, N, fp8, bias);
    if (!p.ok) return false;
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    float beta = 0.f;
    cublasStatus_t st = cublasLtMatmul(g_lt, p.op, &alpha, A, p.a, B, p.b, &beta, out, p.d, out, p.d, &p.algo,
                                       g_ws, WS_SIZE, 0);
    return st == CUBLAS_STATUS_SUCCESS;
}

}  // namespace

void reference(const int32_t M, const int32_t K, const int32_t N, const __nv_fp8_e4m3* x_fp8, float scale_x,
               const __nv_fp8_e4m3* w_fp8, float scale_w, const __nv_bfloat16* bias, __nv_bfloat16* output) {
    if (M <= 0 || N <= 0) return;
    if (!g_lt) {
        cublasLtCreate(&g_lt);
        cudaMalloc(&g_ws, WS_SIZE);
    }
    float alpha = (float)(1.0 / ((double)scale_x * (double)scale_w));
    bool aligned = (((uintptr_t)x_fp8 | (uintptr_t)w_fp8) & 15) == 0 && (((uintptr_t)output | (uintptr_t)bias) & 15) == 0;
    if (aligned && K % 16 == 0 && N % 16 == 0) {
        if (run_lt(M, K, N, w_fp8, x_fp8, alpha, bias, output, true)) return;
    }
    if (K % 8 == 0 && aligned) {
        size_t nx = (size_t)M * K, nw = (size_t)N * K;
        size_t need = nx + nw;
        if (need > g_conv_elems) {
            if (g_conv) cudaFree(g_conv);
            cudaMalloc(&g_conv, need * sizeof(__nv_bfloat16));
            g_conv_elems = need;
        }
        __nv_bfloat16* xb = g_conv;
        __nv_bfloat16* wb = g_conv + nx;
        fp8_to_bf16<<<(unsigned)((nx / 8 + 255) / 256 > 4096 ? 4096 : (nx / 8 + 255) / 256), 256>>>(x_fp8, xb, nx / 8);
        fp8_to_bf16<<<(unsigned)((nw / 8 + 255) / 256 > 4096 ? 4096 : (nw / 8 + 255) / 256), 256>>>(w_fp8, wb, nw / 8);
        if (run_lt(M, K, N, wb, xb, alpha, bias, output, false)) return;
    }
    dim3 grid((N + 127) / 128, M);
    naive_linear<<<grid, 128>>>(M, K, N, x_fp8, scale_x, w_fp8, scale_w, bias, output);
}
