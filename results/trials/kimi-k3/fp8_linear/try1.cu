// FP8 (E4M3) linear forward: output = (x_fp8/scale_x) @ (w_fp8/scale_w)^T + bias, in bf16.
// Strategy:
//  - Fast path: cuBLASLt FP8 (E4M3) GEMM with fp32 accumulate, bias epilogue,
//    alpha = 1/(scale_x*scale_w). Requires 16-byte-aligned leading dims.
//  - Fallback: convert fp8 -> fp16, cuBLASLt HGEMM (fp32 accumulate), bias epilogue.
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

static cublasLtHandle_t g_handle = nullptr;
static void*  g_ws      = nullptr;
static size_t g_ws_size = 0;
static __nv_bfloat16* g_xb   = nullptr;
static size_t  g_xb_n   = 0;
static __nv_bfloat16* g_wb   = nullptr;
static size_t  g_wb_n   = 0;

static void ensure_ws(size_t bytes) {
    if (bytes > g_ws_size) {
        if (g_ws) cudaFree(g_ws);
        size_t want = bytes < (32u << 20) ? (32u << 20) : bytes;
        if (cudaMalloc(&g_ws, want) != cudaSuccess) { g_ws = nullptr; g_ws_size = 0; return; }
        g_ws_size = want;
    }
}

static __nv_bfloat16* ensure_buf(__nv_bfloat16** p, size_t* cap, size_t n) {
    if (n > *cap) {
        if (*p) cudaFree(*p);
        if (cudaMalloc(p, n * sizeof(__nv_bfloat16)) != cudaSuccess) { *p = nullptr; *cap = 0; return nullptr; }
        *cap = n;
    }
    return *p;
}

// n8 = number of 8-element groups; total elements is a multiple of 8.
__global__ void cvt_fp8_bf16_8(const __nv_fp8_e4m3* __restrict__ in, __nv_bfloat16* __restrict__ out, long n8) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    uint2 v = reinterpret_cast<const uint2*>(in)[i];
    const __nv_fp8_storage_t* b = reinterpret_cast<const __nv_fp8_storage_t*>(&v);
    uint4 o;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        __nv_fp8x2_storage_t pk = (__nv_fp8x2_storage_t)((unsigned)b[2 * j] | ((unsigned)b[2 * j + 1] << 8));
        __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(pk, __NV_E4M3);
        __half2 hh = *reinterpret_cast<__half2*>(&r);
        h[j] = __float22bfloat162_rn(__half22float2(hh));
    }
    reinterpret_cast<uint4*>(out)[i] = o;
}

struct LtCleanup {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t Ad, Bd, Cd;
    cublasLtMatmulPreference_t pref;
    LtCleanup() : desc(nullptr), Ad(nullptr), Bd(nullptr), Cd(nullptr), pref(nullptr) {}
    ~LtCleanup() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (Cd) cublasLtMatrixLayoutDestroy(Cd);
        if (Bd) cublasLtMatrixLayoutDestroy(Bd);
        if (Ad) cublasLtMatrixLayoutDestroy(Ad);
        if (desc) cublasLtMatmulDescDestroy(desc);
    }
};

// C (M x N row-major) = op-scale stuff: computes out = alpha * (A_kn^T as (N,K)) ... see call site.
// A buffer: col-major (K, N) ld=K with transa=T  => op(A) = W (N,K) row-major
// B buffer: col-major (K, M) ld=K with transb=N  => op(B) = X^T (K,M) from x row-major (M,K)
// C/D: col-major (N, M) ld=N == row-major (M,N)
static bool lt_gemm(cudaDataType_t abtype, const void* wbuf, const void* xbuf,
                    const void* bias, void* out, int M, int K, int N, float alpha) {
    LtCleanup cl;
    if (cublasLtMatmulDescCreate(&cl.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(cl.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(cl.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(cl.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
    cublasLtMatmulDescSetAttribute(cl.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    cudaDataType_t bdt = CUDA_R_16BF;
    cublasLtMatmulDescSetAttribute(cl.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bdt, sizeof(bdt));

    if (cublasLtMatrixLayoutCreate(&cl.Ad, abtype, K, N, K) != CUBLAS_STATUS_SUCCESS) return false;
    if (cublasLtMatrixLayoutCreate(&cl.Bd, abtype, K, M, K) != CUBLAS_STATUS_SUCCESS) return false;
    if (cublasLtMatrixLayoutCreate(&cl.Cd, CUDA_R_16BF, N, M, N) != CUBLAS_STATUS_SUCCESS) return false;

    if (!g_ws) ensure_ws(32u << 20);
    float beta = 0.0f;
    cublasLtMatmulHeuristicResult_t heur;
    int nres = 0;
    const cublasLtMatmulAlgo_t* algo = nullptr;
    if (cublasLtMatmulPreferenceCreate(&cl.pref) == CUBLAS_STATUS_SUCCESS && g_ws) {
        size_t wsz = g_ws_size;
        cublasLtMatmulPreferenceSetAttribute(cl.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz));
        if (cublasLtMatmulAlgoGetHeuristic(g_handle, cl.desc, cl.Ad, cl.Bd, cl.Cd, cl.Cd,
                                           cl.pref, 1, &heur, &nres) == CUBLAS_STATUS_SUCCESS && nres > 0) {
            algo = &heur.algo;
        }
    }
    cublasStatus_t st = cublasLtMatmul(g_handle, cl.desc, &alpha, wbuf, cl.Ad, xbuf, cl.Bd,
                                       &beta, out, cl.Cd, out, cl.Cd, algo,
                                       g_ws, g_ws ? g_ws_size : 0, 0);
    return st == CUBLAS_STATUS_SUCCESS;
}

void reference(const int32_t M, const int32_t K, const int32_t N,
               const __nv_fp8_e4m3* x_fp8, float scale_x,
               const __nv_fp8_e4m3* w_fp8, float scale_w,
               const __nv_bfloat16* bias, __nv_bfloat16* output) {
    if (!g_handle) {
        if (cublasLtCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) return;
    }
    float alpha = 1.0f / (scale_x * scale_w);

    // Fast path: native FP8 GEMM (needs 16-byte-aligned leading dims / rows).
    if ((K % 16 == 0) && (N % 8 == 0)) {
        if (lt_gemm(CUDA_R_8F_E4M3, w_fp8, x_fp8, bias, output, M, K, N, alpha)) return;
        cudaGetLastError();  // clear any sticky error, fall through to fp16 path
    }

    // Fallback: convert fp8 -> bf16 (exact for all e4m3 values), BF16 GEMM, fp32 accumulate.
    size_t xn = (size_t)M * (size_t)K, wn = (size_t)N * (size_t)K;
    __nv_bfloat16* xb = ensure_buf(&g_xb, &g_xb_n, xn);
    __nv_bfloat16* wb = ensure_buf(&g_wb, &g_wb_n, wn);
    if (!xb || !wb) return;
    {
        long n8 = (long)(xn >> 3);
        int threads = 256;
        long blocks = (n8 + threads - 1) / threads;
        cvt_fp8_bf16_8<<<(unsigned)blocks, threads, 0, 0>>>(x_fp8, xb, n8);
    }
    {
        long n8 = (long)(wn >> 3);
        int threads = 256;
        long blocks = (n8 + threads - 1) / threads;
        cvt_fp8_bf16_8<<<(unsigned)blocks, threads, 0, 0>>>(w_fp8, wb, n8);
    }
    lt_gemm(CUDA_R_16BF, wb, xb, bias, output, M, K, N, alpha);
}
