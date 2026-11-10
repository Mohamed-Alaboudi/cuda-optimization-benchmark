// FP8 (E4M3) linear forward for inference:
//   output = (float(x_fp8)/scale_x) @ (float(w_fp8)/scale_w)^T + bias   (bf16 out)
//
// Strategy: cuBLASLt FP8 GEMM (TN layout, per-tensor dequant scales, bias epilogue,
// bf16 output). K is padded to a multiple of 16 via a copy when needed.
// Fallback: dequantize to fp16 and run a cuBLASLt fp16 GEMM with fp32 compute.

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

namespace {

cublasLtHandle_t g_handle = nullptr;
void*  g_ws = nullptr;
const size_t WS_BYTES = 64ull << 20;
float* g_scales = nullptr;  // device: [0] = 1/scale_w (A), [1] = 1/scale_x (B)

uint8_t* g_pad_x = nullptr; size_t g_pad_x_cap = 0;
uint8_t* g_pad_w = nullptr; size_t g_pad_w_cap = 0;
__half*  g_hx = nullptr;    size_t g_hx_cap = 0;
__half*  g_hw = nullptr;    size_t g_hw_cap = 0;

bool ensure(void** p, size_t* cap, size_t bytes) {
    if (*cap >= bytes) return true;
    if (*p) cudaFree(*p);
    if (cudaMalloc(p, bytes) != cudaSuccess) { *p = nullptr; *cap = 0; return false; }
    *cap = bytes;
    return true;
}

bool init() {
    if (g_handle) return true;
    if (cublasLtCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) return false;
    if (cudaMalloc(&g_ws, WS_BYTES) != cudaSuccess) return false;
    if (cudaMalloc(&g_scales, 2 * sizeof(float)) != cudaSuccess) return false;
    return true;
}

struct MatmulCommon {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t Al = nullptr, Bl = nullptr, Dl = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    ~MatmulCommon() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (Dl) cublasLtMatrixLayoutDestroy(Dl);
        if (Bl) cublasLtMatrixLayoutDestroy(Bl);
        if (Al) cublasLtMatrixLayoutDestroy(Al);
        if (desc) cublasLtMatmulDescDestroy(desc);
    }
};

// C'(N,M) col-major (= output row-major MxN) = opT(W(K,N)) * opN(X(K,M)).
bool run_gemm(cudaDataType_t ab_type, cublasComputeType_t compute,
              const void* w, const void* x, const __nv_bfloat16* bias,
              __nv_bfloat16* output, int M, int K, int N,
              float inv_sw, float inv_sx, bool use_scale_ptrs) {
    MatmulCommon mc;
    if (cublasLtMatmulDescCreate(&mc.desc, compute, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(mc.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(mc.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    if (use_scale_ptrs) {
        float host[2] = {inv_sw, inv_sx};
        if (cudaMemcpyAsync(g_scales, host, 2 * sizeof(float), cudaMemcpyHostToDevice, 0) != cudaSuccess)
            return false;
        const float* aptr = g_scales;
        const float* bptr = g_scales + 1;
        cublasLtMatmulDescSetAttribute(mc.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aptr, sizeof(aptr));
        cublasLtMatmulDescSetAttribute(mc.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bptr, sizeof(bptr));
    }
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(mc.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
    cublasLtMatmulDescSetAttribute(mc.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    cudaDataType_t bdt = CUDA_R_16BF;
    cublasLtMatmulDescSetAttribute(mc.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bdt, sizeof(bdt));

    if (cublasLtMatrixLayoutCreate(&mc.Al, ab_type, K, N, K) != CUBLAS_STATUS_SUCCESS) return false;
    if (cublasLtMatrixLayoutCreate(&mc.Bl, ab_type, K, M, K) != CUBLAS_STATUS_SUCCESS) return false;
    if (cublasLtMatrixLayoutCreate(&mc.Dl, CUDA_R_16BF, N, M, N) != CUBLAS_STATUS_SUCCESS) return false;

    if (cublasLtMatmulPreferenceCreate(&mc.pref) != CUBLAS_STATUS_SUCCESS) return false;
    size_t ws = WS_BYTES;
    cublasLtMatmulPreferenceSetAttribute(mc.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

    cublasLtMatmulHeuristicResult_t res[8];
    int nres = 0;
    if (cublasLtMatmulAlgoGetHeuristic(g_handle, mc.desc, mc.Al, mc.Bl, mc.Dl, mc.Dl,
                                       mc.pref, 8, res, &nres) != CUBLAS_STATUS_SUCCESS)
        return false;
    float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < nres; ++i) {
        if (res[i].workspaceSize > WS_BYTES) continue;
        cublasStatus_t st = cublasLtMatmul(g_handle, mc.desc, &alpha,
                                           w, mc.Al, x, mc.Bl, &beta,
                                           output, mc.Dl, output, mc.Dl,
                                           &res[i].algo, g_ws, WS_BYTES, 0);
        if (st == CUBLAS_STATUS_SUCCESS) return true;
    }
    return false;
}

bool fp8_path(int32_t M, int32_t K, int32_t N,
              const __nv_fp8_e4m3* x, float inv_sx,
              const __nv_fp8_e4m3* w, float inv_sw,
              const __nv_bfloat16* bias, __nv_bfloat16* output) {
    const uint8_t* xp = reinterpret_cast<const uint8_t*>(x);
    const uint8_t* wp = reinterpret_cast<const uint8_t*>(w);
    int Kp = K;
    if (K % 16 != 0) {
        Kp = (K + 15) / 16 * 16;
        if (!ensure((void**)&g_pad_x, &g_pad_x_cap, (size_t)M * Kp)) return false;
        if (!ensure((void**)&g_pad_w, &g_pad_w_cap, (size_t)N * Kp)) return false;
        // zero tail columns (bytes K..Kp-1 of each row) then copy payload
        if (cudaMemsetAsync(g_pad_x, 0, (size_t)M * Kp, 0) != cudaSuccess) return false;
        if (cudaMemsetAsync(g_pad_w, 0, (size_t)N * Kp, 0) != cudaSuccess) return false;
        if (cudaMemcpy2DAsync(g_pad_x, Kp, x, K, K, M, cudaMemcpyDeviceToDevice, 0) != cudaSuccess) return false;
        if (cudaMemcpy2DAsync(g_pad_w, Kp, w, K, K, N, cudaMemcpyDeviceToDevice, 0) != cudaSuccess) return false;
        xp = g_pad_x; wp = g_pad_w;
    }
    return run_gemm(CUDA_R_8F_E4M3, CUBLAS_COMPUTE_32F, wp, xp, bias, output,
                    M, Kp, N, inv_sw, inv_sx, true);
}

__global__ void dequant_kernel(const __nv_fp8_e4m3* __restrict__ in, float inv,
                               __half* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(float(in[i]) * inv);
}

bool fp16_path(int32_t M, int32_t K, int32_t N,
               const __nv_fp8_e4m3* x, float inv_sx,
               const __nv_fp8_e4m3* w, float inv_sw,
               const __nv_bfloat16* bias, __nv_bfloat16* output) {
    if (!ensure((void**)&g_hx, &g_hx_cap, (size_t)M * K * sizeof(__half))) return false;
    if (!ensure((void**)&g_hw, &g_hw_cap, (size_t)N * K * sizeof(__half))) return false;
    const int T = 256;
    dequant_kernel<<<(unsigned)(((size_t)M * K + T - 1) / T), T, 0, 0>>>(x, inv_sx, g_hx, (size_t)M * K);
    dequant_kernel<<<(unsigned)(((size_t)N * K + T - 1) / T), T, 0, 0>>>(w, inv_sw, g_hw, (size_t)N * K);
    // scale already folded into the fp16 operands
    return run_gemm(CUDA_R_16F, CUBLAS_COMPUTE_32F, g_hw, g_hx, bias, output,
                    M, K, N, 1.0f, 1.0f, false);
}

}  // namespace

void reference(const int32_t M, const int32_t K, const int32_t N,
               const __nv_fp8_e4m3* x_fp8, float scale_x,
               const __nv_fp8_e4m3* w_fp8, float scale_w,
               const __nv_bfloat16* bias, __nv_bfloat16* output) {
    float inv_sx = 1.0f / scale_x, inv_sw = 1.0f / scale_w;
    if (init()) {
        if (fp8_path(M, K, N, x_fp8, inv_sx, w_fp8, inv_sw, bias, output)) return;
        if (fp16_path(M, K, N, x_fp8, inv_sx, w_fp8, inv_sw, bias, output)) return;
    }
}
