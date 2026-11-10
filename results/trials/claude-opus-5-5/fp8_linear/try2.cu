#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdint>
#include <map>
#include <tuple>

// ---------------- fallback: generic tiled fp32 kernel ----------------
#define FB_BM 64
#define FB_BN 64
#define FB_BK 16
__global__ void fp8_linear_fallback(int M, int K, int N, const __nv_fp8_e4m3* __restrict__ x,
                                    const __nv_fp8_e4m3* __restrict__ w, float alpha,
                                    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out) {
    __shared__ float As[FB_BK][FB_BM + 1];
    __shared__ float Bs[FB_BK][FB_BN + 1];
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    int m0 = blockIdx.y * FB_BM, n0 = blockIdx.x * FB_BN;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += FB_BK) {
        for (int i = threadIdx.x; i < FB_BM * FB_BK; i += 256) {
            int r = i / FB_BK, c = i % FB_BK;
            int gm = m0 + r, gk = k0 + c;
            As[c][r] = (gm < M && gk < K) ? float(x[(size_t)gm * K + gk]) : 0.f;
            int gn = n0 + r;
            Bs[c][r] = (gn < N && gk < K) ? float(w[(size_t)gn * K + gk]) : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < FB_BK; kk++) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) { a[i] = As[kk][ty + 16 * i]; b[i] = Bs[kk][tx + 16 * i]; }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int gm = m0 + ty + 16 * i;
        if (gm >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int gn = n0 + tx + 16 * j;
            if (gn >= N) continue;
            out[(size_t)gm * N + gn] = __float2bfloat16(acc[i][j] * alpha + __bfloat162float(bias[gn]));
        }
    }
}


// copy rows of K bytes into rows of Kp bytes (zero padded)
__global__ void pad_rows(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int rows, int K, int Kp) {
    size_t total = (size_t)rows * Kp;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        size_t r = i / Kp; int c = (int)(i % Kp);
        dst[i] = c < K ? src[r * K + c] : (uint8_t)0;
    }
}
static void* g_pad = nullptr;
static size_t g_pad_size = 0;
// ---------------- cuBLASLt FP8 path ----------------
struct Plan {
    bool ok = false;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo;
    size_t ws = 0;
};

static cublasLtHandle_t g_lt = nullptr;
static void* g_ws = nullptr;
static const size_t WS_SIZE = 32u << 20;
static std::map<std::tuple<int, int, int, uintptr_t>, Plan> g_plans;

static Plan make_plan(int M, int K, int N, const void* bias) {
    Plan p;
    if (cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return p;
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
    cudaDataType_t bt = CUDA_R_16BF;
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    // A = W (K x N col-major, ld K, transposed), B = X (K x M col-major, ld K), D = out^T (N x M, ld N)
    cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, N, M, N);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws = WS_SIZE;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t res[1];
    int n = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(g_lt, p.op, p.a, p.b, p.c, p.c, pref, 1, res, &n);
    cublasLtMatmulPreferenceDestroy(pref);
    if (st == CUBLAS_STATUS_SUCCESS && n > 0 && res[0].state == CUBLAS_STATUS_SUCCESS) {
        p.algo = res[0].algo;
        p.ws = res[0].workspaceSize;
        p.ok = true;
    }
    return p;
}

void reference(const int32_t M, const int32_t K, const int32_t N, const __nv_fp8_e4m3* x_fp8, float scale_x,
               const __nv_fp8_e4m3* w_fp8, float scale_w, const __nv_bfloat16* bias, __nv_bfloat16* output) {
    if (M <= 0 || N <= 0) return;
    float alpha = (float)(1.0 / ((double)scale_x * (double)scale_w));
    bool outok = ((uintptr_t)output % 16 == 0) && ((uintptr_t)bias % 16 == 0) && (N % 8 == 0) && K > 0;
    if (outok) {
        if (!g_lt) {
            if (cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) g_lt = nullptr;
            if (g_lt && cudaMalloc(&g_ws, WS_SIZE) != cudaSuccess) g_ws = nullptr;
        }
        const void* xa = x_fp8;
        const void* wa = w_fp8;
        int Kp = K;
        bool good = g_lt && g_ws;
        if (good && ((K % 16) || (((uintptr_t)x_fp8 | (uintptr_t)w_fp8) % 16))) {
            Kp = (K + 15) / 16 * 16;
            size_t need = ((size_t)M + N) * Kp;
            if (need > g_pad_size) {
                if (g_pad) cudaFree(g_pad);
                g_pad = nullptr; g_pad_size = 0;
                if (cudaMalloc(&g_pad, need) == cudaSuccess) g_pad_size = need;
            }
            if (g_pad) {
                uint8_t* xp = (uint8_t*)g_pad;
                uint8_t* wp = xp + (size_t)M * Kp;
                size_t tx = (size_t)M * Kp, tw = (size_t)N * Kp;
                int bx = (int)((tx + 255) / 256); if (bx > 4096) bx = 4096;
                int bw = (int)((tw + 255) / 256); if (bw > 4096) bw = 4096;
                pad_rows<<<bx, 256>>>((const uint8_t*)x_fp8, xp, M, K, Kp);
                pad_rows<<<bw, 256>>>((const uint8_t*)w_fp8, wp, N, K, Kp);
                xa = xp; wa = wp;
            } else good = false;
        }
        if (good) {
            auto key = std::make_tuple((int)M, (int)Kp, (int)N, (uintptr_t)bias);
            auto it = g_plans.find(key);
            if (it == g_plans.end()) it = g_plans.emplace(key, make_plan(M, Kp, N, bias)).first;
            Plan& p = it->second;
            if (p.ok) {
                float beta = 0.f;
                cublasStatus_t st = cublasLtMatmul(g_lt, p.op, &alpha, wa, p.a, xa, p.b, &beta, output, p.c,
                                                   output, p.c, &p.algo, g_ws, p.ws, 0);
                if (st == CUBLAS_STATUS_SUCCESS) return;
            }
        }
    }
    dim3 grid((N + FB_BN - 1) / FB_BN, (M + FB_BM - 1) / FB_BM);
    fp8_linear_fallback<<<grid, 256>>>(M, K, N, x_fp8, w_fp8, alpha, bias, output);
}
