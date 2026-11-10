#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cstdint>

// fp8_e4m3 -> bf16 is exact (e4m3 has 3 mantissa bits, exp range inside bf16).
static __global__ void fp8_to_bf16_kernel(const __nv_fp8_e4m3* __restrict__ in,
                                          __nv_bfloat16* __restrict__ out,
                                          long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nv = n >> 3;
    if (i < nv) {
        uint2 raw = reinterpret_cast<const uint2*>(in)[i];
        const __nv_fp8_e4m3* f = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
        __nv_bfloat16 b[8];
#pragma unroll
        for (int j = 0; j < 8; j++) b[j] = __float2bfloat16(float(f[j]));
        reinterpret_cast<uint4*>(out)[i] = *reinterpret_cast<const uint4*>(b);
    }
    if (i == 0) {
        for (long long j = nv << 3; j < n; j++) out[j] = __float2bfloat16(float(in[j]));
    }
}

static void convert(const __nv_fp8_e4m3* in, __nv_bfloat16* out, size_t n) {
    long long nv = (long long)(n >> 3);
    unsigned blocks = (unsigned)((nv + 255) / 256);
    if (blocks == 0) blocks = 1;
    fp8_to_bf16_kernel<<<blocks, 256>>>(in, out, (long long)n);
}

void reference(const int32_t M, const int32_t K, const int32_t N,
               const __nv_fp8_e4m3* x_fp8, float scale_x,
               const __nv_fp8_e4m3* w_fp8, float scale_w,
               const __nv_bfloat16* bias, __nv_bfloat16* output) {
    static cublasLtHandle_t lt = nullptr;
    static __nv_bfloat16 *xb = nullptr, *wb = nullptr, *ws = nullptr;
    static size_t xcap = 0, wcap = 0;
    static const size_t WS_BYTES = 32u << 20;
    if (!lt) {
        cublasLtCreate(&lt);
        cudaMalloc(&ws, WS_BYTES);
    }
    size_t xn = (size_t)M * (size_t)K, wn = (size_t)N * (size_t)K;
    if (xn > xcap) { if (xb) cudaFree(xb); cudaMalloc(&xb, xn * sizeof(__nv_bfloat16)); xcap = xn; }
    if (wn > wcap) { if (wb) cudaFree(wb); cudaMalloc(&wb, wn * sizeof(__nv_bfloat16)); wcap = wn; }

    convert(x_fp8, xb, xn);
    convert(w_fp8, wb, wn);

    // Row-major C(M,N) = xb @ wb^T  <=>  col-major D(N,M) = wb_col^T @ xb_col
    // A = wb viewed col-major (K,N), transa=T ; B = xb viewed col-major (K,M), transb=N.
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    cudaDataType_t bdt = CUDA_R_16BF;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bdt, sizeof(bdt));

    cublasLtMatrixLayout_t A, B, C;
    cublasLtMatrixLayoutCreate(&A, CUDA_R_16BF, K, N, K);
    cublasLtMatrixLayoutCreate(&B, CUDA_R_16BF, K, M, K);
    cublasLtMatrixLayoutCreate(&C, CUDA_R_16BF, N, M, N);

    float alpha = 1.0f / (scale_x * scale_w);
    float beta = 0.0f;
    cublasLtMatmul(lt, desc, &alpha,
                   wb, A, xb, B,
                   &beta, output, C, output, C,
                   nullptr, ws, WS_BYTES, 0);

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(A);
    cublasLtMatrixLayoutDestroy(B);
    cublasLtMatrixLayoutDestroy(C);
}
