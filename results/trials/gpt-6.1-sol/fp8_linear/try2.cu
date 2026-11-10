#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdint>

// A warp computes a dot product. This path also handles very small batches,
// where a tensor-core GEMM would have more setup cost than useful work.
__global__ void fp8_linear_small(int M, int K, int N,
    const __nv_fp8_e4m3* x, const __nv_fp8_e4m3* w,
    const __nv_bfloat16* bias, __nv_bfloat16* out, float scale) {
  const int lane = threadIdx.x & 31;
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (i >= M * N) return;
  const int m = i / N, n = i % N;
  float sum = 0.f;
  for (int k = lane; k < K; k += 32)
    sum = fmaf(float(x[m * K + k]), float(w[n * K + k]), sum);
  for (int d = 16; d; d >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, d);
  if (lane == 0)
    out[i] = __float2bfloat16_rn(fmaf(sum, scale, __bfloat162float(bias[n])));
}

__global__ void fp8_pad_matrix(const unsigned char* in, unsigned char* out,
                              int rows, int cols, int prows, int pcols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < prows * pcols) {
    int r = i / pcols, c = i % pcols;
    out[i] = (r < rows && c < cols) ? in[r * cols + c] : 0;
  }
}
__global__ void fp8_pad_bias(const __nv_bfloat16* in, __nv_bfloat16* out, int n, int pn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pn) out[i] = i < n ? in[i] : __float2bfloat16_rn(0.f);
}
__global__ void fp8_copy_result(const __nv_bfloat16* in, __nv_bfloat16* out,
                              int m, int n, int pn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m * n) out[i] = in[(i / n) * pn + i % n];
}

namespace fp8_linear_detail {
struct Plan {
  int M = 0, K = 0, N = 0, pm = 0, pk = 0, pn = 0;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulAlgo_t algo;
  bool ready = false, padded = false;
  uint32_t align_a = 256, align_b = 256, align_c = 256, align_d = 256;
  unsigned char *px = nullptr, *pw = nullptr;
  __nv_bfloat16 *pb = nullptr, *py = nullptr;

  void clear() {
    if (op) cublasLtMatmulDescDestroy(op);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (px) cudaFree(px);
    if (pw) cudaFree(pw);
    if (pb) cudaFree(pb);
    if (py) cudaFree(py);
    op = nullptr; a = b = c = nullptr;
    px = pw = nullptr; pb = py = nullptr;
    ready = false;
  }

  void ensure_storage() {
    if (px) return;
    cudaMalloc(&px, size_t(pm) * pk);
    cudaMalloc(&pw, size_t(pn) * pk);
    cudaMalloc(&pb, size_t(pn) * sizeof(__nv_bfloat16));
    cudaMalloc(&py, size_t(pm) * pn * sizeof(__nv_bfloat16));
  }

  void init(cublasLtHandle_t handle, int m, int k, int n, size_t workspace_bytes) {
    clear();
    M = m; K = k; N = n;
    pm = (M + 15) & ~15; pk = (K + 31) & ~31; pn = (N + 15) & ~15;
    padded = pm != M || pk != K || pn != N;
    if (padded) ensure_storage();
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t trans = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans));
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    cudaDataType_t bias_type = CUDA_R_16BF;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type));
    // In column-major notation the desired product is W^T X, with a bias
    // along its first (output-channel) dimension.
    cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, pk, pn, pk);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, pk, pm, pk);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, pn, pm, pn);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &workspace_bytes, sizeof(workspace_bytes));
    cublasLtMatmulHeuristicResult_t result;
    int count = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle, op, a, b, c, c, pref, 1, &result, &count);
    if (status == CUBLAS_STATUS_SUCCESS && count > 0) {
      algo = result.algo;
      size_t written = 0;
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES,
                                       &align_a, sizeof(align_a), &written);
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
                                       &align_b, sizeof(align_b), &written);
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES,
                                       &align_c, sizeof(align_c), &written);
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES,
                                       &align_d, sizeof(align_d), &written);
      ready = true;
    }
    cublasLtMatmulPreferenceDestroy(pref);
  }
};
}

void reference(const int32_t M, const int32_t K, const int32_t N,
               const __nv_fp8_e4m3* x_fp8, float scale_x,
               const __nv_fp8_e4m3* w_fp8, float scale_w,
               const __nv_bfloat16* bias, __nv_bfloat16* output) {
  if (M <= 0 || N <= 0) return;
  const float scale = float(1.0 / (double(scale_x) * double(scale_w)));
  if (M <= 16) {
    fp8_linear_small<<<(M * N + 3) / 4, 128>>>(M, K, N, x_fp8, w_fp8, bias, output, scale);
    return;
  }
  static cublasLtHandle_t handle = nullptr;
  static void* workspace = nullptr;
  constexpr size_t workspace_bytes = 32 * 1024 * 1024;
  static fp8_linear_detail::Plan plan;
  if (!handle) cublasLtCreate(&handle);
  if (!workspace) cudaMalloc(&workspace, workspace_bytes);
  if (plan.M != M || plan.K != K || plan.N != N)
    plan.init(handle, M, K, N, workspace_bytes);
  if (!plan.ready) {
    fp8_linear_small<<<(M * N + 3) / 4, 128>>>(M, K, N, x_fp8, w_fp8, bias, output, scale);
    return;
  }
  const void* x = x_fp8;
  const void* w = w_fp8;
  const __nv_bfloat16* b = bias;
  __nv_bfloat16* y = output;
  const bool buffered = plan.padded ||
    (uintptr_t(w_fp8) % plan.align_a) || (uintptr_t(x_fp8) % plan.align_b) ||
    (uintptr_t(output) % plan.align_c) || (uintptr_t(output) % plan.align_d) ||
    (uintptr_t(bias) % 16);
  if (buffered) {
    plan.ensure_storage();
    fp8_pad_matrix<<<(plan.pm * plan.pk + 255) / 256, 256>>>(
      reinterpret_cast<const unsigned char*>(x_fp8), plan.px, M, K, plan.pm, plan.pk);
    fp8_pad_matrix<<<(plan.pn * plan.pk + 255) / 256, 256>>>(
      reinterpret_cast<const unsigned char*>(w_fp8), plan.pw, N, K, plan.pn, plan.pk);
    fp8_pad_bias<<<(plan.pn + 255) / 256, 256>>>(bias, plan.pb, N, plan.pn);
    x = plan.px; w = plan.pw; b = plan.pb; y = plan.py;
  }
  cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &b, sizeof(b));
  const float beta = 0.f;
  cublasStatus_t status = cublasLtMatmul(handle, plan.op, &scale, w, plan.a, x, plan.b,
                                      &beta, y, plan.c, y, plan.c, &plan.algo,
                                      workspace, workspace_bytes, 0);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fp8_linear_small<<<(M * N + 3) / 4, 128>>>(M, K, N, x_fp8, w_fp8, bias, output, scale);
    return;
  }
  if (buffered)
    fp8_copy_result<<<(M * N + 255) / 256, 256>>>(plan.py, output, M, N, plan.pn);
}
