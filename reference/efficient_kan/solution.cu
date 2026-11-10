#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cuda_fp16.h>

// Cox-de Boor recurrence for one (x, grid_row). Max L = 16 -> bases[15], new_bases[14].
__device__ inline void compute_bases_device(
    const float* __restrict__ grid_row,
    int grid_size,
    int spline_order,
    float x,
    float* __restrict__ out_bases)
{
    int L = grid_size + 2 * spline_order + 1;
    float bases[15];
    float new_bases[14];

    for (int j = 0; j < L - 1; ++j) {
        float g0 = grid_row[j];
        float g1 = grid_row[j + 1];
        bases[j] = (x >= g0 && x < g1) ? 1.0f : 0.0f;
    }

    int cur_len = L - 1;
    for (int order = 1; order <= spline_order; ++order) {
        for (int j = 0; j < cur_len - 1; ++j) {
            float ln  = x - grid_row[j];
            float ld  = grid_row[j + order] - grid_row[j];
            float rn  = grid_row[j + order + 1] - x;
            float rd  = grid_row[j + order + 1] - grid_row[j + 1];
            float left  = 0.0f;
            float right = 0.0f;
            if (ld != 0.0f) left  = __fdividef(ln, ld) * bases[j];
            if (rd != 0.0f) right = __fdividef(rn, rd) * bases[j + 1];
            new_bases[j] = left + right;
        }
        cur_len -= 1;
        for (int j = 0; j < cur_len; ++j) bases[j] = new_bases[j];
    }

    int C = grid_size + spline_order;
    for (int c = 0; c < C; ++c) out_bases[c] = bases[c];
}

// build concatenated activation matrix A (B, K)
__global__ void build_A_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grid,
    int B, int I, int K,
    int grid_size, int spline_order,
    float* __restrict__ A)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tid / I;
    int i = tid % I;
    if (b >= B) return;

    int base = b * K;
    float xv   = x[b * I + i];
    float silu = __fdividef(xv, 1.0f + __expf(-xv));
    A[base + i] = silu;

    float bases_arr[16];
    compute_bases_device(grid + i * (grid_size + 2 * spline_order + 1),
                         grid_size, spline_order, xv, bases_arr);

    int C = K / I - 1;
    for (int c = 0; c < C; ++c) {
        A[base + (1 + c) * I + i] = bases_arr[c];
    }
}

// build concatenated weight matrix W (O, K)
__global__ void build_W_kernel(
    const float* __restrict__ base_weight,
    const float* __restrict__ spline_weight,
    const float* __restrict__ spline_scaler,
    int O, int I, int K,
    int grid_size, int spline_order,
    float* __restrict__ W)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int o = tid / I;
    int i = tid % I;
    if (o >= O) return;

    int base = o * K;
    W[base + i] = base_weight[o * I + i];

    float scaler = spline_scaler[o * I + i];
    int C = K / I - 1;
    for (int c = 0; c < C; ++c) {
        float w = spline_weight[((o * I) + i) * C + c];
        W[base + (1 + c) * I + i] = w * scaler;
    }
}


// On the specified uniform grid the cubic Cox-de Boor recurrence reduces to
// the four cardinal cubic polynomials below.  q is the knot interval, and
// only bases q-3 through q can be nonzero.  The output is laid out with the
// feature index contiguous, followed by the spline-basis index.
//
// Keep a pair of half matrices for each operand.  The first stores the usual
// half rounding; the second stores 1024 times its residual.  The scale keeps
// small residuals away from the half subnormal range.
__global__ void fast_A(const float* __restrict__ x, const float* __restrict__ grid,
                       int B, int I, int K, half* __restrict__ hi, half* __restrict__ lo) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=B*I) return;
    int b=idx/I, i=idx%I;
    float v=x[idx];
    float t=(v-grid[i*15])*4.f;
    int q=__float2int_rd(t);
    float f=(q>=0 && q<15)?(v-grid[i*15+q])*4.f:0.f;
    float a=1.f-f, f2=f*f, f3=f2*f;
    float b0=a*a*a*(1.f/6.f);
    float b1=(3.f*f3-6.f*f2+4.f)*(1.f/6.f);
    float b2=(-3.f*f3+3.f*f2+3.f*f+1.f)*(1.f/6.f);
    float b3=f3*(1.f/6.f);
    float silu=v/(1.f+__expf(-v));
    int base=b*K+i;
    half h=__float2half_rn(silu); hi[base]=h; lo[base]=__float2half_rn((silu-__half2float(h))*1024.f);
    #pragma unroll
    for(int c=0;c<11;c++) {
      float val=0.f;
      if(q>=0 && q<15) {
        if(c==q-3) val=b0;
        else if(c==q-2) val=b1;
        else if(c==q-1) val=b2;
        else if(c==q) val=b3;
      }
      h=__float2half_rn(val); hi[base+(c+1)*I]=h;
      lo[base+(c+1)*I]=__float2half_rn((val-__half2float(h))*1024.f);
    }
}
__global__ void fast_W(const float* __restrict__ bw, const float* __restrict__ sw,
                       const float* __restrict__ sc, int O, int I, int K,
                       half* __restrict__ hi, half* __restrict__ lo) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=O*I) return;
    int o=idx/I, i=idx%I;
    int base=o*K+i;
    float v=bw[idx]; half h=__float2half_rn(v);
    hi[base]=h; lo[base]=__float2half_rn((v-__half2float(h))*1024.f);
    float scale=sc[idx];
    #pragma unroll
    for(int c=0;c<11;c++) {
        v=sw[idx*11+c]*scale;
        h=__float2half_rn(v); hi[base+(c+1)*I]=h;
        lo[base+(c+1)*I]=__float2half_rn((v-__half2float(h))*1024.f);
    }
}
static half *ha=nullptr,*la=nullptr,*hw=nullptr,*lw=nullptr;
static size_t acap=0,wcap=0;
// cached resources
static float* g_d_A = nullptr;
static float* g_d_W = nullptr;
static size_t g_cap_A = 0;
static size_t g_cap_W = 0;
static cublasHandle_t g_handle = nullptr;
static bool g_handle_init = false;

static void cleanup_kan_resources() {
    if (ha) {cudaFree(ha); ha=nullptr;}
    if (la) {cudaFree(la); la=nullptr;}
    if (hw) {cudaFree(hw); hw=nullptr;}
    if (lw) {cudaFree(lw); lw=nullptr;}
    acap = wcap = 0;
    if (g_d_A) { cudaFree(g_d_A); g_d_A = nullptr; g_cap_A = 0; }
    if (g_d_W) { cudaFree(g_d_W); g_d_W = nullptr; g_cap_W = 0; }
    if (g_handle) { cublasDestroy(g_handle); g_handle = nullptr; g_handle_init = false; }
}

void reference(
    const float* x,
    const float* grid,
    const float* base_weight,
    const float* spline_weight,
    const float* spline_scaler,
    int32_t grid_size,
    int32_t spline_order,
    int32_t B,
    int32_t I,
    int32_t O,
    float* out)
{
    int C = grid_size + spline_order;
    int K = I * (1 + C);

    static bool atexit_registered = false;
    if (!atexit_registered) {
        std::atexit(cleanup_kan_resources);
        atexit_registered = true;
    }

    if (!g_handle_init) {
        cublasCreate(&g_handle);
        cublasSetMathMode(g_handle, CUBLAS_PEDANTIC_MATH);
        g_handle_init = true;
    }


    if (grid_size==8 && spline_order==3 && B>=64) {
        size_t asz=(size_t)B*K*sizeof(half), wsz=(size_t)O*K*sizeof(half);
        if (asz>acap) { if(ha) cudaFree(ha); if(la) cudaFree(la); cudaMalloc(&ha,asz); cudaMalloc(&la,asz); acap=asz; }
        if (wsz>wcap) { if(hw) cudaFree(hw); if(lw) cudaFree(lw); cudaMalloc(&hw,wsz); cudaMalloc(&lw,wsz); wcap=wsz; }
        fast_A<<<(B*I+255)/256,256>>>(x,grid,B,I,K,ha,la);
        fast_W<<<(O*I+255)/256,256>>>(base_weight,spline_weight,spline_scaler,O,I,K,hw,lw);
        // A*W = Ah*Wh + (Al*Wh + Ah*Wl)/1024, up to the
        // negligible product of the two residuals. Accumulation is fp32.
        float one=1.f, zero=0.f, small=1.f/1024.f;
        cublasGemmEx(g_handle,CUBLAS_OP_T,CUBLAS_OP_N,O,B,K,&one,hw,CUDA_R_16F,K,ha,CUDA_R_16F,K,&zero,out,CUDA_R_32F,O,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasGemmEx(g_handle,CUBLAS_OP_T,CUBLAS_OP_N,O,B,K,&small,lw,CUDA_R_16F,K,ha,CUDA_R_16F,K,&one,out,CUDA_R_32F,O,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasGemmEx(g_handle,CUBLAS_OP_T,CUBLAS_OP_N,O,B,K,&small,hw,CUDA_R_16F,K,la,CUDA_R_16F,K,&one,out,CUDA_R_32F,O,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        return;
    }
    size_t need_A = static_cast<size_t>(B) * K * sizeof(float);
    size_t need_W = static_cast<size_t>(O) * K * sizeof(float);

    if (need_A > g_cap_A) {
        if (g_d_A) cudaFree(g_d_A);
        cudaMalloc(&g_d_A, need_A);
        g_cap_A = need_A;
    }
    if (need_W > g_cap_W) {
        if (g_d_W) cudaFree(g_d_W);
        cudaMalloc(&g_d_W, need_W);
        g_cap_W = need_W;
    }

    constexpr int BLOCK = 64;
    int total_A = B * I;
    int total_W = O * I;
    dim3 block(BLOCK);
    dim3 grid_A((total_A + BLOCK - 1) / BLOCK);
    dim3 grid_W((total_W + BLOCK - 1) / BLOCK);

    build_A_kernel<<<grid_A, block>>>(
        x, grid, B, I, K, grid_size, spline_order, g_d_A);

    build_W_kernel<<<grid_W, block>>>(
        base_weight, spline_weight, spline_scaler,
        O, I, K, grid_size, spline_order, g_d_W);

    // Try a few cublasGemmEx algorithm IDs.  Small/medium FP32 GEMMs often
    // have lower-register alternatives under ALGO0..ALGO5.
    float alpha = 1.0f;
    float beta  = 0.0f;

    static const cublasGemmAlgo_t algos[] = {
        CUBLAS_GEMM_ALGO0,
        CUBLAS_GEMM_ALGO1,
        CUBLAS_GEMM_ALGO2,
        CUBLAS_GEMM_ALGO3,
        CUBLAS_GEMM_DEFAULT,
    };

    cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
    for (size_t a = 0; a < sizeof(algos)/sizeof(algos[0]); ++a) {
        status = cublasGemmEx(
            g_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            O, B, K,
            &alpha,
            g_d_W, CUDA_R_32F, K,
            g_d_A, CUDA_R_32F, K,
            &beta,
            out, CUDA_R_32F, O,
            CUBLAS_COMPUTE_32F_PEDANTIC,
            algos[a]);
        if (status == CUBLAS_STATUS_SUCCESS) break;
    }

    // Ultimate fallback
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasSgemm(g_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    O, B, K,
                    &alpha,
                    g_d_W, K,
                    g_d_A, K,
                    &beta,
                    out, O);
    }
}
