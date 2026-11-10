#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdint>
#include <vector>

namespace fp8_linear_impl {
__global__ void small_gemm(int M,int K,int N,const __nv_fp8_e4m3* x,const __nv_fp8_e4m3* w,const __nv_bfloat16* b,__nv_bfloat16* y,float scale) {
    int lane=threadIdx.x&31;
    int out=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(out>=M*N) return;
    int m=out/N,n=out%N;
    float s=0;
    for(int k=lane;k<K;k+=32) s=fmaf(float(x[(size_t)m*K+k]),float(w[(size_t)n*K+k]),s);
    for(int d=16;d;d>>=1) s+=__shfl_down_sync(0xffffffff,s,d);
    if(lane==0) y[out]=__float2bfloat16_rn(fmaf(s,scale,__bfloat162float(b[n])));
}
__global__ void pad_fp8(const __nv_fp8_e4m3* in,__nv_fp8_e4m3* out,int R,int K,int Rp,int Kp) {
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<(size_t)Rp*Kp;i+=(size_t)blockDim.x*gridDim.x){
        int r=i/Kp,k=i%Kp;
        reinterpret_cast<unsigned char*>(out)[i]=(r<R && k<K)?reinterpret_cast<const unsigned char*>(in)[(size_t)r*K+k]:0;
    }
}
__global__ void pad_bias(const __nv_bfloat16* in,__nv_bfloat16* out,int N,int Np){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<Np) out[i]=i<N?in[i]:__float2bfloat16(0);
}
__global__ void copy_out(const __nv_bfloat16* in,__nv_bfloat16* out,int M,int N,int Np){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<(size_t)M*N;i+=(size_t)gridDim.x*blockDim.x) out[i]=in[(i/N)*Np+i%N];
}
struct Plan {
    int M,K,N,Mp,Kp,Np;
    cublasLtHandle_t lt;
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t a,b,c,d;
    cublasLtMatmulAlgo_t algo;
    void* workspace=nullptr;
    size_t workspace_size=0;
    __nv_fp8_e4m3 *xp=nullptr,*wp=nullptr;
    __nv_bfloat16 *bp=nullptr,*yp=nullptr;
    bool padded,valid=false;
    Plan(int m,int k,int n):M(m),K(k),N(n) {
        Mp=(M+15)/16*16; Kp=(K+15)/16*16; Np=(N+15)/16*16;
        padded=Mp!=M||Kp!=K||Np!=N;
        cublasLtCreate(&lt);
        cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F);
        cublasOperation_t trans=CUBLAS_OP_T;
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans));
        cublasLtEpilogue_t epi=CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_EPILOGUE,&epi,sizeof(epi));
        cudaDataType_t bias_type=CUDA_R_16BF;
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,&bias_type,sizeof(bias_type));
        int8_t fast=0;
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fast,sizeof(fast));
        // Interpret the row-major buffers as column-major transposes:
        // D(N,M) = transpose(W_col(K,N)) * X_col(K,M).
        cublasLtMatrixLayoutCreate(&a,CUDA_R_8F_E4M3,Kp,Np,Kp);
        cublasLtMatrixLayoutCreate(&b,CUDA_R_8F_E4M3,Kp,Mp,Kp);
        cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,Np,Mp,Np);
        cublasLtMatrixLayoutCreate(&d,CUDA_R_16BF,Np,Mp,Np);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_size,sizeof(workspace_size));
        cublasLtMatmulHeuristicResult_t heur[16];
        int count=0;
        cublasStatus_t st=cublasLtMatmulAlgoGetHeuristic(lt,op,a,b,c,d,pref,16,heur,&count);
        valid=st==CUBLAS_STATUS_SUCCESS && count>0;
        if(valid) {
            algo=heur[0].algo;
            // A square tile gives better reuse for the large dense products.
            if(Mp>=1024 && Np>=1024) {
                for(int i=0;i<count;++i) {
                    int tile=0; size_t written=0;
                    cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile),&written);
                    if(tile==CUBLASLT_MATMUL_TILE_128x128) { algo=heur[i].algo; break; }
                }
            }
        }
        cublasLtMatmulPreferenceDestroy(pref);
        if(padded){
            cudaMalloc(&xp,(size_t)Mp*Kp);
            cudaMalloc(&wp,(size_t)Np*Kp);
            cudaMalloc(&bp,(size_t)Np*sizeof(__nv_bfloat16));
            cudaMalloc(&yp,(size_t)Mp*Np*sizeof(__nv_bfloat16));
        }
    }
};
}
void reference(const int32_t M,const int32_t K,const int32_t N,const __nv_fp8_e4m3* x_fp8,float scale_x,const __nv_fp8_e4m3* w_fp8,float scale_w,const __nv_bfloat16* bias,__nv_bfloat16* output){
    using namespace fp8_linear_impl;
    float alpha=(float)(1.0/(double(scale_x)*double(scale_w)));
    if(M<32){
        small_gemm<<<(M*N+3)/4,128>>>(M,K,N,x_fp8,w_fp8,bias,output,alpha);
        return;
    }
    // Cache only descriptors and scratch allocations; inputs are read anew.
    static std::vector<Plan*> plans;
    Plan* p=nullptr;
    for(auto q:plans) if(q->M==M && q->K==K && q->N==N) {p=q;break;}
    if(!p){p=new Plan(M,K,N);plans.push_back(p);}
    if(!p->valid){
        small_gemm<<<(M*N+3)/4,128>>>(M,K,N,x_fp8,w_fp8,bias,output,alpha);
        return;
    }
    const __nv_fp8_e4m3 *x=x_fp8,*w=w_fp8;
    const __nv_bfloat16 *b=bias;
    __nv_bfloat16 *y=output;
    if(p->padded){
        pad_fp8<<<256,256>>>(x,p->xp,M,K,p->Mp,p->Kp);
        pad_fp8<<<256,256>>>(w,p->wp,N,K,p->Np,p->Kp);
        pad_bias<<<(p->Np+255)/256,256>>>(bias,p->bp,N,p->Np);
        x=p->xp;w=p->wp;b=p->bp;y=p->yp;
    }
    cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&b,sizeof(b));
    float beta=0;
    cublasStatus_t st=cublasLtMatmul(p->lt,p->op,&alpha,w,p->a,x,p->b,&beta,y,p->c,y,p->d,&p->algo,p->workspace,p->workspace_size,0);
    if(st!=CUBLAS_STATUS_SUCCESS){
        small_gemm<<<(M*N+3)/4,128>>>(M,K,N,x_fp8,w_fp8,bias,output,alpha);
    } else if(p->padded) copy_out<<<256,256>>>(y,output,M,N,p->Np);
}
