#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdint>
#include <vector>

// A warp per dot product is efficient for the matrix-vector and tiny problems.
__global__ void small_linear(int M, int K, int N, const __nv_fp8_e4m3* __restrict__ x,
 const __nv_fp8_e4m3* __restrict__ w, const __nv_bfloat16* __restrict__ bias,
 __nv_bfloat16* __restrict__ out, float scale) {
 int lane=threadIdx.x&31;
 int t=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 if(t>=M*N) return;
 int m=t/N, n=t%N;
 float sum=0.f;
 for(int k=lane;k<K;k+=32) sum=fmaf(float(x[m*K+k]),float(w[n*K+k]),sum);
 for(int delta=16;delta>0;delta/=2) sum+=__shfl_down_sync(0xffffffff,sum,delta);
 if(lane==0) out[t]=__float2bfloat16_rn(fmaf(sum,scale,__bfloat162float(bias[n])));
}

__global__ void pad_inputs(const unsigned char* __restrict__ x, const unsigned char* __restrict__ w,
 const __nv_bfloat16* __restrict__ bias, unsigned char* __restrict__ xp, unsigned char* __restrict__ wp,
 __nv_bfloat16* __restrict__ bp, int M, int K, int N, int Mp, int Kp, int Np) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 int row=i/Kp,col=i%Kp;
 if(i<Np*Kp) wp[i]=(row<N&&col<K)?w[row*K+col]:0;
 if(i<Mp*Kp) xp[i]=(row<M&&col<K)?x[row*K+col]:0;
 if(i<Np) bp[i]=i<N?bias[i]:__float2bfloat16_rn(0.f);
}
__global__ void unpad_output(const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
 int M,int N,int Np) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<M*N) out[i]=in[(i/N)*Np+i%N];
}

struct LinearPlan {
 int M,K,N,Mp,Kp,Np;
 cublasLtMatmulDesc_t desc;
 cublasLtMatrixLayout_t a,b,c;
 cublasLtMatmulAlgo_t algo;
 unsigned char *xp=nullptr,*wp=nullptr;
 __nv_bfloat16 *bp=nullptr,*op=nullptr;
 bool padded,valid=false;
};

void reference(const int32_t M, const int32_t K, const int32_t N, const __nv_fp8_e4m3* x_fp8, float scale_x, const __nv_fp8_e4m3* w_fp8, float scale_w, const __nv_bfloat16* bias, __nv_bfloat16* output) {
 const float alpha=float(1.0/(double(scale_x)*double(scale_w)));
 if(M<=16) {
   small_linear<<<(M*N+7)/8,256>>>(M,K,N,x_fp8,w_fp8,bias,output,alpha);
   return;
 }
 const bool aligned=((reinterpret_cast<uintptr_t>(x_fp8)|reinterpret_cast<uintptr_t>(w_fp8)|
                       reinterpret_cast<uintptr_t>(bias)|reinterpret_cast<uintptr_t>(output))&15)==0;
 const bool padded=(M%16!=0 || K%32!=0 || N%16!=0 || !aligned);
 static cublasLtHandle_t handle=nullptr;
 static void* workspace=nullptr;
 constexpr size_t ws=32*1024*1024;
 static std::vector<LinearPlan> plans;
 if(!handle) {cublasLtCreate(&handle);cudaMalloc(&workspace,ws);}
 LinearPlan* p=nullptr;
 for(auto& v:plans) if(v.M==M&&v.K==K&&v.N==N&&v.padded==padded) {p=&v;break;}
 if(!p) {
   plans.emplace_back(); p=&plans.back(); p->M=M;p->K=K;p->N=N;
   p->Mp=(M+15)/16*16;p->Kp=(K+31)/32*32;p->Np=(N+15)/16*16;
   p->padded=padded;
   if(p->padded) {
     cudaMalloc(&p->xp,size_t(p->Mp)*p->Kp);
     cudaMalloc(&p->wp,size_t(p->Np)*p->Kp);
     cudaMalloc(&p->bp,size_t(p->Np)*2);
     cudaMalloc(&p->op,size_t(p->Mp)*p->Np*2);
   }
   cublasLtMatmulDescCreate(&p->desc,CUBLAS_COMPUTE_32F,CUDA_R_32F);
   // Row-major W and X are viewed as column-major KxN and KxM.
   cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
   cublasLtMatmulDescSetAttribute(p->desc,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta));
   cublasLtMatmulDescSetAttribute(p->desc,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb));
   cublasLtEpilogue_t epi=CUBLASLT_EPILOGUE_BIAS;
   cublasLtMatmulDescSetAttribute(p->desc,CUBLASLT_MATMUL_DESC_EPILOGUE,&epi,sizeof(epi));
   const __nv_bfloat16* bp=p->padded?p->bp:bias;
   cublasLtMatmulDescSetAttribute(p->desc,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bp,sizeof(bp));
   cublasLtMatrixLayoutCreate(&p->a,CUDA_R_8F_E4M3,p->Kp,p->Np,p->Kp);
   cublasLtMatrixLayoutCreate(&p->b,CUDA_R_8F_E4M3,p->Kp,p->Mp,p->Kp);
   cublasLtMatrixLayoutCreate(&p->c,CUDA_R_16BF,p->Np,p->Mp,p->Np);
   cublasLtMatmulPreference_t pref;
   cublasLtMatmulPreferenceCreate(&pref);
   cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws));
   uint32_t alignment=16;
   cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,&alignment,sizeof(alignment));
   cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,&alignment,sizeof(alignment));
   cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,&alignment,sizeof(alignment));
   cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,&alignment,sizeof(alignment));
   uint32_t reduction=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
   cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,&reduction,sizeof(reduction));
   cublasLtMatmulHeuristicResult_t hr[16];
   int count=0;
   auto status=cublasLtMatmulAlgoGetHeuristic(handle,p->desc,p->a,p->b,p->c,p->c,pref,16,hr,&count);
   if(status==CUBLAS_STATUS_SUCCESS&&count) {
     int choice=0;
     // Prefer a square tile when there are enough output tiles for full occupancy.
     if(size_t(M)*N>=8*1024*1024) {
       for(int j=0;j<count;j++) {
         int tile=0,split=0;size_t written;
         cublasLtMatmulAlgoConfigGetAttribute(&hr[j].algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile),&written);
         cublasLtMatmulAlgoConfigGetAttribute(&hr[j].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split),&written);
         if(tile==CUBLASLT_MATMUL_TILE_128x128 && split<=1) {choice=j;break;}
       }
     }
     p->algo=hr[choice].algo;p->valid=true;
   }
   cublasLtMatmulPreferenceDestroy(pref);
 }
 if(!p->valid) {
   small_linear<<<(M*N+7)/8,256>>>(M,K,N,x_fp8,w_fp8,bias,output,alpha);
   return;
 }
 const void *a=w_fp8,*b=x_fp8;
 const __nv_bfloat16* bp=bias;
 __nv_bfloat16* d=output;
 if(p->padded) {
   int size=(p->Mp>p->Np?p->Mp:p->Np)*p->Kp;
   pad_inputs<<<(size+255)/256,256>>>((const unsigned char*)x_fp8,(const unsigned char*)w_fp8,bias,p->xp,p->wp,p->bp,M,K,N,p->Mp,p->Kp,p->Np);
   a=p->wp;b=p->xp;bp=p->bp;d=p->op;
 }
 cublasLtMatmulDescSetAttribute(p->desc,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bp,sizeof(bp));
 const float beta=0.f;
 auto status=cublasLtMatmul(handle,p->desc,&alpha,a,p->a,b,p->b,&beta,d,p->c,d,p->c,&p->algo,workspace,ws,0);
 if(status==CUBLAS_STATUS_SUCCESS) {
   if(p->padded) unpad_output<<<(M*N+255)/256,256>>>(d,output,M,N,p->Np);
 } else {
   small_linear<<<(M*N+7)/8,256>>>(M,K,N,x_fp8,w_fp8,bias,output,alpha);
 }
}
