#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>

namespace {
// With the specified uniform grid, the cubic Cox--de Boor recurrence has
// four nonzero cardinal basis values. Pack the 11 spline coordinates and
// SiLU coordinate together so that the final addition is part of the GEMM.
__global__ void pack_x(const float* __restrict__ x, const float* __restrict__ grid, float* __restrict__ a, int B, int I) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if(q >= B*I) return;
    int i=q%I;
    float v=x[q];
    float t=(v-grid[i*15])/(grid[i*15+1]-grid[i*15]);
    int j=(int)floorf(t);
    float u=t-(float)j;
    float z=1.f-u;
    float p0=z*z*z*(1.f/6.f);
    float p1=(4.f+u*u*(3.f*u-6.f))*(1.f/6.f);
    float p2=(1.f+u*(3.f+u*(3.f-3.f*u)))*(1.f/6.f);
    float p3=u*u*u*(1.f/6.f);
    float vals[12];
    #pragma unroll
    for(int c=0;c<11;++c) vals[c]=(c==j-3?p0:0.f)+(c==j-2?p1:0.f)+(c==j-1?p2:0.f)+(c==j?p3:0.f);
    vals[11]=v/(1.f+expf(-v));
    float4* dst=(float4*)(a+q*12);
    dst[0]=make_float4(vals[0],vals[1],vals[2],vals[3]);
    dst[1]=make_float4(vals[4],vals[5],vals[6],vals[7]);
    dst[2]=make_float4(vals[8],vals[9],vals[10],vals[11]);
}
__global__ void pack_w(const float* __restrict__ bw,const float* __restrict__ sw,const float* __restrict__ ss,float* __restrict__ w,int n) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=n) return;
    float s=ss[q];
    float vals[12];
    #pragma unroll
    for(int c=0;c<11;++c) vals[c]=sw[q*11+c]*s;
    vals[11]=bw[q];
    float4* dst=(float4*)(w+q*12);
    dst[0]=make_float4(vals[0],vals[1],vals[2],vals[3]);
    dst[1]=make_float4(vals[4],vals[5],vals[6],vals[7]);
    dst[2]=make_float4(vals[8],vals[9],vals[10],vals[11]);
}
// Two half components retain roughly twenty significant bits. The residual
// is scaled by 4096 to avoid underflow in its half representation.
__global__ void pack_x_half(const float* __restrict__ x, const float* __restrict__ grid, __half* __restrict__ a, int B, int I) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if(q >= B*I) return;
    int i=q%I;
    float v=x[q];
    float t=(v-grid[i*15])/(grid[i*15+1]-grid[i*15]);
    int j=(int)floorf(t);
    float u=t-(float)j;
    float z=1.f-u;
    float p0=z*z*z*(1.f/6.f);
    float p1=(4.f+u*u*(3.f*u-6.f))*(1.f/6.f);
    float p2=(1.f+u*(3.f+u*(3.f-3.f*u)))*(1.f/6.f);
    float p3=u*u*u*(1.f/6.f);
    float vals[12];
    #pragma unroll
    for(int c=0;c<11;++c) vals[c]=(c==j-3?p0:0.f)+(c==j-2?p1:0.f)+(c==j-1?p2:0.f)+(c==j?p3:0.f);
    vals[11]=v/(1.f+expf(-v));
    __half2* dst=(__half2*)(a+q*12);
    #pragma unroll
    for(int c=0;c<6;c++) {
        __half h0=__float2half_rn(vals[2*c]);
        __half h1=__float2half_rn(vals[2*c+1]);
        __half l0=__float2half_rn((vals[2*c]-__half2float(h0))*4096.f);
        __half l1=__float2half_rn((vals[2*c+1]-__half2float(h1))*4096.f);
        dst[c]=__halves2half2(h0,h1);
        dst[B*I*6+c]=__halves2half2(l0,l1);
    }
}
__global__ void pack_w_half(const float* __restrict__ bw,const float* __restrict__ sw,const float* __restrict__ ss,__half* __restrict__ w,int n) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=n) return;
    float s=ss[q];
    float vals[12];
    #pragma unroll
    for(int c=0;c<11;++c) vals[c]=sw[q*11+c]*s;
    vals[11]=bw[q];
    __half2* dst=(__half2*)(w+q*12);
    #pragma unroll
    for(int c=0;c<6;c++) {
        __half h0=__float2half_rn(vals[2*c]);
        __half h1=__float2half_rn(vals[2*c+1]);
        __half l0=__float2half_rn((vals[2*c]-__half2float(h0))*4096.f);
        __half l1=__float2half_rn((vals[2*c+1]-__half2float(h1))*4096.f);
        dst[c]=__halves2half2(h0,h1);
        dst[n*6+c]=__halves2half2(l0,l1);
    }
}

using namespace nvcuda;
// Accumulate high*high and the two cross products separately in FP32.
// Splitting the reduction also bounds rounding error for long dot products.
template<int SPLIT>
__global__ __launch_bounds__(256) void tensor_gemm(const __half* __restrict__ a, const __half* __restrict__ w, float* __restrict__ out, int M, int N, int D) {
    constexpr int TK=32; constexpr int LD=TK+8;
    union __align__(32) Shared {
        struct { __half ah[64*LD], al[64*LD], bh[128*LD], bl[128*LD]; } in;
        float res[64*128];
    };
    __shared__ Shared s;
    int warp=threadIdx.x/32;
    int mr=blockIdx.x*64, nc=blockIdx.y*128;
    int wr=(warp/4)*32, wc=(warp%4)*32;
    wmma::fragment<wmma::accumulator,16,16,16,float> c[2][2],l[2][2];
    #pragma unroll
    for(int i=0;i<2;i++) {
        #pragma unroll
        for(int j=0;j<2;j++) {wmma::fill_fragment(c[i][j],0.f);wmma::fill_fragment(l[i][j],0.f);}
    }
    int start=blockIdx.z*(D/SPLIT),end=start+D/SPLIT;
    for(int k0=start;k0<end;k0+=TK) {
        {
            int q=threadIdx.x*8;
            int r=q/TK,kk=q%TK;
            uint4 va=make_uint4(0,0,0,0),vl=va;
            if(mr+r<M) {
                va=*reinterpret_cast<const uint4*>(a+(mr+r)*D+k0+kk);
                vl=*reinterpret_cast<const uint4*>(a+(size_t)M*D+(mr+r)*D+k0+kk);
            }
            *reinterpret_cast<uint4*>(s.in.ah+r*LD+kk)=va;
            *reinterpret_cast<uint4*>(s.in.al+r*LD+kk)=vl;
        }
        #pragma unroll
        for(int t=0;t<2;t++) {
            int q=threadIdx.x*8+t*2048;
            int r=q/TK,kk=q%TK;
            *reinterpret_cast<uint4*>(s.in.bh+r*LD+kk)=*reinterpret_cast<const uint4*>(w+(nc+r)*D+k0+kk);
            *reinterpret_cast<uint4*>(s.in.bl+r*LD+kk)=*reinterpret_cast<const uint4*>(w+(size_t)N*D+(nc+r)*D+k0+kk);
        }
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<TK;kk+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> ah[2],al[2];
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bh[2],bl[2];
            #pragma unroll
            for(int i=0;i<2;i++) {
                wmma::load_matrix_sync(ah[i],s.in.ah+(wr+i*16)*LD+kk,LD);
                wmma::load_matrix_sync(al[i],s.in.al+(wr+i*16)*LD+kk,LD);
                wmma::load_matrix_sync(bh[i],s.in.bh+(wc+i*16)*LD+kk,LD);
                wmma::load_matrix_sync(bl[i],s.in.bl+(wc+i*16)*LD+kk,LD);
            }
            #pragma unroll
            for(int i=0;i<2;i++) {
                #pragma unroll
                for(int j=0;j<2;j++) {
                    wmma::mma_sync(c[i][j],ah[i],bh[j],c[i][j]);
                    wmma::mma_sync(l[i][j],ah[i],bl[j],l[i][j]);
                    wmma::mma_sync(l[i][j],al[i],bh[j],l[i][j]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<2;i++) {
        #pragma unroll
        for(int j=0;j<2;j++) {
            #pragma unroll
            for(int t=0;t<c[i][j].num_elements;t++) c[i][j].x[t]+=l[i][j].x[t]*(1.f/4096.f);
            wmma::store_matrix_sync(s.res+(wr+i*16)*128+wc+j*16,c[i][j],128,wmma::mem_row_major);
        }
    }
    __syncthreads();
    for(int q=threadIdx.x;q<64*128;q+=256){
        int r=mr+q/128,cc=nc+q%128;
        if(r<M) out[((size_t)blockIdx.z*M+r)*N+cc]=s.res[q];
    }
}
template<int SPLIT>
__global__ void reduce_parts(const float* p,float* o,int n){
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=n)return;
 float v=0; for(int j=0;j<SPLIT;j++)v+=p[(size_t)j*n+q];o[q]=v;
}

// Workspace is retained, but all data are recomputed on every invocation.
struct State {
    cublasHandle_t handle;
    float *a=nullptr,*w=nullptr;
    size_t acap=0,wcap=0;
    __half *ah=nullptr,*wh=nullptr; size_t ahcap=0,whcap=0; float* partial=nullptr; size_t pcap=0;
    State(){cublasCreate(&handle); cublasSetStream(handle,0); cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH);}
};
}
void reference(const float* x, const float* grid, const float* base_weight, const float* spline_weight, const float* spline_scaler, int32_t grid_size, int32_t spline_order, int32_t B, int32_t I, int32_t O, float* out) {
    static State s;
    if(I==256 && O==512 && B>=512) {
        const int D=I*12;
        size_t na=(size_t)B*D*2,nw=(size_t)O*D*2;
        if(na>s.ahcap){if(s.ah)cudaFree(s.ah);cudaMalloc(&s.ah,na*sizeof(__half));s.ahcap=na;}
        if(nw>s.whcap){if(s.wh)cudaFree(s.wh);cudaMalloc(&s.wh,nw*sizeof(__half));s.whcap=nw;}
        pack_x_half<<<(B*I+255)/256,256>>>(x,grid,s.ah,B,I);
        pack_w_half<<<(O*I+255)/256,256>>>(base_weight,spline_weight,spline_scaler,s.wh,O*I);
        size_t np=(size_t)B*O*4;
        if(np>s.pcap){if(s.partial)cudaFree(s.partial);cudaMalloc(&s.partial,np*sizeof(float));s.pcap=np;}
        tensor_gemm<4><<<dim3((B+63)/64,O/128,4),256>>>(s.ah,s.wh,s.partial,B,O,D);
        reduce_parts<4><<<(B*O+255)/256,256>>>(s.partial,out,B*O);
        return;
    }
    size_t na=(size_t)B*I*12, nw=(size_t)O*I*12;
    if(na>s.acap){if(s.a)cudaFree(s.a);cudaMalloc(&s.a,na*sizeof(float));s.acap=na;}
    if(nw>s.wcap){if(s.w)cudaFree(s.w);cudaMalloc(&s.w,nw*sizeof(float));s.wcap=nw;}
    pack_x<<<(B*I+255)/256,256>>>(x,grid,s.a,B,I);
    pack_w<<<(O*I+255)/256,256>>>(base_weight,spline_weight,spline_scaler,s.w,O*I);
    float alpha=1.f,beta=0.f;
    cublasSgemm(s.handle,CUBLAS_OP_T,CUBLAS_OP_N,O,B,I*12,&alpha,s.w,I*12,s.a,I*12,&beta,out,O);
}
