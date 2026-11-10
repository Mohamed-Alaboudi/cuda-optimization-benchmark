#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

__global__ void form_x(const float* x,const float* grid,float* a,int B,int I,int G,int K) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=B*I) return;
    int i=t%I; int C=G+K; int L=G+2*K+1; int D=I*(C+1);
    float v=x[t];
    float* dst=a+(size_t)(t/I)*D+i;
    dst[0]=v/(1.f+expf(-v));
    if(G==8 && K==3) {
        const float* g=grid+i*L;
        float h=g[1]-g[0];
        int m=(int)floorf((v-g[0])/h);
        float u=(v-(g[0]+m*h))/h, z=1.f-u;
        float bs0=z*z*z*(1.f/6.f);
        float bs1=(3.f*u*u*u-6.f*u*u+4.f)*(1.f/6.f);
        float bs2=(-3.f*u*u*u+3.f*u*u+3.f*u+1.f)*(1.f/6.f);
        float bs3=u*u*u*(1.f/6.f);
        #pragma unroll
        for(int j=0;j<11;++j) {
            float val=0.f;
            if(j==m-3)val=bs0;
            if(j==m-2)val=bs1;
            if(j==m-1)val=bs2;
            if(j==m)val=bs3;
            dst[(j+1)*I]=val;
        }
    } else {
        float bs[32];
        for(int j=0;j<L-1;++j) bs[j]=(v>=grid[i*L+j] && v<grid[i*L+j+1]) ? 1.f:0.f;
        for(int k=1;k<=K;++k) {
            for(int j=0;j<L-k-1;++j) {
                bs[j]=(v-grid[i*L+j])/(grid[i*L+j+k]-grid[i*L+j])*bs[j]+(grid[i*L+j+k+1]-v)/(grid[i*L+j+k+1]-grid[i*L+j+1])*bs[j+1];
            }
        }
        for(int j=0;j<C;++j) dst[(j+1)*I]=bs[j];
    }
}

__global__ void form_w(const float* bw,const float* sw,const float* ss,float* w,int I,int O,int C) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=I*O*(C+1)) return;
    int i=t%I,c=(t/I)%(C+1),o=t/(I*(C+1));
    int oi=o*I+i;
    w[t]=c==0?bw[oi]:sw[oi*C+c-1]*ss[oi];
}

// The small-batch path avoids materializing a dense expanded matrix.
__global__ void direct_small(const float* x,const float* grid,const float* bw,
    const float* sw,const float* scaler,int B,int I,int O,float* out) {
    int o=blockIdx.x%O,b=blockIdx.x/O;
    float sum=0.f;
    for(int i=threadIdx.x;i<I;i+=blockDim.x) {
        float v=x[b*I+i];
        const float* g=grid+i*15;
        float h=g[1]-g[0];
        int m=(int)floorf((v-g[0])/h);
        float u=(v-(g[0]+m*h))/h,z=1.f-u;
        float bs[4]={z*z*z*(1.f/6.f),
            (3.f*u*u*u-6.f*u*u+4.f)*(1.f/6.f),
            (-3.f*u*u*u+3.f*u*u+3.f*u+1.f)*(1.f/6.f),
            u*u*u*(1.f/6.f)};
        int oi=o*I+i;
        float val=bw[oi]*(v/(1.f+expf(-v)));
        float scale=scaler[oi];
        #pragma unroll
        for(int r=0;r<4;r++) {
            int j=m-3+r;
            if(j>=0 && j<11) val=fmaf(sw[oi*11+j]*scale,bs[r],val);
        }
        sum+=val;
    }
    #pragma unroll
    for(int s=16;s>0;s>>=1) sum+=__shfl_down_sync(0xffffffff,sum,s);
    __shared__ float warp_sums[8];
    if((threadIdx.x&31)==0) warp_sums[threadIdx.x/32]=sum;
    __syncthreads();
    if(threadIdx.x<32) {
        sum=threadIdx.x<8?warp_sums[threadIdx.x]:0.f;
        #pragma unroll
        for(int s=16;s>0;s>>=1)sum+=__shfl_down_sync(0xffffffff,sum,s);
        if(threadIdx.x==0)out[b*O+o]=sum;
    }
}

void reference(const float* x,const float* grid,const float* base_weight,const float* spline_weight,const float* spline_scaler,int32_t grid_size,int32_t spline_order,int32_t B,int32_t I,int32_t O,float* out) {
    if(B<=8 && grid_size==8 && spline_order==3) {
        direct_small<<<B*O,256>>>(x,grid,base_weight,spline_weight,spline_scaler,B,I,O,out);
        return;
    }
    static cublasHandle_t handle=nullptr;
    if(!handle) {cublasCreate(&handle);cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH);}
    static float *a=nullptr,*w=nullptr;
    static size_t acap=0,wcap=0;
    int C=grid_size+spline_order,D=I*(C+1);
    size_t na=(size_t)B*D,nw=(size_t)O*D;
    if(na>acap) {if(a)cudaFree(a);cudaMalloc(&a,na*sizeof(float));acap=na;}
    if(nw>wcap) {if(w)cudaFree(w);cudaMalloc(&w,nw*sizeof(float));wcap=nw;}
    form_x<<<(B*I+255)/256,256>>>(x,grid,a,B,I,grid_size,spline_order);
    form_w<<<(nw+255)/256,256>>>(base_weight,spline_weight,spline_scaler,w,I,O,C);
    float alpha=1.f,beta=0.f;
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,O,B,D,&alpha,w,D,a,D,&beta,out,O);
}
