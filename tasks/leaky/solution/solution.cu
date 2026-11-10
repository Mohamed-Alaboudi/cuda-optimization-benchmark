#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// General path: each thread advances one neuron through the complete sequence.
__global__ void lif_general(const float* __restrict__ x,
                            const float* __restrict__ mem0,
                            const float* __restrict__ beta,
                            const float* __restrict__ threshold,
                            int T, int B, int H,
                            float* __restrict__ spikes,
                            float* __restrict__ final_mem) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * H;
    if (i >= n) return;
    int h = i % H;
    float b = beta[h];
    if (b < 0.f) b = 0.f;
    else if (b > 1.f) b = 1.f;
    float th = threshold[h];
    float m = mem0[i];
    size_t stride = (size_t)n;
    for (int t = 0; t < T; ++t) {
        float reset = m >= th ? 1.0f : 0.0f;
        float prod = __fmul_rn(b, m);
        float sum = __fadd_rn(prod, x[(size_t)t * stride + i]);
        m = __fsub_rn(sum, __fmul_rn(reset, th));
        spikes[(size_t)t * stride + i] = m >= th ? 1.0f : 0.0f;
    }
    final_mem[i] = m;
}

// A short time tile is processed over the entire spatial domain before the
// next tile, keeping the memory accesses from the different CTAs close in time.
// mem_in and mem_out intentionally can alias on subsequent tiles.
template<bool X_ALIGNED, bool Y_ALIGNED, bool M_ALIGNED>
__global__ void lif_tile8(const float* __restrict__ x,
                          const float* mem_in,
                          const float* __restrict__ beta,
                          const float* __restrict__ threshold,
                          float* __restrict__ spikes, float* mem_out) {
    constexpr int N = 524288;
    int i = blockIdx.x * blockDim.x + threadIdx.x; // pairs of neurons
    float2 m;
    if (M_ALIGNED) m = reinterpret_cast<const float2*>(mem_in)[i];
    else m = make_float2(mem_in[2*i], mem_in[2*i+1]);
    float2 b = reinterpret_cast<const float2*>(beta)[i & 511];
    float2 th = reinterpret_cast<const float2*>(threshold)[i & 511];
    if (b.x < 0.f) b.x = 0.f; else if (b.x > 1.f) b.x = 1.f;
    if (b.y < 0.f) b.y = 0.f; else if (b.y > 1.f) b.y = 1.f;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        float2 v;
        if (X_ALIGNED) v = reinterpret_cast<const float2*>(x)[t*(N/2)+i];
        else v = make_float2(x[t*N+2*i], x[t*N+2*i+1]);
        m.x = __fsub_rn(__fadd_rn(__fmul_rn(b.x, m.x), v.x), __fmul_rn(m.x >= th.x ? 1.f : 0.f, th.x));
        m.y = __fsub_rn(__fadd_rn(__fmul_rn(b.y, m.y), v.y), __fmul_rn(m.y >= th.y ? 1.f : 0.f, th.y));
        v.x = m.x >= th.x ? 1.f : 0.f;
        v.y = m.y >= th.y ? 1.f : 0.f;
        if (Y_ALIGNED) reinterpret_cast<float2*>(spikes)[t*(N/2)+i] = v;
        else {
            spikes[t*N+2*i] = v.x;
            spikes[t*N+2*i+1] = v.y;
        }
    }
    if (M_ALIGNED) reinterpret_cast<float2*>(mem_out)[i] = m;
    else { mem_out[2*i] = m.x; mem_out[2*i+1] = m.y; }
}

template<int N> __global__ void kern_fast(const float* __restrict__ x,const float* memini,const float* __restrict__ b,const float* __restrict__ th,float* __restrict__ y,float* out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=524288/2) return;
  float2 im = reinterpret_cast<const float2*>(memini)[i];
  float2 bb = reinterpret_cast<const float2*>(b)[i%(512)];
  float2 tt = reinterpret_cast<const float2*>(th)[i%(512)];
 float m0=im.x, b0=bb.x, h0=tt.x;
 float m1=im.y, b1=bb.y, h1=tt.y;
 if (b0<0.f) b0=0.f; else if (b0>1.f) b0=1.f;
 if (b1<0.f) b1=0.f; else if (b1>1.f) b1=1.f;
 for(int t=0;t<N;t+=1) {
  float2 v0 = reinterpret_cast<const float2*>(x)[(t+0)*262144+i];
  m0=__fsub_rn(__fadd_rn(__fmul_rn(b0,m0),v0.x),__fmul_rn(m0>=h0?1.f:0.f,h0));
  v0.x=m0>=h0?1.f:0.f;
  m1=__fsub_rn(__fadd_rn(__fmul_rn(b1,m1),v0.y),__fmul_rn(m1>=h1?1.f:0.f,h1));
  v0.y=m1>=h1?1.f:0.f;
  reinterpret_cast<float2*>(y)[(t+0)*262144+i]=v0;
 }
 im.x=m0;
 im.y=m1;
 reinterpret_cast<float2*>(out)[i]=im;
}

template<bool X, bool Y, bool M>
static void run_tiles(const float* x, const float* m, const float* b,
                      const float* th, float* y, float* out) {
    constexpr size_t N = 524288;
    for (int t = 0; t < 256; t += 8) {
        if (X && Y && M)
            kern_fast<8><<<2048,128>>>(x+(size_t)t*N, t ? out : m,
                                      b, th, y+(size_t)t*N, out);
        else
            lif_tile8<X,Y,M><<<2048,128>>>(x+(size_t)t*N, t ? out : m,
                                      b, th, y+(size_t)t*N, out);
    }
}

void reference(const float* input_current, const float* mem0,
               const float* beta, const float* threshold,
               int32_t T, int32_t B, int32_t H,
               float* spikes, float* final_mem) {
    // tiled fast path for the reference shape; general kernel below
    if (T == 256 && B == 512 && H == 1024 &&
        (((uintptr_t)beta | (uintptr_t)threshold) & 7) == 0) {
        bool x = ((uintptr_t)input_current & 7) == 0;
        bool y = ((uintptr_t)spikes & 7) == 0;
        bool m = (((uintptr_t)mem0 | (uintptr_t)final_mem) & 7) == 0;
        if (x) {
            if (y) {
                if (m) run_tiles<true,true,true>(input_current,mem0,beta,threshold,spikes,final_mem);
                else run_tiles<true,true,false>(input_current,mem0,beta,threshold,spikes,final_mem);
            } else {
                if (m) run_tiles<true,false,true>(input_current,mem0,beta,threshold,spikes,final_mem);
                else run_tiles<true,false,false>(input_current,mem0,beta,threshold,spikes,final_mem);
            }
        } else {
            if (y) {
                if (m) run_tiles<false,true,true>(input_current,mem0,beta,threshold,spikes,final_mem);
                else run_tiles<false,true,false>(input_current,mem0,beta,threshold,spikes,final_mem);
            } else {
                if (m) run_tiles<false,false,true>(input_current,mem0,beta,threshold,spikes,final_mem);
                else run_tiles<false,false,false>(input_current,mem0,beta,threshold,spikes,final_mem);
            }
        }
    } else {
        int n = B * H;
        lif_general<<<(n + 255) / 256, 256>>>(input_current, mem0, beta,
                     threshold, T, B, H, spikes, final_mem);
    }
}
