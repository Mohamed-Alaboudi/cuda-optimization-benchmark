#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ float clampb(float b){ return b < 0.f ? 0.f : (b > 1.f ? 1.f : b); }

__device__ __forceinline__ float step(float m, float x, float b, float th, float &s){
    float r = (m >= th) ? 1.f : 0.f;
    float nm = __fsub_rn(__fadd_rn(__fmul_rn(b, m), x), __fmul_rn(r, th));
    s = (nm >= th) ? 1.f : 0.f;
    return nm;
}

__global__ void lif_vec4(const float4* __restrict__ x, const float4* __restrict__ mem0,
                         const float4* __restrict__ beta, const float4* __restrict__ th,
                         int T, long long BH4, int H4, float4* __restrict__ spk, float4* __restrict__ fm){
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BH4) return;
    int h = (int)(i % H4);
    float4 b = __ldg(&beta[h]); b.x=clampb(b.x); b.y=clampb(b.y); b.z=clampb(b.z); b.w=clampb(b.w);
    float4 t4 = __ldg(&th[h]);
    float4 m = mem0[i];
    const float4* xp = x + i; float4* sp = spk + i;
    int t = 0;
    for (; t + 8 <= T; t += 8) {
        float4 xv[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) xv[k] = __ldcs(xp + (long long)(t + k) * BH4);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float4 s;
            m.x = step(m.x, xv[k].x, b.x, t4.x, s.x);
            m.y = step(m.y, xv[k].y, b.y, t4.y, s.y);
            m.z = step(m.z, xv[k].z, b.z, t4.z, s.z);
            m.w = step(m.w, xv[k].w, b.w, t4.w, s.w);
            __stcs(sp + (long long)(t + k) * BH4, s);
        }
    }
    for (; t < T; ++t) {
        float4 xv = __ldcs(xp + (long long)t * BH4);
        float4 s;
        m.x = step(m.x, xv.x, b.x, t4.x, s.x);
        m.y = step(m.y, xv.y, b.y, t4.y, s.y);
        m.z = step(m.z, xv.z, b.z, t4.z, s.z);
        m.w = step(m.w, xv.w, b.w, t4.w, s.w);
        __stcs(sp + (long long)t * BH4, s);
    }
    fm[i] = m;
}

__global__ void lif_scalar(const float* __restrict__ x, const float* __restrict__ mem0,
                           const float* __restrict__ beta, const float* __restrict__ th,
                           int T, long long BH, int H, float* __restrict__ spk, float* __restrict__ fm){
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BH) return;
    int h = (int)(i % H);
    float b = clampb(beta[h]); float tt = th[h];
    float m = mem0[i];
    int t = 0;
    for (; t + 8 <= T; t += 8) {
        float xv[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) xv[k] = __ldcs(x + (long long)(t + k) * BH + i);
#pragma unroll
        for (int k = 0; k < 8; ++k) { float s; m = step(m, xv[k], b, tt, s); __stcs(spk + (long long)(t + k) * BH + i, s); }
    }
    for (; t < T; ++t) { float s; m = step(m, __ldcs(x + (long long)t * BH + i), b, tt, s); spk[(long long)t * BH + i] = s; }
    fm[i] = m;
}

void reference(const float* input_current, const float* mem0, const float* beta, const float* threshold,
               int32_t T, int32_t B, int32_t H, float* spikes, float* final_mem) {
    long long BH = (long long)B * H;
    if (BH <= 0) return;
    const int th = 256;
    bool al = ((((uintptr_t)input_current) | ((uintptr_t)mem0) | ((uintptr_t)beta) | ((uintptr_t)threshold) |
                ((uintptr_t)spikes) | ((uintptr_t)final_mem)) & 15) == 0;
    if (H % 4 == 0 && al) {
        long long n = BH / 4;
        lif_vec4<<<(unsigned)((n + th - 1) / th), th>>>((const float4*)input_current, (const float4*)mem0,
            (const float4*)beta, (const float4*)threshold, T, n, H / 4, (float4*)spikes, (float4*)final_mem);
    } else {
        lif_scalar<<<(unsigned)((BH + th - 1) / th), th>>>(input_current, mem0, beta, threshold, T, BH, H, spikes, final_mem);
    }
}
