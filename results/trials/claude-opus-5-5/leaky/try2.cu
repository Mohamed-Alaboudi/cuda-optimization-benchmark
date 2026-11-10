#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ float clamp01(float b) { return b < 0.f ? 0.f : (b > 1.f ? 1.f : b); }

__device__ __forceinline__ float lif_step(float m, float b, float th, float x, float& s) {
    float reset = (m >= th) ? 1.0f : 0.0f;
    float nm = __fsub_rn(__fadd_rn(__fmul_rn(b, m), x), __fmul_rn(reset, th));
    s = (nm >= th) ? 1.0f : 0.0f;
    return nm;
}

#define U 8
__global__ void lif_vec4(const float4* __restrict__ x, const float4* __restrict__ mem0,
                         const float* __restrict__ beta, const float* __restrict__ th,
                         int T, int BH4, int H4, float4* __restrict__ sp, float4* __restrict__ fm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BH4) return;
    int h = (i % H4) * 4;
    float b0 = clamp01(beta[h]), b1 = clamp01(beta[h+1]), b2 = clamp01(beta[h+2]), b3 = clamp01(beta[h+3]);
    float t0 = th[h], t1 = th[h+1], t2 = th[h+2], t3 = th[h+3];
    float4 m = mem0[i];
    size_t stride = (size_t)BH4;
    int t = 0;
    for (; t + U <= T; t += U) {
        float4 v[U];
#pragma unroll
        for (int k = 0; k < U; k++) v[k] = __ldcs(&x[(size_t)(t + k) * stride + i]);
#pragma unroll
        for (int k = 0; k < U; k++) {
            float4 s;
            m.x = lif_step(m.x, b0, t0, v[k].x, s.x);
            m.y = lif_step(m.y, b1, t1, v[k].y, s.y);
            m.z = lif_step(m.z, b2, t2, v[k].z, s.z);
            m.w = lif_step(m.w, b3, t3, v[k].w, s.w);
            __stcs(&sp[(size_t)(t + k) * stride + i], s);
        }
    }
    for (; t < T; t++) {
        float4 v = __ldcs(&x[(size_t)t * stride + i]);
        float4 s;
        m.x = lif_step(m.x, b0, t0, v.x, s.x);
        m.y = lif_step(m.y, b1, t1, v.y, s.y);
        m.z = lif_step(m.z, b2, t2, v.z, s.z);
        m.w = lif_step(m.w, b3, t3, v.w, s.w);
        __stcs(&sp[(size_t)t * stride + i], s);
    }
    fm[i] = m;
}

__global__ void lif_scalar(const float* __restrict__ x, const float* __restrict__ mem0,
                           const float* __restrict__ beta, const float* __restrict__ th,
                           int T, int BH, int H, float* __restrict__ sp, float* __restrict__ fm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BH) return;
    int h = i % H;
    float b = clamp01(beta[h]), tt = th[h];
    float m = mem0[i];
    size_t stride = (size_t)BH;
    int t = 0;
    for (; t + U <= T; t += U) {
        float v[U];
#pragma unroll
        for (int k = 0; k < U; k++) v[k] = __ldcs(&x[(size_t)(t + k) * stride + i]);
#pragma unroll
        for (int k = 0; k < U; k++) {
            float s;
            m = lif_step(m, b, tt, v[k], s);
            __stcs(&sp[(size_t)(t + k) * stride + i], s);
        }
    }
    for (; t < T; t++) {
        float s;
        m = lif_step(m, b, tt, __ldcs(&x[(size_t)t * stride + i]), s);
        __stcs(&sp[(size_t)t * stride + i], s);
    }
    fm[i] = m;
}

void reference(const float* input_current, const float* mem0, const float* beta, const float* threshold,
               int32_t T, int32_t B, int32_t H, float* spikes, float* final_mem) {
    int BH = B * H;
    if (BH <= 0) return;
    bool aligned = ((uintptr_t)input_current | (uintptr_t)mem0 | (uintptr_t)spikes | (uintptr_t)final_mem) % 16 == 0;
    if (H % 4 == 0 && aligned) {
        int BH4 = BH / 4;
        int threads = 128;
        lif_vec4<<<(BH4 + threads - 1) / threads, threads>>>((const float4*)input_current, (const float4*)mem0,
            beta, threshold, T, BH4, H / 4, (float4*)spikes, (float4*)final_mem);
    } else {
        int threads = 128;
        lif_scalar<<<(BH + threads - 1) / threads, threads>>>(input_current, mem0, beta, threshold, T, BH, H,
            spikes, final_mem);
    }
}
