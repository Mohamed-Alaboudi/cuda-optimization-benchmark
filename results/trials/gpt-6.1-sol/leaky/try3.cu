#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// Separate rounded operations are important: the reference does not fuse the
// multiply with the addition of the input current.
__device__ __forceinline__ float lif_step(float mem, float current,
                                         float beta, float threshold) {
    float reset = __fmul_rn(mem >= threshold ? 1.0f : 0.0f, threshold);
    float decayed = __fmul_rn(beta, mem);
    return __fsub_rn(__fadd_rn(decayed, current), reset);
}

__device__ __forceinline__ float clamp_beta(float b) {
    b = b < 0.0f ? 0.0f : b;
    return b > 1.0f ? 1.0f : b;
}

__global__ void lif_scalar(const float* __restrict__ x,
                           const float* __restrict__ initial,
                           const float* __restrict__ beta,
                           const float* __restrict__ threshold,
                           int T, int H, int N,
                           float* __restrict__ spikes,
                           float* __restrict__ final_mem) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int h = i % H;
    float b = clamp_beta(beta[h]);
    float th = threshold[h];
    float m = initial[i];
    // Unrolling lets independent loads of upcoming currents overlap.
    #pragma unroll 16
    for (int t = 0; t < T; ++t) {
        size_t p = (size_t)t * N + i;
        m = lif_step(m, x[p], b, th);
        spikes[p] = m >= th ? 1.0f : 0.0f;
    }
    final_mem[i] = m;
}

// For a large population, four adjacent neurons per thread provide wide,
// coalesced memory operations while keeping all recurrent state in registers.
__global__ void lif_vector(const float* __restrict__ x,
                           const float* __restrict__ initial,
                           const float* __restrict__ beta,
                           const float* __restrict__ threshold,
                           int T, int H, int N,
                           float* __restrict__ spikes,
                           float* __restrict__ final_mem) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= N) return;
    alignas(16) float m[4], b[4], th[4];
    *reinterpret_cast<float4*>(m) = *reinterpret_cast<const float4*>(initial + i);
    *reinterpret_cast<float4*>(b) = *reinterpret_cast<const float4*>(beta + i % H);
    *reinterpret_cast<float4*>(th) = *reinterpret_cast<const float4*>(threshold + i % H);
    #pragma unroll
    for (int j = 0; j < 4; ++j) b[j] = clamp_beta(b[j]);

    #pragma unroll 16
    for (int t = 0; t < T; ++t) {
        size_t p = (size_t)t * N + i;
        alignas(16) float v[4];
        *reinterpret_cast<float4*>(v) = *reinterpret_cast<const float4*>(x + p);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            m[j] = lif_step(m[j], v[j], b[j], th[j]);
            v[j] = m[j] >= th[j] ? 1.0f : 0.0f;
        }
        *reinterpret_cast<float4*>(spikes + p) = *reinterpret_cast<float4*>(v);
    }
    *reinterpret_cast<float4*>(final_mem + i) = *reinterpret_cast<float4*>(m);
}

} // namespace

void reference(const float* input_current, const float* mem0, const float* beta,
               const float* threshold, int32_t T, int32_t B, int32_t H,
               float* spikes, float* final_mem) {
    int N = B * H;
    if (N == 0) return;
    uintptr_t alignment = reinterpret_cast<uintptr_t>(input_current)
                        | reinterpret_cast<uintptr_t>(mem0)
                        | reinterpret_cast<uintptr_t>(beta)
                        | reinterpret_cast<uintptr_t>(threshold)
                        | reinterpret_cast<uintptr_t>(spikes)
                        | reinterpret_cast<uintptr_t>(final_mem);
    if (N >= 65536 && (H % 4 == 0) && ((alignment & 15) == 0)) {
        lif_vector<<<(N + 1023) / 1024, 256>>>(
            input_current, mem0, beta, threshold, T, H, N, spikes, final_mem);
    } else {
        lif_scalar<<<(N + 127) / 128, 128>>>(
            input_current, mem0, beta, threshold, T, H, N, spikes, final_mem);
    }
}
