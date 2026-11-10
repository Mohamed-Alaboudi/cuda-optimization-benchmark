#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// The separate rounded operations are intentional: the reference computes the
// product and sum separately rather than using a fused multiply-add.
__device__ __forceinline__ float lif_step(float mem, float x, float b, float th) {
    float reset = mem >= th ? th : 0.0f;
    return __fsub_rn(__fadd_rn(__fmul_rn(b, mem), x), reset);
}

__global__ void lif_scalar(const float* __restrict__ x,
                           const float* __restrict__ m0,
                           const float* __restrict__ beta,
                           const float* __restrict__ threshold,
                           int T, int H, size_t N,
                           float* __restrict__ spikes,
                           float* __restrict__ final_mem) {
    size_t neuron = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= N) return;
    int h = neuron % H;
    float b = fminf(fmaxf(beta[h], 0.0f), 1.0f);
    float th = threshold[h];
    float mem = m0[neuron];
    size_t i = neuron;
    #pragma unroll 1
    for (int t = 0; t < T; ++t) {
        mem = lif_step(mem, x[i], b, th);
        spikes[i] = mem >= th ? 1.0f : 0.0f;
        i += N;
    }
    final_mem[neuron] = mem;
}

// Four independent neurons per thread make the large streaming case use
// 128-bit memory operations, with the recurrent states kept in registers.
__global__ void lif_vector4(const float* __restrict__ x,
                            const float* __restrict__ m0,
                            const float* __restrict__ beta,
                            const float* __restrict__ threshold,
                            int T, int H, size_t N,
                            float* __restrict__ spikes,
                            float* __restrict__ final_mem) {
    size_t neuron4 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron4 >= N / 4) return;
    size_t i = neuron4 * 4;
    int h = i % H;
    float b[4], th[4], mem[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        b[j] = fminf(fmaxf(beta[h+j], 0.0f), 1.0f);
        th[j] = threshold[h+j];
        mem[j] = m0[i+j];
    }
    #pragma unroll 4
    for (int t = 0; t < T; ++t) {
        float4 xv = reinterpret_cast<const float4*>(x)[i / 4];
        mem[0] = lif_step(mem[0], xv.x, b[0], th[0]);
        mem[1] = lif_step(mem[1], xv.y, b[1], th[1]);
        mem[2] = lif_step(mem[2], xv.z, b[2], th[2]);
        mem[3] = lif_step(mem[3], xv.w, b[3], th[3]);
        float4 sv = make_float4(mem[0] >= th[0] ? 1.0f : 0.0f,
                               mem[1] >= th[1] ? 1.0f : 0.0f,
                               mem[2] >= th[2] ? 1.0f : 0.0f,
                               mem[3] >= th[3] ? 1.0f : 0.0f);
        reinterpret_cast<float4*>(spikes)[i / 4] = sv;
        i += N;
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) final_mem[neuron4 * 4 + j] = mem[j];
}

void reference(const float* input_current, const float* mem0,
               const float* beta, const float* threshold,
               int32_t T, int32_t B, int32_t H,
               float* spikes, float* final_mem) {
    size_t N = (size_t)B * H;
    if (N == 0) return;
    uintptr_t vector_pointers = reinterpret_cast<uintptr_t>(input_current) |
                                reinterpret_cast<uintptr_t>(spikes);
    if (N >= 65536 && H % 4 == 0 && (vector_pointers & 15) == 0) {
        lif_vector4<<<(N / 4 + 127) / 128, 128>>>(input_current, mem0, beta, threshold,
                                                T, H, N, spikes, final_mem);
    } else {
        lif_scalar<<<(N + 255) / 256, 256>>>(input_current, mem0, beta, threshold,
                                           T, H, N, spikes, final_mem);
    }
}
