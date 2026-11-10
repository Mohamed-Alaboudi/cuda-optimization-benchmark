#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace {

// Explicit rounding prevents a fused multiply-add from changing the trajectory
// (even a very small change can move a spike to a different time step).
__device__ __forceinline__ float lif_step(float m, float x, float b, float th) {
    float reset = m >= th ? 1.0f : 0.0f;
    float decayed = __fmul_rn(b, m);
    float integrated = __fadd_rn(decayed, x);
    return __fsub_rn(integrated, __fmul_rn(reset, th));
}

__device__ __forceinline__ float clamp_beta(float b) {
    b = b < 0.0f ? 0.0f : b;
    return b > 1.0f ? 1.0f : b;
}

__global__ void lif_sequence(const float* __restrict__ input,
                             const float* __restrict__ mem0,
                             const float* __restrict__ beta,
                             const float* __restrict__ threshold,
                             int T, int N, int H,
                             float* __restrict__ spikes,
                             float* __restrict__ final_mem) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float m = mem0[i];
    float b = clamp_beta(beta[i % H]);
    float th = threshold[i % H];
    for (int t = 0; t < T; ++t) {
        size_t p = (size_t)t * N + i;
        m = lif_step(m, input[p], b, th);
        spikes[p] = m >= th ? 1.0f : 0.0f;
    }
    final_mem[i] = m;
}

// Short temporal tiles keep each launch's working set compact.  The state
// between tiles lives in final_mem, so no temporary storage is needed.
template<int Steps>
__global__ void lif_tile(const float* __restrict__ input,
                         const float* __restrict__ mem0,
                         const float* __restrict__ beta,
                         const float* __restrict__ threshold,
                         int N, int H, bool first,
                         float* __restrict__ spikes,
                         float* __restrict__ final_mem) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float m = first ? mem0[i] : final_mem[i];
    float b = clamp_beta(beta[i % H]);
    float th = threshold[i % H];
    float x[Steps];
    #pragma unroll
    for (int t = 0; t < Steps; ++t)
        x[t] = input[(size_t)t * N + i];
    #pragma unroll
    for (int t = 0; t < Steps; ++t) {
        m = lif_step(m, x[t], b, th);
        spikes[(size_t)t * N + i] = m >= th ? 1.0f : 0.0f;
    }
    final_mem[i] = m;
}

} // namespace

void reference(const float* input_current, const float* mem0, const float* beta,
               const float* threshold, int32_t T, int32_t B, int32_t H,
               float* spikes, float* final_mem) {
    int N = B * H;
    if (N <= 0) return;
    const int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (N < 262144 || T < 16) {
        lif_sequence<<<blocks, threads>>>(input_current, mem0, beta, threshold,
                                         T, N, H, spikes, final_mem);
        return;
    }
    int t = 0;
    for (; t + 4 <= T; t += 4) {
        lif_tile<4><<<blocks, threads>>>(input_current + (size_t)t * N,
                                       mem0, beta, threshold, N, H, t == 0,
                                       spikes + (size_t)t * N, final_mem);
    }
    // A tail of up to three timesteps, also useful for non-multiple lengths.
    for (; t < T; ++t) {
        lif_tile<1><<<blocks, threads>>>(input_current + (size_t)t * N,
                                       mem0, beta, threshold, N, H, t == 0,
                                       spikes + (size_t)t * N, final_mem);
    }
}
