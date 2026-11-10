// Leaky integrate-and-fire (LIF) layer, forward pass.
// One thread per (B,H) lane; each thread walks the T timesteps sequentially.
// Consecutive threads map to consecutive H -> coalesced reads/writes.
// Explicit IEEE intrinsics (__fmul_rn/__fadd_rn/__fsub_rn) prevent FMA
// contraction so results match the fp32 eager reference bit-for-bit.

#include <cuda_runtime.h>
#include <cstdint>

__global__ void lif_forward_kernel(const float* __restrict__ input_current,
                                   const float* __restrict__ mem0,
                                   const float* __restrict__ beta,
                                   const float* __restrict__ threshold,
                                   int32_t T, int32_t B, int32_t H,
                                   float* __restrict__ spikes,
                                   float* __restrict__ final_mem) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t BH = (int64_t)B * (int64_t)H;
    if (idx >= BH) return;

    int32_t h = (int32_t)(idx % H);

    // Clamp beta to [0, 1], propagating NaN like torch.clamp does.
    float b = beta[h];
    if (b < 0.0f) b = 0.0f;
    else if (b > 1.0f) b = 1.0f;

    const float th = threshold[h];
    float mem = mem0[idx];

    const float* x = input_current + idx;
    float* s = spikes + idx;

    for (int32_t t = 0; t < T; ++t) {
        const float xt = *x;
        const float reset = (mem >= th) ? 1.0f : 0.0f;
        mem = __fsub_rn(__fadd_rn(__fmul_rn(b, mem), xt), __fmul_rn(reset, th));
        *s = (mem >= th) ? 1.0f : 0.0f;
        x += BH;
        s += BH;
    }
    final_mem[idx] = mem;
}

void reference(const float* input_current, const float* mem0, const float* beta,
               const float* threshold, int32_t T, int32_t B, int32_t H,
               float* spikes, float* final_mem) {
    int64_t BH = (int64_t)B * (int64_t)H;
    if (BH <= 0 || T <= 0) {
        if (BH > 0) {
            // T == 0: no timesteps, final_mem = mem0, spikes empty.
            cudaMemcpyAsync(final_mem, mem0, sizeof(float) * (size_t)BH,
                            cudaMemcpyDeviceToDevice, 0);
        }
        return;
    }
    const int threads = 256;
    const int64_t blocks = (BH + threads - 1) / threads;
    lif_forward_kernel<<<(unsigned int)blocks, threads, 0, 0>>>(
        input_current, mem0, beta, threshold, T, B, H, spikes, final_mem);
}
