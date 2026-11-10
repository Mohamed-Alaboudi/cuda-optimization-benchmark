#include <cuda_runtime.h>
#include <stdint.h>

// One thread per (b, h) column; loops over T sequentially.
// Uses explicit round-to-nearest intrinsics (no FMA contraction) so results
// match the PyTorch fp32 reference bit-for-bit.
__global__ void lif_forward_kernel(const float* __restrict__ input_current,
                                   const float* __restrict__ mem0,
                                   const float* __restrict__ beta,
                                   const float* __restrict__ threshold,
                                   int T, int B, int H,
                                   float* __restrict__ spikes,
                                   float* __restrict__ final_mem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * H;
    if (idx >= n) return;

    int h = idx % H;
    float b = fminf(fmaxf(beta[h], 0.0f), 1.0f);
    float th = threshold[h];
    float mem = mem0[idx];

    const float* x = input_current + idx;
    float* s = spikes + idx;

    for (int t = 0; t < T; ++t) {
        float reset = (mem >= th) ? 1.0f : 0.0f;
        // mem = beta * mem + x[t] - reset * th, with each op rounded (no FMA)
        mem = __fsub_rn(__fadd_rn(__fmul_rn(b, mem), *x), __fmul_rn(reset, th));
        *s = (mem >= th) ? 1.0f : 0.0f;
        x += n;
        s += n;
    }
    final_mem[idx] = mem;
}

void reference(const float* input_current, const float* mem0, const float* beta,
               const float* threshold, int32_t T, int32_t B, int32_t H,
               float* spikes, float* final_mem)
{
    int n = B * H;
    if (n <= 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    lif_forward_kernel<<<grid, block>>>(input_current, mem0, beta, threshold,
                                        (int)T, (int)B, (int)H, spikes, final_mem);
}
