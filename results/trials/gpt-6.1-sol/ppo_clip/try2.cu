#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace ppo_detail {
struct Stats {
    float gain, clipped, kl, sum, sumsq, maximum;
};

__device__ __forceinline__ Stats empty_stats() {
    return {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

__device__ __forceinline__ Stats warp_reduce(Stats x) {
    #pragma unroll
    for (int d = 16; d; d >>= 1) {
        x.gain += __shfl_down_sync(0xffffffff, x.gain, d);
        x.clipped += __shfl_down_sync(0xffffffff, x.clipped, d);
        x.kl += __shfl_down_sync(0xffffffff, x.kl, d);
        x.sum += __shfl_down_sync(0xffffffff, x.sum, d);
        x.sumsq += __shfl_down_sync(0xffffffff, x.sumsq, d);
        x.maximum = fmaxf(x.maximum, __shfl_down_sync(0xffffffff, x.maximum, d));
    }
    return x;
}

__device__ __forceinline__ Stats block_reduce(Stats x, Stats* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    x = warp_reduce(x);
    if (lane == 0) smem[warp] = x;
    __syncthreads();
    if (warp == 0) {
        x = lane < blockDim.x / 32 ? smem[lane] : empty_stats();
        x = warp_reduce(x);
    }
    return x;
}

__device__ __forceinline__ void write_result(Stats x, int64_t n, float* out) {
    const float inv = 1.f / (float)n;
    out[0] = -x.gain * inv;
    out[1] = x.clipped * inv;
    out[2] = x.kl * inv;
    // Equivalent to exp(2*logsumexp(lw)-logsumexp(2*lw))/N.
    // The bounded input log weights ensure these sums cannot overflow.
    out[3] = (x.sum * inv) * (x.sum / x.sumsq);
    out[4] = x.maximum;
    out[5] = x.sum * inv;
}

__device__ __forceinline__ void accumulate(Stats &x, float c, float o, float a,
                                         float low_log, float high_log,
                                         float low_ratio, float high_ratio) {
    float lw = c - o;
    float r = __expf(lw);
    float rc = fminf(fmaxf(r, low_ratio), high_ratio);
    x.gain += fminf(r * a, rc * a);
    x.clipped += (lw < low_log || lw > high_log) ? 1.f : 0.f;
    x.kl += (r - 1.f) - lw;
    x.sum += r;
    x.sumsq += r * r;
    x.maximum = fmaxf(x.maximum, r);
}

template<bool FINAL>
__global__ void first_pass(int64_t n, const float* __restrict__ current,
                           const float* __restrict__ old,
                           const float* __restrict__ advantage,
                           float low_log, float high_log,
                           float low_ratio, float high_ratio,
                           Stats* partial, float* output) {
    Stats x = empty_stats();
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        accumulate(x, current[idx], old[idx], advantage[idx], low_log, high_log, low_ratio, high_ratio);
    }
    __shared__ Stats smem[8];
    x = block_reduce(x, smem);
    if (threadIdx.x == 0) {
        if (FINAL) write_result(x, n, output);
        else partial[blockIdx.x] = x;
    }
}

__global__ void finish(const Stats* __restrict__ partial, int blocks, int64_t n, float* output) {
    Stats x = empty_stats();
    for (int i = threadIdx.x; i < blocks; i += blockDim.x) {
        Stats y = partial[i];
        x.gain += y.gain;
        x.clipped += y.clipped;
        x.kl += y.kl;
        x.sum += y.sum;
        x.sumsq += y.sumsq;
        x.maximum = fmaxf(x.maximum, y.maximum);
    }
    __shared__ Stats smem[8];
    x = block_reduce(x, smem);
    if (threadIdx.x == 0) write_result(x, n, output);
}
} // namespace ppo_detail

void reference(const int32_t B, const int32_t T, const int32_t H,
               const float* current_log_prob, const float* old_log_prob,
               const float* advantage, const float clip_epsilon, float* output) {
    const int64_t n = (int64_t)B * T * H;
    const float low_log = log1pf(-clip_epsilon);
    const float high_log = log1pf(clip_epsilon);
    const float low_ratio = 1.f - clip_epsilon;
    const float high_ratio = 1.f + clip_epsilon;
    if (n <= 2048) {
        ppo_detail::first_pass<true><<<1,256>>>(n, current_log_prob, old_log_prob, advantage,
            low_log, high_log, low_ratio, high_ratio, nullptr, output);
    } else {
        static ppo_detail::Stats* workspace = nullptr;
        if (!workspace) cudaMalloc(&workspace, 1024 * sizeof(ppo_detail::Stats));
        int blocks = (int)((n + 2047) / 2048);
        if (blocks > 1024) blocks = 1024;
        ppo_detail::first_pass<false><<<blocks,256>>>(n, current_log_prob, old_log_prob, advantage,
            low_log, high_log, low_ratio, high_ratio, workspace, output);
        ppo_detail::finish<<<1,256>>>(workspace, blocks, n, output);
    }
}
