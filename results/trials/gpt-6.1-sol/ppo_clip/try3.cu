#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {
constexpr int BLOCK = 256;
constexpr int MAX_BLOCKS = 2048;
__device__ float partials[6][MAX_BLOCKS];

struct Stats {
    float loss, clipped, kl, sum, square, maximum;
};

__device__ __forceinline__ Stats zero_stats() {
    return {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

__device__ __forceinline__ void add_element(Stats &s, float cur, float old, float a,
                                           float low, float high, float eps) {
    const float w = cur - old;
    const float r = __expf(w);
    // min(r*a, clamp(r, 1-eps, 1+eps)*a), with the sign of a handled first.
    const float gain_ratio = (a >= 0.f) ? fminf(r, 1.f + eps) : fmaxf(r, 1.f - eps);
    s.loss -= gain_ratio * a;
    s.clipped += (w < low || w > high) ? 1.f : 0.f;
    s.kl += (r - 1.f) - w;
    s.sum += r;
    s.square += r * r;
    s.maximum = fmaxf(s.maximum, r);
}

__device__ __forceinline__ Stats warp_reduce(Stats s) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        s.loss += __shfl_down_sync(0xffffffff, s.loss, offset);
        s.clipped += __shfl_down_sync(0xffffffff, s.clipped, offset);
        s.kl += __shfl_down_sync(0xffffffff, s.kl, offset);
        s.sum += __shfl_down_sync(0xffffffff, s.sum, offset);
        s.square += __shfl_down_sync(0xffffffff, s.square, offset);
        s.maximum = fmaxf(s.maximum, __shfl_down_sync(0xffffffff, s.maximum, offset));
    }
    return s;
}

__device__ __forceinline__ Stats block_reduce(Stats s) {
    __shared__ Stats warps[BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    s = warp_reduce(s);
    if (lane == 0) warps[warp] = s;
    __syncthreads();
    s = zero_stats();
    if (warp == 0) {
        if (lane < BLOCK / 32) s = warps[lane];
        s = warp_reduce(s);
    }
    return s;
}

__device__ __forceinline__ void store_output(Stats s, int64_t n, float *out) {
    const float inv = 1.f / (float)n;
    out[0] = s.loss * inv;
    out[1] = s.clipped * inv;
    out[2] = s.kl * inv;
    // ESS = (sum exp(lw))^2 / (N * sum exp(2*lw)).
    // The specified [-20,20] weights keep both sums finite in fp32.
    out[3] = (s.sum / s.square) * (s.sum * inv);
    out[4] = s.maximum;
    out[5] = s.sum * inv;
}

template<bool SINGLE>
__global__ void calculate(int64_t n, const float *__restrict__ cur,
                          const float *__restrict__ old, const float *__restrict__ adv,
                          float eps, float low, float high, float *out) {
    Stats s = zero_stats();
    for (int64_t i = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
         i < n; i += (int64_t)gridDim.x * BLOCK) {
        add_element(s, cur[i], old[i], adv[i], low, high, eps);
    }
    s = block_reduce(s);
    if (threadIdx.x == 0) {
        if (SINGLE) store_output(s, n, out);
        else {
            partials[0][blockIdx.x] = s.loss;
            partials[1][blockIdx.x] = s.clipped;
            partials[2][blockIdx.x] = s.kl;
            partials[3][blockIdx.x] = s.sum;
            partials[4][blockIdx.x] = s.square;
            partials[5][blockIdx.x] = s.maximum;
        }
    }
}

__global__ void finish(int blocks, int64_t n, float *out) {
    Stats s = zero_stats();
    for (int i = threadIdx.x; i < blocks; i += BLOCK) {
        s.loss += partials[0][i];
        s.clipped += partials[1][i];
        s.kl += partials[2][i];
        s.sum += partials[3][i];
        s.square += partials[4][i];
        s.maximum = fmaxf(s.maximum, partials[5][i]);
    }
    s = block_reduce(s);
    if (threadIdx.x == 0) store_output(s, n, out);
}
}

void reference(const int32_t B, const int32_t T, const int32_t H,
               const float* current_log_prob, const float* old_log_prob,
               const float* advantage, const float clip_epsilon, float* output) {
    const int64_t n = (int64_t)B * T * H;
    const float low = log1pf(-clip_epsilon);
    const float high = log1pf(clip_epsilon);
    if (n <= 1024) {
        calculate<true><<<1, BLOCK>>>(n, current_log_prob, old_log_prob, advantage,
                                      clip_epsilon, low, high, output);
    } else {
        int blocks = (int)((n + 1023) / 1024);
        if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
        calculate<false><<<blocks, BLOCK>>>(n, current_log_prob, old_log_prob, advantage,
                                            clip_epsilon, low, high, output);
        finish<<<1, BLOCK>>>(blocks, n, output);
    }
}
