#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace ppo_detail {
// Every block produces the same six sufficient statistics; no atomic sums
// are needed, so the reduction order is deterministic.
struct Stats {
    float loss, clips, kl, sum, squares, maximum;
};

__device__ __forceinline__ Stats combine(Stats a, Stats b) {
    a.loss += b.loss;
    a.clips += b.clips;
    a.kl += b.kl;
    a.sum += b.sum;
    a.squares += b.squares;
    a.maximum = fmaxf(a.maximum, b.maximum);
    return a;
}

__device__ __forceinline__ Stats warp_reduce(Stats a) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        Stats b;
        b.loss = __shfl_down_sync(0xffffffff, a.loss, d);
        b.clips = __shfl_down_sync(0xffffffff, a.clips, d);
        b.kl = __shfl_down_sync(0xffffffff, a.kl, d);
        b.sum = __shfl_down_sync(0xffffffff, a.sum, d);
        b.squares = __shfl_down_sync(0xffffffff, a.squares, d);
        b.maximum = __shfl_down_sync(0xffffffff, a.maximum, d);
        a = combine(a, b);
    }
    return a;
}

__device__ __forceinline__ Stats block_reduce(Stats s, Stats* shared) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    s = warp_reduce(s);
    if (lane == 0) shared[warp] = s;
    __syncthreads();
    s = {0, 0, 0, 0, 0, 0};
    if (warp == 0) {
        if (lane < blockDim.x / 32) s = shared[lane];
        s = warp_reduce(s);
    }
    return s;
}

__device__ __forceinline__ void finish(Stats s, int64_t n, float* out) {
    float inv = 1.0f / (float)n;
    out[0] = s.loss * inv;
    out[1] = s.clips * inv;
    out[2] = s.kl * inv;
    // exp(2*LSE(lw)-LSE(2*lw))/N = sum(r)^2/(N*sum(r*r)).
    // The specified bounded log weights keep both moments in fp32 range.
    out[3] = (s.sum / s.squares) * (s.sum * inv);
    out[4] = s.maximum;
    out[5] = s.sum * inv;
}

__device__ __forceinline__ void add_sample(Stats& s, float c, float o, float a,
                                          float lo, float hi, float rlo, float rhi) {
    float lw = c - o;
    float r = __expf(lw);
    // Exponentiation is monotone, so clamping the ratio avoids a second exp.
    float clipped = fminf(fmaxf(r, rlo), rhi);
    s.loss -= fminf(r * a, clipped * a);
    s.clips += (lw < lo || lw > hi) ? 1.0f : 0.0f;
    s.kl += (r - 1.0f) - lw;
    s.sum += r;
    s.squares += r*r;
    s.maximum = fmaxf(s.maximum, r);
}

template<int ITEMS, bool SINGLE>
__global__ void first(int64_t n, const float* __restrict__ current,
                      const float* __restrict__ old, const float* __restrict__ advantage,
                      float lo, float hi, float rlo, float rhi,
                      Stats* partial, float* output) {
    __shared__ Stats shared[8];
    Stats s = {0, 0, 0, 0, 0, 0};
    int64_t start = (int64_t)blockIdx.x * (blockDim.x * ITEMS) + threadIdx.x;
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        int64_t i = start + k * blockDim.x;
        if (i < n) add_sample(s, current[i], old[i], advantage[i], lo, hi, rlo, rhi);
    }
    s = block_reduce(s, shared);
    if (threadIdx.x == 0) {
        if (SINGLE) finish(s, n, output);
        else partial[blockIdx.x] = s;
    }
}

template<int VECTORS>
__global__ void first_vec(int64_t n, const float* __restrict__ current,
                      const float* __restrict__ old, const float* __restrict__ advantage,
                      float lo, float hi, float rlo, float rhi,
                      Stats* partial, float* output) {
    __shared__ Stats shared[8];
    Stats s = {0, 0, 0, 0, 0, 0};
    int64_t start = (int64_t)blockIdx.x * (blockDim.x * 4 * VECTORS) + 4 * threadIdx.x;
    #pragma unroll
    for (int k = 0; k < VECTORS; ++k) {
        int64_t i = start + k * blockDim.x * 4;
        if (i + 3 < n) {
            float4 c = *reinterpret_cast<const float4*>(current+i);
            float4 o = *reinterpret_cast<const float4*>(old+i);
            float4 a = *reinterpret_cast<const float4*>(advantage+i);
            add_sample(s, c.x, o.x, a.x, lo, hi, rlo, rhi);
            add_sample(s, c.y, o.y, a.y, lo, hi, rlo, rhi);
            add_sample(s, c.z, o.z, a.z, lo, hi, rlo, rhi);
            add_sample(s, c.w, o.w, a.w, lo, hi, rlo, rhi);
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                if (i+j<n) add_sample(s, current[i+j], old[i+j], advantage[i+j], lo, hi, rlo, rhi);
        }
    }
    s = block_reduce(s, shared);
    if (threadIdx.x == 0) partial[blockIdx.x] = s;
}

__global__ void last(const Stats* __restrict__ partial, int count, int64_t n, float* output) {
    __shared__ Stats shared[8];
    Stats s = {0, 0, 0, 0, 0, 0};
    for (int i = threadIdx.x; i < count; i += blockDim.x) s = combine(s, partial[i]);
    s = block_reduce(s, shared);
    if (threadIdx.x == 0) finish(s, n, output);
}
} // namespace ppo_detail

void reference(const int32_t B, const int32_t T, const int32_t H,
               const float* current_log_prob, const float* old_log_prob,
               const float* advantage, const float clip_epsilon, float* output) {
    using namespace ppo_detail;
    const int64_t n = (int64_t)B * T * H;
    const float lo = (float)log1p(-(double)clip_epsilon);
    const float hi = (float)log1p((double)clip_epsilon);
    const float rlo = 1.0f - clip_epsilon;
    const float rhi = 1.0f + clip_epsilon;
    if (n <= 1024) {
        first<4, true><<<1, 256>>>(n, current_log_prob, old_log_prob, advantage,
                                 lo, hi, rlo, rhi, nullptr, output);
        return;
    }
    int blocks = (int)((n + 2047) / 2048);
    static Stats* scratch = nullptr;
    static size_t capacity = 0;
    if ((size_t)blocks > capacity) {
        if (scratch) cudaFree(scratch);
        cudaMalloc(&scratch, (size_t)blocks * sizeof(Stats));
        capacity = blocks;
    }
    const bool aligned = (((uintptr_t)current_log_prob | (uintptr_t)old_log_prob |
                           (uintptr_t)advantage) & 15) == 0;
    if (aligned) {
        first_vec<2><<<blocks, 256>>>(n, current_log_prob, old_log_prob, advantage,
                                     lo, hi, rlo, rhi, scratch, output);
    } else {
        first<8, false><<<blocks, 256>>>(n, current_log_prob, old_log_prob, advantage,
                                       lo, hi, rlo, rhi, scratch, output);
    }
    last<<<1, 256>>>(scratch, blocks, n, output);
}
