#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

namespace {
// All work is launched on the legacy default stream, so consecutive calls
// use the scratch space in stream order. Each block publishes one partial.
constexpr int kBlocks = 256;
__device__ float scratch[6 * kBlocks];
__device__ unsigned completed;

struct Partial {
    float gain;
    unsigned clipped;
    float kl;
    float max_ratio;
    float sum_ratio;
    float sum_ratio2;
};

__device__ __forceinline__ Partial warp_reduce(Partial a) {
    unsigned clipped = __reduce_add_sync(0xffffffff, a.clipped);
    float max_ratio = __uint_as_float(__reduce_max_sync(
        0xffffffff, __float_as_uint(a.max_ratio)));
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        a.gain += __shfl_down_sync(0xffffffff, a.gain, offset);
        a.kl += __shfl_down_sync(0xffffffff, a.kl, offset);
        a.sum_ratio += __shfl_down_sync(0xffffffff, a.sum_ratio, offset);
        a.sum_ratio2 += __shfl_down_sync(0xffffffff, a.sum_ratio2, offset);
    }
    a.clipped = clipped;
    a.max_ratio = max_ratio;
    return a;
}

__device__ __forceinline__ Partial block_reduce(Partial a, Partial *shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    a = warp_reduce(a);
    if (lane == 0) shared[warp] = a;
    __syncthreads();
    if (warp == 0) {
        a = lane < 8 ? shared[lane] : Partial{0, 0, 0, 0, 0, 0};
        a = warp_reduce(a);
    }
    return a;
}

__device__ __forceinline__ void accumulate(
    Partial &a, float cur, float old, float adv, float lo, float hi, float eps) {
    const float lw = cur - old;
    const float r = __expf(lw);
    // Only the high clip can be active for positive advantages, and only
    // the low clip can be active for negative advantages.
    const float gain_ratio = (adv < 0.f && lw < lo) ? 1.f - eps :
                             ((adv > 0.f && lw > hi) ? 1.f + eps : r);
    a.gain += gain_ratio * adv;
    a.clipped += (unsigned)(lw < lo || lw > hi);
    a.kl += (r - 1.f) - lw;
    a.max_ratio = fmaxf(a.max_ratio, r);
    a.sum_ratio += r;
    a.sum_ratio2 += r * r;
}

__device__ __forceinline__ void write_result(Partial a, int N, float *out) {
    float inv = __fdividef(1.f, (float)N);
    out[0] = -a.gain * inv;
    out[1] = a.clipped * inv;
    out[2] = a.kl * inv;
    out[3] = (a.sum_ratio * inv) * __fdividef(a.sum_ratio, a.sum_ratio2);
    out[4] = a.max_ratio;
    out[5] = a.sum_ratio * inv;
}

template <bool Vectorized>
__global__ void ppo(int N, const float *__restrict__ cur,
                    const float *__restrict__ old,
                    const float *__restrict__ adv,
                    float lo, float hi, float eps, float *__restrict__ out) {
    __shared__ Partial shared[8];
    __shared__ bool last;
    __shared__ float final_values[6];
    Partial a{0, 0, 0, 0, 0, 0};
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (Vectorized) {
        auto c4 = reinterpret_cast<const float4 *>(cur);
        auto o4 = reinterpret_cast<const float4 *>(old);
        auto a4 = reinterpret_cast<const float4 *>(adv);
        for (int i = thread; i < N / 4; i += stride) {
            float4 c = c4[i], o = o4[i], v = a4[i];
            accumulate(a, c.x, o.x, v.x, lo, hi, eps);
            accumulate(a, c.y, o.y, v.y, lo, hi, eps);
            accumulate(a, c.z, o.z, v.z, lo, hi, eps);
            accumulate(a, c.w, o.w, v.w, lo, hi, eps);
        }
    } else {
        for (int i = thread; i < N; i += stride) {
            accumulate(a, cur[i], old[i], adv[i], lo, hi, eps);
        }
    }

    a = block_reduce(a, shared);
    if (gridDim.x == 1) {
        if (threadIdx.x == 0) write_result(a, N, out);
        return;
    }

    // The last block reads the partial results. Volatile stores/loads and
    // the device fence make the publication precede the completion counter.
    if (threadIdx.x == 0) {
        volatile float *p = scratch;
        p[blockIdx.x] = a.gain;
        p[kBlocks + blockIdx.x] = (float)a.clipped;
        p[2 * kBlocks + blockIdx.x] = a.kl;
        p[3 * kBlocks + blockIdx.x] = a.max_ratio;
        p[4 * kBlocks + blockIdx.x] = a.sum_ratio;
        p[5 * kBlocks + blockIdx.x] = a.sum_ratio2;
        __threadfence();
        unsigned ticket = atomicInc(&completed, gridDim.x - 1);
        last = ticket == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;

    // Each warp handles one contiguous column of partials.
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (warp < 6) {
        float v = 0;
        volatile float *p = scratch + kBlocks * warp;
        if (gridDim.x == kBlocks) {
            #pragma unroll
            for (int j = 0; j < kBlocks / 32; ++j) {
                float x = p[lane + 32 * j];
                if (warp == 3) v = fmaxf(v, x); else v += x;
            }
        } else {
            for (int i = lane; i < gridDim.x; i += 32) {
                float x = p[i];
                if (warp == 3) v = fmaxf(v, x); else v += x;
            }
        }
        if (warp == 3) {
            v = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(v)));
        } else if (warp == 1) {
            v = (float)__reduce_add_sync(0xffffffff, (unsigned)v);
        } else {
            #pragma unroll
            for (int d = 16; d > 0; d >>= 1)
                v += __shfl_down_sync(0xffffffff, v, d);
        }
        if (lane == 0) final_values[warp] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        Partial b{final_values[0], (unsigned)final_values[1], final_values[2],
                  final_values[3], final_values[4], final_values[5]};
        write_result(b, N, out);
    }
}
} // namespace

void reference(const int32_t B, const int32_t T, const int32_t H,
               const float *current_log_prob, const float *old_log_prob,
               const float *advantage, const float clip_epsilon, float *output) {
    const int N = B * T * H;
    const float lo = log1pf(-clip_epsilon);
    const float hi = log1pf(clip_epsilon);
    int blocks = (N + 1023) / 1024;
    if (blocks > kBlocks) blocks = kBlocks;
    bool vectorized = (N % 4 == 0) &&
        (((uintptr_t)current_log_prob | (uintptr_t)old_log_prob |
          (uintptr_t)advantage) & 15) == 0;
    if (vectorized)
        ppo<true><<<blocks, 256>>>(N, current_log_prob, old_log_prob, advantage,
                                   lo, hi, clip_epsilon, output);
    else
        ppo<false><<<blocks, 256>>>(N, current_log_prob, old_log_prob, advantage,
                                    lo, hi, clip_epsilon, output);
}
