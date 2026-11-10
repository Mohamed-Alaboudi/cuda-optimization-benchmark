#include <cuda_runtime.h>
#include <stdint.h>

// A reverse scan of affine maps x -> reward + keep * gamma * x.
// The maps are composed in time order, so a zero keep coefficient also
// cuts off all rewards beyond a terminal flag.
namespace {
constexpr int kThreads = 256;
constexpr int kItems = 4;

__global__ void reward_scan(const float* __restrict__ reward,
                            const bool* __restrict__ done,
                            float gamma, int T,
                            float* __restrict__ output) {
    __shared__ float warp_a[8];
    __shared__ float warp_b[8];
    __shared__ float warp_carry[9];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    size_t row = (size_t)blockIdx.x * T;
    float carry = 0.0f;

    for (int base = 0; base < T; base += kThreads * kItems) {
        float r[kItems], a[kItems];
        float local_b[kItems], local_a[kItems];
        float s = 0.0f, p = 1.0f;
        #pragma unroll
        for (int j = 0; j < kItems; ++j) {
            int t = T - 1 - (base + tid * kItems + j);
            r[j] = t >= 0 ? reward[row + t] : 0.0f;
            a[j] = t >= 0 ? (done[row + t] ? 0.0f : gamma) : 1.0f;
            s = fmaf(a[j], s, r[j]);
            p = a[j] * p;
            local_b[j] = s;
            local_a[j] = p;
        }
        // Inclusive affine scan within each warp.
        #pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            float ps = __shfl_up_sync(0xffffffffu, s, d);
            float pp = __shfl_up_sync(0xffffffffu, p, d);
            if (lane >= d) {
                s = fmaf(p, ps, s);
                p *= pp;
            }
        }
        if (lane == 31) {
            warp_a[warp] = p;
            warp_b[warp] = s;
        }
        // Exclusive prefix for this thread inside its warp.
        float prev_s = __shfl_up_sync(0xffffffffu, s, 1);
        float prev_p = __shfl_up_sync(0xffffffffu, p, 1);
        if (lane == 0) { prev_s = 0.0f; prev_p = 1.0f; }
        __syncthreads();
        if (warp == 0) {
            float ws = lane < 8 ? warp_b[lane] : 0.0f;
            float wp = lane < 8 ? warp_a[lane] : 1.0f;
            #pragma unroll
            for (int d = 1; d < 8; d *= 2) {
                float ps = __shfl_up_sync(0xffffffffu, ws, d);
                float pp = __shfl_up_sync(0xffffffffu, wp, d);
                if (lane >= d) {
                    ws = fmaf(wp, ps, ws);
                    wp *= pp;
                }
            }
            if (lane < 8) warp_carry[lane + 1] = fmaf(wp, carry, ws);
            if (lane == 0) warp_carry[0] = carry;
        }
        __syncthreads();
        float before = fmaf(prev_p, warp_carry[warp], prev_s);
        carry = warp_carry[8];
        #pragma unroll
        for (int j = 0; j < kItems; ++j) {
            int t = T - 1 - (base + tid * kItems + j);
            if (t >= 0) output[row + t] = fmaf(local_a[j], before, local_b[j]);
        }
    }
}
}

void reference(const float* reward, const bool* done, float gamma,
               int32_t B, int32_t T, float* output) {
    if (B <= 0 || T <= 0) return;
    reward_scan<<<B, kThreads>>>(reward, done, gamma, T, output);
}
