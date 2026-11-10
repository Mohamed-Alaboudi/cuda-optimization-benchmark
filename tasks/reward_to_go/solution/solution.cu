#include <cstdint>

__global__ void reward_to_go_kernel(
    const float* __restrict__ reward,
    const bool* __restrict__ done,
    float gamma,
    int32_t B, int32_t T,
    float* __restrict__ output)
{
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ char smem[];
    float* s_reward = reinterpret_cast<float*>(smem);
    bool* s_done = reinterpret_cast<bool*>(smem + T * sizeof(float));
    size_t chunk_off = ((size_t)(T * sizeof(float) + T * sizeof(bool)) + 7) & ~(size_t)7;
    double* s_chunk_a = reinterpret_cast<double*>(smem + chunk_off);
    double* s_chunk_b = s_chunk_a + blockDim.x;

    int C = blockDim.x;
    int cs = (T + C - 1) / C;
    int cstart = threadIdx.x * cs;
    int cend = cstart + cs < T ? cstart + cs : T;
    int nactive = (T + cs - 1) / cs;
    double g = static_cast<double>(gamma);

    // load rewards and done flags into shared memory
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        s_reward[t] = reward[b * T + t];
        s_done[t] = done[b * T + t];
    }
    s_chunk_a[threadIdx.x] = 1.0;
    s_chunk_b[threadIdx.x] = 0.0;
    __syncthreads();

    // serial backward composition within each chunk
    // Each position t: f_t(x) = a_t*x + b_t, a_t = done?0:gamma, b_t = reward[t]
    // compose((a1,b1),(a2,b2)) = (a1*a2, a1*b2+b1) is associative
    if (cstart < T) {
        double aa = 1.0, bb = 0.0;
        for (int t = cend - 1; t >= cstart; t--) {
            double at = s_done[t] ? 0.0 : g;
            double bt = static_cast<double>(s_reward[t]);
            aa = at * aa;
            bb = at * bb + bt;
        }
        s_chunk_a[threadIdx.x] = aa;
        s_chunk_b[threadIdx.x] = bb;
    }
    __syncthreads();

    // backward Kogge-Stone inclusive scan across chunk results (a, b)
    for (int k = 0; (1 << k) < C; k++) {
        int stride = 1 << k;
        double my_a = s_chunk_a[threadIdx.x];
        double my_b = s_chunk_b[threadIdx.x];
        double ra = 1.0, rb = 0.0;
        if (threadIdx.x + stride < C) {
            ra = s_chunk_a[threadIdx.x + stride];
            rb = s_chunk_b[threadIdx.x + stride];
        }
        __syncthreads();
        if (threadIdx.x + stride < C) {
            s_chunk_a[threadIdx.x] = my_a * ra;
            s_chunk_b[threadIdx.x] = my_a * rb + my_b;
        }
        __syncthreads();
    }

    // exclusive carry for each chunk
    double carry = 0.0;
    if (threadIdx.x + 1 < nactive) {
        carry = s_chunk_b[threadIdx.x + 1];
    }

    // second serial pass per chunk, applying the carry
    if (cstart < T) {
        for (int t = cend - 1; t >= cstart; t--) {
            if (s_done[t]) carry = 0.0;
            double val = static_cast<double>(s_reward[t]) + g * carry;
            s_reward[t] = static_cast<float>(val);
            carry = val;
        }
    }
    __syncthreads();

    // store results
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        output[b * T + t] = s_reward[t];
    }
}



// Inclusive suffix scan of 32 affine chunks.  'mask' marks the chunks that
// contain a done flag, and g, g2, ... are successive powers of the chunk
// discount.  A chunk's first terminal cuts off all following contributions.
__device__ __forceinline__ float scan32(float v, unsigned mask, int lane,
                                       float g, float g2, float g4,
                                       float g8, float g16) {
    unsigned m = mask >> lane;
    int dist = m ? __ffs(m) - 1 : 31 - lane;
    float x;
    x = __shfl_down_sync(0xffffffff, v, 1);  if (dist >= 1)  v = fmaf(g, x, v);
    x = __shfl_down_sync(0xffffffff, v, 2);  if (dist >= 2)  v = fmaf(g2, x, v);
    x = __shfl_down_sync(0xffffffff, v, 4);  if (dist >= 4)  v = fmaf(g4, x, v);
    x = __shfl_down_sync(0xffffffff, v, 8);  if (dist >= 8)  v = fmaf(g8, x, v);
    x = __shfl_down_sync(0xffffffff, v, 16); if (dist >= 16) v = fmaf(g16, x, v);
    return v;
}

// Scalar-load fallback for the 1024-long rows, including unaligned inputs.
// Each warp processes two sets of 32 consecutive timesteps.
__global__ void timed_kernel(const float* __restrict__ r,
                             const bool* __restrict__ d, float g,
                             int B, int T, float* __restrict__ o) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.y;
    int t0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 64;
    int idx = row * 1024 + t0 + lane;
    float g2 = g*g, g4 = g2*g2, g8 = g4*g4, g16 = g8*g8, g32 = g16*g16;
    float v0 = r[idx];
    float v1 = r[idx + 32];
    unsigned m0 = __ballot_sync(0xffffffff, d[idx]);
    unsigned m1 = __ballot_sync(0xffffffff, d[idx + 32]);
    v0 = scan32(v0, m0, lane, g, g2, g4, g8, g16);
    v1 = scan32(v1, m1, lane, g, g2, g4, g8, g16);

    float c = 0.f;
    if (!(m1 & 0x80000000u) && t0 + 64 < 1024) {
        float factor = 1.f;
        for (int p = t0 + 64; p < 1024; p += 32) {
            float q = r[row * 1024 + p + lane];
            unsigned m = __ballot_sync(0xffffffff, d[row * 1024 + p + lane]);
            q = scan32(q, m, lane, g, g2, g4, g8, g16);
            float x = __shfl_sync(0xffffffff, q, 0);
            c = fmaf(factor, x, c);
            if (m) break;
            factor *= g32;
        }
    }
    float power = __powf(g, float(32 - lane));
    if ((m1 >> lane) == 0) v1 = fmaf(power, c, v1);
    c = __shfl_sync(0xffffffff, v1, 0);
    if ((m0 >> lane) == 0) v0 = fmaf(power, c, v0);
    o[idx] = v0;
    o[idx + 32] = v1;
}

// Summary of four adjacent timesteps, with zero carry after the fourth.
__device__ __forceinline__ float local4(int idx, const float* __restrict__ r,
                                        const bool* __restrict__ d,
                                        float g, bool& term) {
    float4 rr = *reinterpret_cast<const float4*>(r + idx);
    unsigned flags = *reinterpret_cast<const unsigned*>(d + idx);
    bool d0 = (flags & 255) != 0;
    bool d1 = ((flags >> 8) & 255) != 0;
    bool d2 = ((flags >> 16) & 255) != 0;
    term = flags != 0;
    float v = rr.w;
    v = fmaf(d2 ? 0.f : g, v, rr.z);
    v = fmaf(d1 ? 0.f : g, v, rr.y);
    v = fmaf(d0 ? 0.f : g, v, rr.x);
    return v;
}

// A warp handles 128 outputs: each lane owns four adjacent values. The warp
// scans their summaries and, if needed, looks ahead through subsequent chunks
// until a terminal or the row end. This avoids inter-block synchronization.
template <bool Gamma99, bool VecOutput>
__global__ void inter4_kernel(const float* __restrict__ r,
                              const bool* __restrict__ d, float gamma,
                              int B, int T, float* __restrict__ o) {
    const float g = Gamma99 ? 0.99f : gamma;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x;
    int t0 = (threadIdx.x >> 5) * 128;
    int idx = row * 1024 + t0 + 4 * lane;
    float g2 = g*g, g4 = g2*g2, g8 = g4*g4, g16 = g8*g8;
    float g32 = g16*g16, g64 = g32*g32, g128 = g64*g64;

    float4 rr = *reinterpret_cast<const float4*>(r + idx);
    unsigned flags = *reinterpret_cast<const unsigned*>(d + idx);
    bool d0 = (flags & 255) != 0;
    bool d1 = ((flags >> 8) & 255) != 0;
    bool d2 = ((flags >> 16) & 255) != 0;
    bool d3 = ((flags >> 24) & 255) != 0;
    unsigned mask = __ballot_sync(0xffffffff, flags != 0);
    float v = rr.w;
    v = fmaf(d2 ? 0.f : g, v, rr.z);
    v = fmaf(d1 ? 0.f : g, v, rr.y);
    v = fmaf(d0 ? 0.f : g, v, rr.x);
    v = scan32(v, mask, lane, g4, g8, g16, g32, g64);

    // The last output in the warp needs a carry only if its done flag is off.
    float c = 0.f;
    unsigned lastflags = __ballot_sync(0xffffffff, d3);
    if (!(lastflags & 0x80000000u) && t0 + 128 < 1024) {
        float factor = 1.f;
        for (int p = t0 + 128; p < 1024; p += 128) {
            bool term;
            float q = local4(row * 1024 + p + 4 * lane, r, d, g, term);
            unsigned m = __ballot_sync(0xffffffff, term);
            q = scan32(q, m, lane, g4, g8, g16, g32, g64);
            float x = __shfl_sync(0xffffffff, q, 0);
            c = fmaf(factor, x, c);
            if (m) break;
            factor *= g128;
        }
    }
    if ((mask >> lane) == 0) {
        float power;
        if (Gamma99) {
            // log2 of the float32 value 0.99f.
            float exponent = -0.0144995557975271944f * float(4 * (32 - lane));
            asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(power) : "f"(exponent));
        } else {
            power = __powf(g, float(4 * (32 - lane)));
        }
        v = fmaf(power, c, v);
    }
    float next = __shfl_down_sync(0xffffffff, v, 1);
    if (lane == 31) next = c;
    float out3 = fmaf(d3 ? 0.f : g, next, rr.w);
    float out2 = fmaf(d2 ? 0.f : g, out3, rr.z);
    float out1 = fmaf(d1 ? 0.f : g, out2, rr.y);
    if (VecOutput) {
        *reinterpret_cast<float4*>(o + idx) = make_float4(v, out1, out2, out3);
    } else {
        o[idx] = v;
        o[idx + 1] = out1;
        o[idx + 2] = out2;
        o[idx + 3] = out3;
    }
}

void reference(const float* reward, const bool* done, float gamma,
               int32_t B, int32_t T, float* output) {
    if (B <= 0 || T <= 0) return;
    if (B == 64 && T == 1024) {
        if (((uintptr_t)reward & 15) == 0 && ((uintptr_t)done & 3) == 0) {
            dim3 grid(B);
            if (gamma == 0.99f) {
                if (((uintptr_t)output & 15) == 0)
                    inter4_kernel<true, true><<<grid, 256>>>(reward, done, gamma, B, T, output);
                else
                    inter4_kernel<true, false><<<grid, 256>>>(reward, done, gamma, B, T, output);
            } else {
                if (((uintptr_t)output & 15) == 0)
                    inter4_kernel<false, true><<<grid, 256>>>(reward, done, gamma, B, T, output);
                else
                    inter4_kernel<false, false><<<grid, 256>>>(reward, done, gamma, B, T, output);
            }
        } else {
            dim3 grid(2, B);
            timed_kernel<<<grid, 256>>>(reward, done, gamma, B, T, output);
        }
    } else {
        constexpr int blockSize = 256;
        size_t chunk_off = ((size_t)(T * sizeof(float) + T * sizeof(bool)) + 7) & ~(size_t)7;
        size_t sharedMemBytes = chunk_off + 2 * blockSize * sizeof(double);
        reward_to_go_kernel<<<B, blockSize, sharedMemBytes>>>(reward, done, gamma, B, T, output);
    }
}
