#include <cuda_runtime.h>
#include <math.h>

// Accumulators: [0]=sum_gain, [1]=clip_count, [2]=sum_kl, [3]=sum_w, [4]=sum_w2, [5]=max_ratio
__device__ double g_acc[6];

__global__ void ppo_reduce_kernel(const float* __restrict__ cur,
                                  const float* __restrict__ old,
                                  const float* __restrict__ adv,
                                  long long n, double lo, double hi) {
    double s_gain = 0.0, s_clip = 0.0, s_kl = 0.0, s_w = 0.0, s_w2 = 0.0, s_max = 0.0;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float lwf = cur[i] - old[i]; double lw = (double)lwf;
        double a = (double)adv[i];
        float wf = expf(lwf); double w = (double)wf;
        double lwc = lw < lo ? lo : (lw > hi ? hi : lw);
        double wc = (lwc==lw) ? w : (double)expf((float)lwc);
        double g1 = w * a, g2 = wc * a;
        double gain = g1 < g2 ? g1 : g2;
        s_gain += gain;
        s_clip += (lwc != lw) ? 1.0 : 0.0;
        s_kl += (w - 1.0) - lw;
        s_w += w;
        s_w2 += w * w;
        s_max = w > s_max ? w : s_max;
    }
    // warp reduce
    for (int off = 16; off > 0; off >>= 1) {
        s_gain += __shfl_down_sync(0xffffffffu, s_gain, off);
        s_clip += __shfl_down_sync(0xffffffffu, s_clip, off);
        s_kl   += __shfl_down_sync(0xffffffffu, s_kl, off);
        s_w    += __shfl_down_sync(0xffffffffu, s_w, off);
        s_w2   += __shfl_down_sync(0xffffffffu, s_w2, off);
        double o = __shfl_down_sync(0xffffffffu, s_max, off);
        s_max = o > s_max ? o : s_max;
    }
    __shared__ double sm[6][32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) {
        sm[0][warp] = s_gain; sm[1][warp] = s_clip; sm[2][warp] = s_kl;
        sm[3][warp] = s_w;    sm[4][warp] = s_w2;   sm[5][warp] = s_max;
    }
    __syncthreads();
    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        s_gain = lane < nw ? sm[0][lane] : 0.0;
        s_clip = lane < nw ? sm[1][lane] : 0.0;
        s_kl   = lane < nw ? sm[2][lane] : 0.0;
        s_w    = lane < nw ? sm[3][lane] : 0.0;
        s_w2   = lane < nw ? sm[4][lane] : 0.0;
        s_max  = lane < nw ? sm[5][lane] : 0.0;
        for (int off = 16; off > 0; off >>= 1) {
            s_gain += __shfl_down_sync(0xffffffffu, s_gain, off);
            s_clip += __shfl_down_sync(0xffffffffu, s_clip, off);
            s_kl   += __shfl_down_sync(0xffffffffu, s_kl, off);
            s_w    += __shfl_down_sync(0xffffffffu, s_w, off);
            s_w2   += __shfl_down_sync(0xffffffffu, s_w2, off);
            double o = __shfl_down_sync(0xffffffffu, s_max, off);
            s_max = o > s_max ? o : s_max;
        }
        if (lane == 0) {
            atomicAdd(&g_acc[0], s_gain);
            atomicAdd(&g_acc[1], s_clip);
            atomicAdd(&g_acc[2], s_kl);
            atomicAdd(&g_acc[3], s_w);
            atomicAdd(&g_acc[4], s_w2);
            atomicMax(reinterpret_cast<unsigned long long*>(&g_acc[5]),
                      __double_as_longlong(s_max) > 0 ? __double_as_longlong(s_max) : 0); // s_max >= 0 always
        }
    }
}

__global__ void ppo_finalize_kernel(float* __restrict__ out, double n) {
    double s_gain = g_acc[0], s_clip = g_acc[1], s_kl = g_acc[2];
    double s_w = g_acc[3], s_w2 = g_acc[4], s_max = g_acc[5];
    g_acc[0] = 0.0; g_acc[1] = 0.0; g_acc[2] = 0.0; g_acc[3] = 0.0; g_acc[4] = 0.0; g_acc[5] = 0.0;
    out[0] = (float)(-s_gain / n);
    out[1] = (float)(s_clip / n);
    out[2] = (float)(s_kl / n);
    out[3] = (float)((s_w * s_w) / (s_w2 * n));
    out[4] = (float)s_max;
    out[5] = (float)(s_w / n);
}

void reference(const int32_t B, const int32_t T, const int32_t H, const float* current_log_prob,
               const float* old_log_prob, const float* advantage, const float clip_epsilon, float* output) {
    long long n = (long long)B * (long long)T * (long long)H;
    double eps = (double)clip_epsilon;
    double lo = log1p(-eps), hi = log1p(eps);
    int threads = 256;
    long long want = (n + threads - 1) / threads;
    int blocks = (int)(want < 1024 ? want : 1024);
    if (blocks < 1) blocks = 1;
    ppo_reduce_kernel<<<blocks, threads>>>(current_log_prob, old_log_prob, advantage, n, lo, hi);
    ppo_finalize_kernel<<<1, 1>>>(output, (double)n);
}
