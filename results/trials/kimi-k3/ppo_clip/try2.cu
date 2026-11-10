#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// Global accumulators (zeroed per call via cudaMemsetAsync).
struct Acc {
    double v[6];                 // 0 gain, 1 clip count, 2 kl, 3 ratio sum, 4 exp(lw) sum, 5 exp(2lw) sum
    unsigned long long maxbits;  // max ratio (positive double bits are order-isomorphic)
};
__device__ Acc g_acc;

__global__ void ppo_reduce(const float* __restrict__ cur,
                           const float* __restrict__ old,
                           const float* __restrict__ adv,
                           long long n, double lo, double hi) {
    float gain = 0.f, clip = 0.f, kl = 0.f, rsum = 0.f, esum = 0.f, e2sum = 0.f, rmax = 0.f;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        double lw = (double)cur[i] - (double)old[i];   // exact, matches fp64 reference
        float a = adv[i];
        float r = expf((float)lw);
        bool clipped = (lw < lo) || (lw > hi);
        float rc;
        if (clipped) {
            double lwc = lw < lo ? lo : hi;
            rc = expf((float)lwc);
        } else {
            rc = r;
        }
        float g1 = r * a, g2 = rc * a;
        gain += g1 < g2 ? g1 : g2;
        clip += clipped ? 1.f : 0.f;
        kl += (float)(((double)r - 1.0) - lw);
        rsum += r;
        rmax = fmaxf(rmax, r);
        esum += r;
        e2sum += r * r;
    }

    // warp reduction
    for (int off = 16; off > 0; off >>= 1) {
        gain += __shfl_down_sync(0xffffffffu, gain, off);
        clip += __shfl_down_sync(0xffffffffu, clip, off);
        kl   += __shfl_down_sync(0xffffffffu, kl, off);
        rsum += __shfl_down_sync(0xffffffffu, rsum, off);
        esum += __shfl_down_sync(0xffffffffu, esum, off);
        e2sum += __shfl_down_sync(0xffffffffu, e2sum, off);
        rmax = fmaxf(rmax, __shfl_down_sync(0xffffffffu, rmax, off));
    }

    __shared__ double sh[6][32];
    __shared__ float shmax[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sh[0][warp] = gain; sh[1][warp] = clip; sh[2][warp] = kl;
        sh[3][warp] = rsum; sh[4][warp] = esum; sh[5][warp] = e2sum;
        shmax[warp] = rmax;
    }
    __syncthreads();
    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        double v0 = lane < nw ? sh[0][lane] : 0.0;
        double v1 = lane < nw ? sh[1][lane] : 0.0;
        double v2 = lane < nw ? sh[2][lane] : 0.0;
        double v3 = lane < nw ? sh[3][lane] : 0.0;
        double v4 = lane < nw ? sh[4][lane] : 0.0;
        double v5 = lane < nw ? sh[5][lane] : 0.0;
        float vm = lane < nw ? shmax[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            v0 += __shfl_down_sync(0xffffffffu, v0, off);
            v1 += __shfl_down_sync(0xffffffffu, v1, off);
            v2 += __shfl_down_sync(0xffffffffu, v2, off);
            v3 += __shfl_down_sync(0xffffffffu, v3, off);
            v4 += __shfl_down_sync(0xffffffffu, v4, off);
            v5 += __shfl_down_sync(0xffffffffu, v5, off);
            vm = fmaxf(vm, __shfl_down_sync(0xffffffffu, vm, off));
        }
        if (lane == 0) {
            atomicAdd(&g_acc.v[0], v0);
            atomicAdd(&g_acc.v[1], v1);
            atomicAdd(&g_acc.v[2], v2);
            atomicAdd(&g_acc.v[3], v3);
            atomicAdd(&g_acc.v[4], v4);
            atomicAdd(&g_acc.v[5], v5);
            atomicMax(&g_acc.maxbits, (unsigned long long)__double_as_longlong((double)vm));
        }
    }
}

__global__ void ppo_final(double n, float* __restrict__ out) {
    if (threadIdx.x != 0) return;
    double sg = g_acc.v[0], sc = g_acc.v[1], skl = g_acc.v[2];
    double sr = g_acc.v[3], se = g_acc.v[4], se2 = g_acc.v[5];
    out[0] = (float)(-sg / n);
    out[1] = (float)(sc / n);
    out[2] = (float)(skl / n);
    out[3] = (float)(exp(2.0 * log(se) - log(se2)) / n);
    out[4] = (float)__longlong_as_double((long long)g_acc.maxbits);
    out[5] = (float)(sr / n);
}

void reference(const int32_t B, const int32_t T, const int32_t H,
               const float* current_log_prob, const float* old_log_prob,
               const float* advantage, const float clip_epsilon, float* output) {
    long long n = (long long)B * (long long)T * (long long)H;
    double eps = (double)clip_epsilon;
    double lo = log1p(-eps), hi = log1p(eps);

    static Acc* d_acc = nullptr;
    if (!d_acc) cudaGetSymbolAddress((void**)&d_acc, g_acc);
    cudaMemsetAsync(d_acc, 0, sizeof(Acc), 0);

    int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    if (blocks < 1) blocks = 1;
    ppo_reduce<<<(int)blocks, threads>>>(current_log_prob, old_log_prob, advantage, n, lo, hi);
    ppo_final<<<1, 32>>>((double)n, output);
}
