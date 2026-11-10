#include <cstdint>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

#define NT 256

struct Aff { double a, b; };

struct ComposeOp {
    // acc: composite of chunks to the right (earlier in scan order); x: chunk to the left
    __device__ __forceinline__ Aff operator()(const Aff& acc, const Aff& x) const {
        Aff r; r.a = x.a * acc.a; r.b = fma(x.a, acc.b, x.b); return r;
    }
};

__global__ void __launch_bounds__(NT) r2g_kernel(const float* __restrict__ reward,
                                                 const bool* __restrict__ done,
                                                 float gamma, int T, float* __restrict__ out) {
    typedef cub::BlockScan<Aff, NT> Scan;
    __shared__ typename Scan::TempStorage tmp;
    const long long row = blockIdx.x;
    const float* r = reward + row * (long long)T;
    const bool* d = done + row * (long long)T;
    float* o = out + row * (long long)T;
    const double g = (double)gamma;

    int chunk = (T + NT - 1) / NT;
    int c = NT - 1 - threadIdx.x;  // chunk index: thread 0 handles rightmost chunk
    int start = c * chunk;
    int end = min(start + chunk, T);

    Aff loc; loc.a = 1.0; loc.b = 0.0;
    for (int t = end - 1; t >= start; --t) {
        double gt = (d[t] || t == T - 1) ? 0.0 : g;
        loc.b = fma(gt, loc.b, (double)r[t]);
        loc.a = gt * loc.a;
    }
    Aff init; init.a = 1.0; init.b = 0.0;
    Aff pre;
    Scan(tmp).ExclusiveScan(loc, pre, init, ComposeOp());
    double y = pre.b;  // value of output just after this chunk (0 beyond the row)
    for (int t = end - 1; t >= start; --t) {
        double gt = (d[t] || t == T - 1) ? 0.0 : g;
        y = fma(gt, y, (double)r[t]);
        o[t] = (float)y;
    }
}

void reference(const float* reward, const bool* done, float gamma, int32_t B, int32_t T, float* output) {
    if (B <= 0 || T <= 0) return;
    r2g_kernel<<<B, NT>>>(reward, done, gamma, T, output);
}
