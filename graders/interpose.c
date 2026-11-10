// Call counter, preloaded (LD_PRELOAD) into the driver process.
// It counts CUDA runtime calls that the task header forbids inside the call:
// allocation, host copies, device-wide sync, and work put on any stream other than s
// (kernel launches, async copies and memsets, graph launches, and the synchronous calls
// that use the legacy default stream). Counting is armed by the driver only around the
// answer's call; the driver also tells it which stream is s.
// Built without linking cudart; every call is forwarded to the real cudart with RTLD_NEXT.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <string.h>
#include <cuda_runtime_api.h>

enum { C_ALLOC, C_HOST_COPY, C_DEVICE_SYNC, C_OTHER_STREAM, C_STREAM_SYNC, C_LAUNCH, C_N };
static long g_count[C_N];
static int g_armed;
static cudaStream_t g_s;

void grader_counter_arm(int on) { g_armed = on; }
void grader_counter_set_stream(cudaStream_t s) { g_s = s; }
void grader_counter_reset(void) { memset(g_count, 0, sizeof g_count); }
void grader_counter_read(long* out) { memcpy(out, g_count, sizeof g_count); }
int grader_counter_n(void) { return C_N; }

static void hit(int c) { if (g_armed) g_count[c]++; }

#define REAL(name) static __typeof__(&name) real_##name; \
  if (!real_##name) real_##name = (__typeof__(&name))dlsym(RTLD_NEXT, #name)

static int is_host_kind(enum cudaMemcpyKind k, void* dst, const void* src) {
  if (k == cudaMemcpyDeviceToDevice) return 0;
  if (k != cudaMemcpyDefault) return 1;
  REAL(cudaPointerGetAttributes);
  struct cudaPointerAttributes a;
  const void* p[2] = {dst, src};
  for (int i = 0; i < 2; i++) {
    if (real_cudaPointerGetAttributes(&a, p[i]) != cudaSuccess) return 1;
    if (a.type != cudaMemoryTypeDevice && a.type != cudaMemoryTypeManaged) return 1;
  }
  return 0;
}
static int is_legacy(cudaStream_t s) { return s == 0 || s == cudaStreamLegacy; }
static int off_s(cudaStream_t s) { return s != g_s; }

// allocation and free
cudaError_t cudaMalloc(void** p, size_t n) { REAL(cudaMalloc); hit(C_ALLOC); return real_cudaMalloc(p, n); }
cudaError_t cudaMallocManaged(void** p, size_t n, unsigned f) { REAL(cudaMallocManaged); hit(C_ALLOC); return real_cudaMallocManaged(p, n, f); }
cudaError_t cudaMallocHost(void** p, size_t n) { REAL(cudaMallocHost); hit(C_ALLOC); return real_cudaMallocHost(p, n); }
cudaError_t cudaHostAlloc(void** p, size_t n, unsigned f) { REAL(cudaHostAlloc); hit(C_ALLOC); return real_cudaHostAlloc(p, n, f); }
cudaError_t cudaMallocPitch(void** p, size_t* pitch, size_t w, size_t h) { REAL(cudaMallocPitch); hit(C_ALLOC); return real_cudaMallocPitch(p, pitch, w, h); }
cudaError_t cudaMallocAsync(void** p, size_t n, cudaStream_t s) { REAL(cudaMallocAsync); hit(C_ALLOC); return real_cudaMallocAsync(p, n, s); }
cudaError_t cudaMallocFromPoolAsync(void** p, size_t n, cudaMemPool_t m, cudaStream_t s) { REAL(cudaMallocFromPoolAsync); hit(C_ALLOC); return real_cudaMallocFromPoolAsync(p, n, m, s); }
cudaError_t cudaFree(void* p) { REAL(cudaFree); hit(C_ALLOC); return real_cudaFree(p); }
cudaError_t cudaFreeHost(void* p) { REAL(cudaFreeHost); hit(C_ALLOC); return real_cudaFreeHost(p); }
cudaError_t cudaFreeAsync(void* p, cudaStream_t s) { REAL(cudaFreeAsync); hit(C_ALLOC); return real_cudaFreeAsync(p, s); }

// copies
cudaError_t cudaMemcpy(void* d, const void* s, size_t n, enum cudaMemcpyKind k) {
  REAL(cudaMemcpy); hit(is_host_kind(k, d, s) ? C_HOST_COPY : C_OTHER_STREAM); return real_cudaMemcpy(d, s, n, k);
}
cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, enum cudaMemcpyKind k, cudaStream_t st) {
  REAL(cudaMemcpyAsync);
  if (is_host_kind(k, d, s)) hit(C_HOST_COPY); else if (off_s(st)) hit(C_OTHER_STREAM);
  return real_cudaMemcpyAsync(d, s, n, k, st);
}
cudaError_t cudaMemcpy2D(void* d, size_t dp, const void* s, size_t sp, size_t w, size_t h, enum cudaMemcpyKind k) {
  REAL(cudaMemcpy2D); hit(is_host_kind(k, d, s) ? C_HOST_COPY : C_OTHER_STREAM); return real_cudaMemcpy2D(d, dp, s, sp, w, h, k);
}
cudaError_t cudaMemcpy2DAsync(void* d, size_t dp, const void* s, size_t sp, size_t w, size_t h, enum cudaMemcpyKind k, cudaStream_t st) {
  REAL(cudaMemcpy2DAsync);
  if (is_host_kind(k, d, s)) hit(C_HOST_COPY); else if (off_s(st)) hit(C_OTHER_STREAM);
  return real_cudaMemcpy2DAsync(d, dp, s, sp, w, h, k, st);
}
cudaError_t cudaMemcpyToSymbol(const void* sym, const void* src, size_t n, size_t off, enum cudaMemcpyKind k) {
  REAL(cudaMemcpyToSymbol); hit(k == cudaMemcpyDeviceToDevice ? C_OTHER_STREAM : C_HOST_COPY); return real_cudaMemcpyToSymbol(sym, src, n, off, k);
}
cudaError_t cudaMemcpyToSymbolAsync(const void* sym, const void* src, size_t n, size_t off, enum cudaMemcpyKind k, cudaStream_t st) {
  REAL(cudaMemcpyToSymbolAsync);
  if (k != cudaMemcpyDeviceToDevice) hit(C_HOST_COPY); else if (off_s(st)) hit(C_OTHER_STREAM);
  return real_cudaMemcpyToSymbolAsync(sym, src, n, off, k, st);
}
cudaError_t cudaMemcpyFromSymbol(void* dst, const void* sym, size_t n, size_t off, enum cudaMemcpyKind k) {
  REAL(cudaMemcpyFromSymbol); hit(k == cudaMemcpyDeviceToDevice ? C_OTHER_STREAM : C_HOST_COPY); return real_cudaMemcpyFromSymbol(dst, sym, n, off, k);
}
cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const void* sym, size_t n, size_t off, enum cudaMemcpyKind k, cudaStream_t st) {
  REAL(cudaMemcpyFromSymbolAsync);
  if (k != cudaMemcpyDeviceToDevice) hit(C_HOST_COPY); else if (off_s(st)) hit(C_OTHER_STREAM);
  return real_cudaMemcpyFromSymbolAsync(dst, sym, n, off, k, st);
}

// memset on the legacy stream
cudaError_t cudaMemset(void* p, int v, size_t n) { REAL(cudaMemset); hit(C_OTHER_STREAM); return real_cudaMemset(p, v, n); }
cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t s) {
  REAL(cudaMemsetAsync); if (off_s(s)) hit(C_OTHER_STREAM); return real_cudaMemsetAsync(p, v, n, s);
}

// syncs
cudaError_t cudaDeviceSynchronize(void) { REAL(cudaDeviceSynchronize); hit(C_DEVICE_SYNC); return real_cudaDeviceSynchronize(); }
cudaError_t cudaStreamSynchronize(cudaStream_t s) {
  REAL(cudaStreamSynchronize); hit(is_legacy(s) ? C_DEVICE_SYNC : C_STREAM_SYNC); return real_cudaStreamSynchronize(s);
}

// kernel and graph launches: any stream other than s is a contract fail
cudaError_t cudaLaunchKernel(const void* f, dim3 g, dim3 b, void** a, size_t sm, cudaStream_t st) {
  REAL(cudaLaunchKernel); hit(C_LAUNCH); if (off_s(st)) hit(C_OTHER_STREAM); return real_cudaLaunchKernel(f, g, b, a, sm, st);
}
cudaError_t cudaLaunchCooperativeKernel(const void* f, dim3 g, dim3 b, void** a, size_t sm, cudaStream_t st) {
  REAL(cudaLaunchCooperativeKernel); hit(C_LAUNCH); if (off_s(st)) hit(C_OTHER_STREAM); return real_cudaLaunchCooperativeKernel(f, g, b, a, sm, st);
}
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* c, const void* f, void** a) {
  REAL(cudaLaunchKernelExC); hit(C_LAUNCH); if (off_s(c->stream)) hit(C_OTHER_STREAM); return real_cudaLaunchKernelExC(c, f, a);
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t g, cudaStream_t st) {
  REAL(cudaGraphLaunch); hit(C_LAUNCH); if (off_s(st)) hit(C_OTHER_STREAM); return real_cudaGraphLaunch(g, st);
}
