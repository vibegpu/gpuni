# gpuni

A small AI-friendly CUDA-truth kernel dialect for cross-platform GPU compute (CUDA, HIP, OpenCL C 1.2).

**For AI coding (Codex/Claude Code):** use the `gpuni` skill from https://github.com/vibegpu/gpuni-skills (prompt: use `$gpuni`).

**Package:** `gpuni.h` + `render.c`

## Kernel (Device Code)

Write `*.gu.cu`:

```cpp
#include "gpuni.h"

extern "C" __global__ void saxpy(int n,
                               __global float* __restrict__ y,        // __restrict__: no-alias hint
                               __global const float* __restrict__ x,
                               float a,
                               __local float* smem) {  // dynamic smem: MUST be last param
  bindSharedMem(smem);                // bind to CUDA extern __shared__

  int tid = threadIdx.x;
  int i = (int)(blockIdx.x * blockDim.x + tid);

  smem[tid] = (i < n) ? x[i] : 0.0f;  // load to shared memory
  __syncthreads();

  if (i < n) y[i] = a * smem[tid] + y[i];
}
```

### Dialect Rules

**Required:**
- Entry: `extern "C" __global__ void <name>(...)` (prevents C++ name mangling)
- Annotate pointers: `__global` / `__local` / `__constant` (including aliases)
  - Note: `__global__` (kernel modifier) ≠ `__global` (pointer address space)
  - Aliases must retain address space: `__global float* p = x;`, `__local float* tile = smem + offset;`
- Includes: dialect kernels should only `#include "gpuni.h"` (and optionally `#include "gpuni/..."`), no `<...>` system includes.

**Avoid:** templates, classes, `__shfl*`, `__ballot*`, `float3` in buffers, divergent `__syncthreads()`

### API Reference

| Category | API |
|----------|-----|
| Types | `int`, `uint`, `int64`, `uint64`, `float`, `double` |
| Indexing | `threadIdx.x/y/z`, `blockIdx.x/y/z`, `blockDim.x/y/z`, `gridDim.x/y/z` — all dims available |
| Atomics (int) | `atomicAdd`, `atomicSub`, `atomicExch`, `atomicMin`, `atomicMax`, `atomicCAS`, `atomicAnd`, `atomicOr`, `atomicXor` |
| Atomics (float) | `atomicAddFloat(p,v)`, `atomicMinFloat(p,v)`, `atomicMaxFloat(p,v)` — `p` is `__global float*`. No `atomicAddDouble`, use Q32.32 |
| Accumulator (Q32.32) | Kernel: `atomicAddFixed(__global int64* acc, double v)`, `doubleToFixed(v)`, `fixedToDouble(x)`. Host: `DoubleToFixed(v)`, `FixedToDouble(x)`. Range ±2^31, ~9 decimal digits. |
| Dynamic smem | **Optional.** Kernel: `__local T* smem` as last param + `bindSharedMem(smem)`. Host: `Launch(k, grid, block, smem_bytes, args...)`. Multi-array: `__local float* arr2 = smem + size1;` |
| Restrict | `__restrict__` (pointer no-alias hint) |
| Math | CUDA-style `sinf`, `cosf`, `rsqrtf`, `fminf`, `fmaxf`, `fmaf`, etc. work directly |

## Host

```cpp
#include "gpuni.h"
#include "saxpy.gu.h"  // OpenCL JIT needs this; CUDA/HIP auto
using namespace gu;   // recommended for unqualified API access

int main() {
  int n = 1024; float a = 2.0f;
  int block = 256;
  int grid = (n + block - 1) / block;
  size_t smem = block * sizeof(float);  // dynamic shared memory size

  SetDevice(0);  // must call before Malloc/GetKernel

  float* d_x = Malloc<float>(n);
  float* d_y = Malloc<float>(n);
  float* h_x = MallocHost<float>(n);  // pinned memory
  float* h_y = MallocHost<float>(n);

  for (int i = 0; i < n; i++) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

  Memcpy(d_x, h_x, n * sizeof(float), H2D);
  Memcpy(d_y, h_y, n * sizeof(float), H2D);

  auto k = GetKernel(saxpy);  // auto-cached; repeated calls return same kernel
  Launch(k, grid, block, smem, n, d_y, d_x, a);  // smem before kernel args

  DeviceSync();
  Memcpy(h_y, d_y, n * sizeof(float), D2H);

  Free(d_x); Free(d_y);
  FreeHost(h_x); FreeHost(h_y);
}
```

### Host API Reference

| Category | API |
|----------|-----|
| Device | `SetDevice(id)`, `GetDevice()`, `GetDeviceCount()`, `DeviceSync()` |
| Memory | `Malloc<T>(n)`, `Free(p)`, `Memset(p,v,bytes)`, `MallocHost<T>(n)`, `FreeHost(p)` |
| Copy | `Memcpy(dst,src,bytes,kind)`, `MemcpyAsync(...,stream)` |
| Kernel | `GetKernel(fn)`, `Launch(k, grid, block, [smem,] [stream,] args...)` — order: smem before stream |
| Stream | `stream s; s.sync();` or `StreamSynchronize(s)` |
| Event | `event e; e.record(s); e.sync();` or `EventRecord(e,s); EventSynchronize(e)` |
| Timing | `ElapsedTime(e1, e2)` |
| Error | `Error_t`, `Success`, `GetLastError()`, `GetErrorString(e)`, `Check(expr)` |
| Dim3 | `dim3(x,y,z)` — grid/block can be `int` or `dim3` |

**MemcpyKind:** `H2D`, `D2H`, `D2D`, `H2H` (aliases: `MemcpyHostToDevice`, `MemcpyDeviceToHost`, `MemcpyDeviceToDevice`, `MemcpyHostToHost`)

## Build

```bash
# OpenCL (render + host JIT)
cc -O2 -std=c99 -o gpuni-render render.c         # build render tool
./gpuni-render saxpy.gu.cu -o saxpy.gu.h        # emit OpenCL source string header
c++  -I. host.cpp -lOpenCL                      # uses saxpy.gu.h for JIT

# CUDA/HIP (direct compile; no rendering)
nvcc  -I. host.cpp saxpy.gu.cu
hipcc -I. host.cpp saxpy.gu.cu
```
