# gpuni

A small AI-friendly CUDA-truth kernel dialect for cross-platform GPU compute (CUDA, HIP, OpenCL C 1.2).

**For AI coding (Codex/Claude Code):** load the `gpuni` skill at `skills/gpuni/SKILL.md` (prompt: use `$gpuni`).

**Package:** `gpuni.h` + `tools/render.c`

## Kernel

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

## Dialect Rules

**Required:**
- Entry: `extern "C" __global__ void <name>(...)` (prevents C++ name mangling)
- Annotate pointers: `__global` / `__local` / `__constant` (including aliases)

**Avoid:** templates, classes, `__shfl*`, `__ballot*`, `float3` in buffers, divergent `__syncthreads()`

## Kernel API Reference

| Category | API |
|----------|-----|
| Types | `int`, `uint`, `int64`, `uint64`, `float`, `double` |
| Atomics (int) | `atomicAdd`, `atomicSub`, `atomicExch`, `atomicMin`, `atomicMax`, `atomicCAS`, `atomicAnd`, `atomicOr`, `atomicXor` |
| Atomics (float) | `atomicAddFloat`, `atomicMinFloat`, `atomicMaxFloat` |
| Accumulator (Q32.32) | Kernel: `atomicAddFixed(int64* acc, double v)`. Host: `DoubleToFixed(double v)`, `FixedToDouble(int64 acc)`. Usage: (1) init `int64 acc=0`, (2) kernel calls `atomicAddFixed(&acc, v)`, (3) host reads `FixedToDouble(acc)`. Range ±2^31 (~2e9), ~9 digits. |
| Dynamic smem | `__local T* smem` as **last kernel parameter** (must be last!) + `bindSharedMem(smem)` at function start |
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

  auto k = GetKernel(saxpy);  // cache and reuse; avoid repeated JIT
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
| Kernel | `GetKernel(fn)`, `Launch(k, grid, block, args...)` |
| Stream | `stream s; s.sync();` or `StreamSynchronize(s)` |
| Event | `event e; e.record(s); e.sync();` or `EventRecord(e,s); EventSynchronize(e)` |
| Timing | `ElapsedTime(e1, e2)` |
| Error | `Error_t`, `Success`, `GetLastError()`, `GetErrorString(e)`, `Check(expr)` |
| Dim3 | `dim3(x,y,z)` for 3D grid/block in `Launch(k, dim3 grid, dim3 block, ...)` |

**MemcpyKind:** `H2D`, `D2H`, `D2D`, `H2H` (or `MemcpyHostToDevice`, `MemcpyDeviceToHost`, `MemcpyDeviceToDevice`, `MemcpyHostToHost`)
**Launch overloads:** All combinations of `int` or `dim3` for grid/block, with optional `smem` and/or `stream`:
- `Launch(k, g, b, args)`, `Launch(k, dim3, dim3, args)`
- `Launch(k, g, b, smem, args)`, `Launch(k, dim3, dim3, smem, args)`
- `Launch(k, g, b, stream, args)`, `Launch(k, dim3, dim3, stream, args)`
- `Launch(k, g, b, smem, stream, args)`, `Launch(k, dim3, dim3, smem, stream, args)`

## Build

```bash
# OpenCL (render + host JIT)
cc -O2 -std=c99 -o gpuni-render tools/render.c   # build render tool
./gpuni-render saxpy.gu.cu -o saxpy.gu.h        # emit OpenCL source string header
c++  -I. host.cpp -lOpenCL                      # uses saxpy.gu.h for JIT

# CUDA/HIP (direct compile; no rendering)
nvcc  -I. host.cpp saxpy.gu.cu
hipcc -I. host.cpp saxpy.gu.cu
```
