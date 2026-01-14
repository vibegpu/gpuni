# gpuni

A small AI-friendly CUDA-truth kernel dialect for cross-platform GPU compute (CUDA, HIP, OpenCL C 1.2).

**For AI coding (Codex/Claude Code):** load the `gpuni` skill at `skills/gpuni/SKILL.md` (prompt: use `$gpuni`).

**Package:** `gpuni.h` + `tools/render.c`

## Kernel

Write `*.gu.cu`:

```cpp
#include "gpuni.h"

GU_EXTERN_C __global__ void gu_saxpy(int n,
                                     GU_GLOBAL float* y,
                                     GU_GLOBAL const float* x,
                                     float a) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  GU_GLOBAL const float* p = x + i;   // pointer alias must keep qualifier
  __shared__ float tile[256];         // shared memory
  GU_LOCAL float* t = tile;           // alias to shared needs GU_LOCAL
  if (i < n) y[i] = a * (*p) + y[i];
}
```

## Dialect Rules

**Required:**
- Entry: `GU_EXTERN_C __global__ void gu_<name>(...)`
- Annotate pointers: `GU_GLOBAL` / `GU_LOCAL` / `GU_CONSTANT` (including aliases)

**Avoid:** templates, classes, `__shfl*`, `__ballot*`, `float3` in buffers, divergent `__syncthreads()`

## Kernel API Reference

| Category | API |
|----------|-----|
| Types | `int`, `uint`, `int64`, `uint64`, `float`, `double` |
| Atomics (int) | `atomicAdd`, `atomicSub`, `atomicExch`, `atomicMin`, `atomicMax`, `atomicCAS`, `atomicAnd`, `atomicOr`, `atomicXor`, `guAtomicAddU64` (void, add-only) |
| Atomics (float) | `guAtomicAddF`, `guAtomicMinF`, `guAtomicMaxF` |
| Accumulator | `gu_atomic_add_fixed_q32_32(ptr, val)` - portable float accumulation via Q32.32 fixed-point; convert back with `gu_fixed_q32_32_to_real()` |
| Dynamic smem | `GU_BIND_DYNAMIC_SMEM(gu_smem)` with `GU_LOCAL float* gu_smem` as **last param** |
| Restrict | `GU_RESTRICT` (pointer no-alias hint) |
| Math | CUDA-style `sinf`, `cosf`, `rsqrtf`, `fminf`, `fmaxf`, `fmaf`, etc. work directly |

## Host

```cpp
#include "gpuni.h"
#include "saxpy.gu.h"  // OpenCL JIT needs this; CUDA/HIP auto

int main() {
  int n = 1024; float a = 2.0f;

  gu::SetDevice(0);  // must call before Malloc/GU_KERNEL

  float* d_x = gu::Malloc<float>(n);
  float* d_y = gu::Malloc<float>(n);
  float* h_x = gu::MallocHost<float>(n);  // pinned memory
  float* h_y = gu::MallocHost<float>(n);

  for (int i = 0; i < n; i++) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

  gu::Memcpy(d_x, h_x, n * sizeof(float), gu::H2D);
  gu::Memcpy(d_y, h_y, n * sizeof(float), gu::H2D);

  auto k = GU_KERNEL(gu_saxpy);  // cache and reuse; avoid repeated JIT
  gu::Launch(k, (n + 255) / 256, 256, n, d_y, d_x, a);

  gu::DeviceSync();
  gu::Memcpy(h_y, d_y, n * sizeof(float), gu::D2H);

  gu::Free(d_x); gu::Free(d_y);
  gu::FreeHost(h_x); gu::FreeHost(h_y);
}
```

### Host API Reference

| Category | API |
|----------|-----|
| Device | `SetDevice(id)`, `GetDevice()`, `GetDeviceCount()`, `DeviceSync()` |
| Memory | `Malloc<T>(n)`, `Free(p)`, `Memset(p,v,bytes)`, `MallocHost<T>(n)`, `FreeHost(p)` |
| Copy | `Memcpy(dst,src,bytes,kind)`, `MemcpyAsync(...,stream)` |
| Kernel | `GU_KERNEL(fn)`, `Launch(k, grid, block, args...)` |
| Stream | `stream s; s.sync();` |
| Event | `event e; e.record(s); e.sync(); ElapsedTime(e1,e2)` |
| Error | `GetLastError()`, `GetErrorString(e)`, `GU_CHECK(expr)` |
| Dim3 | `dim3(x,y,z)` for 3D grid/block in `Launch(k, dim3 grid, dim3 block, ...)` |

**MemcpyKind:** `H2D`, `D2H`, `D2D`, `H2H`
**Launch overloads:** `Launch(k, g, b, args)`, `Launch(k, g, b, smem, args)`, `Launch(k, g, b, stream, args)`, `Launch(k, g, b, smem, stream, args)`

## Build

```bash
cc -O2 -std=c99 -o gpuni-render tools/render.c   # build render tool
./gpuni-render saxpy.gu.cu -o saxpy.gu.h        # OpenCL needs kernel source string
nvcc  -I. host.cpp saxpy.gu.cu
hipcc -I. host.cpp saxpy.gu.cu
c++   -I. host.cpp -lOpenCL                # uses saxpy.gu.h for JIT
```
