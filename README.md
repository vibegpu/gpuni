# gpuni

gpuni is **AI-first** and built for **cross-platform many-core GPU computing**: a small, explicit CUDA-truth kernel dialect that targets CUDA, HIP, and OpenCL C 1.2.

Start here:
- Write kernels as `*.gu.cu` and follow **Dialect contract (must)** below.
- For AI coding (Codex/Claude Code), activate the `gpuni` skill: `skills/gpuni/SKILL.md` (Codex: use `$gpuni`; Claude Code: say `Use the gpuni skill`).

**Package:** `gpuni.h` + `tools/render.c` (+ optional `gpunih.h`, `skills/`).

## Why gpuni

- **One kernel source:** write once in CUDA style, reuse across backends.
- **OpenCL 1.2 as baseline:** forces the “portable surface” (explicit address spaces, uniform barriers).
- **No kernel `#ifdef` maze:** backend differences live in `gpuni.h` and the render step.

## Quickstart

Write a kernel (typically `*.gu.cu`):

```cpp
#include "gpuni.h"

GU_EXTERN_C __global__ void gu_saxpy(int n,
                                    __global float* y,
                                    __global const float* x,
                                    float a) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) y[i] = a * x[i] + y[i];
}
```

Build / render:

```bash
# OpenCL C 1.2 (render to a single .cl)
cc -O2 -std=c99 -o tools/render tools/render.c
tools/render my_kernel.gu.cu -o my_kernel.cl
```

Optional sanity checks (no runtime required):

```bash
nvcc  -I. -c my_kernel.gu.cu
hipcc -I. -c my_kernel.gu.cu
clang -x cl -cl-std=CL1.2 -fsyntax-only my_kernel.cl  # optional
```

## Dialect contract (must)

- **Entry point:** `GU_EXTERN_C __global__ void gu_<name>(...)`
- **Include:** only `#include "gpuni.h"` in dialect kernels (avoid other includes on the OpenCL path)
- **C-like subset:** no templates/classes/overloads/references/exceptions/RTTI/`new`/`delete`/standard library
- **CUDA/C99 spellings in kernels:** use `sinf/expf/...` and `atomicAdd/atomicCAS/...`; use `gu_*` only for real OpenCL 1.2 gaps.
- **Pointer address spaces (OpenCL 1.2):** annotate every non-private pointer with `__global/__local/__constant` (params + aliases + helper args). Required for OpenCL; no-ops under CUDA/HIP via `gpuni.h`. Prefer `__global/__local/__constant`; synonyms: `GU_GLOBAL/GU_LOCAL/GU_CONSTANT`, `GU_*_PTR(T)`.
- **Don’t confuse:** `__global__` (kernel qualifier) vs `__global` (pointer address space qualifier in OpenCL)
- **Uniform barriers:** every `__syncthreads()` is reached by the whole block/work-group (no divergent barrier / early return)
- **Correctness-first:** don’t rely on warp/subgroup intrinsics (`__shfl*`, `__ballot*`, `__syncwarp`, cooperative groups)
- **ABI/layout:** don’t store `float3/int3/...3` in global/constant buffers or structs (use `float4` or SoA)

## OpenCL 1.2 pointer rule (the one that bites)

OpenCL 1.2 has **no generic pointer**: an unqualified pointer defaults to `__private`, so aliases must keep address space:

```cpp
// x is __global; p must also be __global (otherwise it becomes __private in OpenCL)
__global const float* p = x + off;

__shared__ float tile[256];
__local float* t = tile;
```

## What `gpuni.h` provides (GU_DIALECT_VERSION=1)

- **Backends/caps:** `GU_BACKEND_{CUDA,HIP,OPENCL,HOST}`, `GU_HAS_{FP64,I64_ATOMICS,LOCAL_ATOMICS}`
- **Types:** `gu_{i32,u32,i64,u64}`, `gu_real` (default `float`; define `GU_USE_DOUBLE` and check `GU_REAL_IS_*`)
- **Builtins:** `threadIdx`, `blockIdx`, `blockDim`, `gridDim` (`x/y/z`)
- **Keywords:** `__global__`, `__device__`, `__host__`, `__shared__`, `__constant__`, `__launch_bounds__(t,b)`
- **Address spaces:** `__global/__local/__constant`, plus legacy `GU_*` helpers
- **Utilities:** `GU_RESTRICT`, `GU_INLINE`, `GU_EXTERN_C`, `GU_BIND_DYNAMIC_SMEM(ptr)`
- **Math:** CUDA-style `*f` float math is mapped for OpenCL

### Atomics (portable baseline)

OpenCL 1.2 core atomics are **32-bit integer**. For portable float accumulation, prefer **fixed-point(Q32.32) + integer atomics**.

Provided APIs:
- `atomicAdd/atomicSub/atomicExch/atomicMin/atomicMax/atomicAnd/atomicOr/atomicXor/atomicCAS` (CUDA-style; **`int`/`unsigned int` only** on the portable OpenCL 1.2 baseline)
- `gu_atomic_add_u64` (returns `void`; OpenCL may require int64 atomics support, else uses a portable accumulation fallback)
- `gu_atomic_add_f32` (OpenCL 1.2 CAS fallback; correctness-first, slower than fixed-point)
- `gu_real_to_fixed_q32_32`, `gu_fixed_q32_32_to_real`, `gu_atomic_add_fixed_q32_32`

Minimal usage pattern:

```cpp
// acc points to a Q32.32 buffer (gu_u64 per element)
gu_atomic_add_fixed_q32_32(acc + i, value);
```

### Dynamic shared memory (portable ABI)

OpenCL needs an explicit `__local` kernel argument; CUDA/HIP use `extern __shared__`:

```cpp
GU_EXTERN_C __global__ void gu_reduce_sum(/* ... */, __local float* gu_smem) {
  GU_BIND_DYNAMIC_SMEM(gu_smem);  // OpenCL: no-op; CUDA/HIP: binds extern __shared__
  __local float* s = gu_smem;
  /* ... */
}
```

Host-side contract:
- CUDA/HIP: set `smem_bytes` as the dynamic shared size; pass `NULL` for `gu_smem`
- OpenCL: `clSetKernelArg(gu_smem_arg_index, smem_bytes, NULL)`

Note (important):
- Prefer a **typed** local parameter (`__local float*`, `__local int*`, ...) for dynamic shared memory.
  Some OpenCL drivers may only guarantee alignment based on the pointee type; `__local unsigned char*` + cast can crash.

## Host API (`gpunih.h`)

Optional unified host-side API for context, memory, and kernel launch:

```cpp
#include "gpunih.h"  // auto-detects GUH_CUDA/GUH_HIP, or define GUH_OPENCL
```

**Types:** `gu_ctx`, `gu_kernel`

**API:**
- Context: `gu_ctx_init(&ctx, dev)` / `gu_ctx_destroy(&ctx)` / `gu_sync(&ctx)`
- Memory: `gu_malloc(&ctx, n)` / `gu_free(&ctx, p)`
- Memcpy: `gu_h2d(&ctx, d, s, n)` / `gu_d2h(&ctx, d, s, n)` / `gu_d2d(&ctx, d, s, n)`
- Kernel: `GU_KERNEL(&ctx, &k, gu_<name>)` / `gu_kernel_destroy(&k)`
- Launch: `gu_arg(&k, val)` / `gu_run(&ctx, &k, grid, block, smem)` (args auto-reset)

**Usage example:**

```cpp
#include "gpunih.h"
#include "saxpy_cl.h"  // always include: OpenCL gets source, CUDA/HIP gets no-op

// Kernel declaration (CUDA/HIP needs it; harmless for OpenCL)
GU_EXTERN_C __global__ void gu_saxpy(int, __global float*, __global const float*, float);

int main() {
  gu_ctx ctx; gu_kernel k;
  int n = 1024;
  float a = 2.0f;
  float *h_x, *h_y;  // host arrays

  gu_ctx_init(&ctx, 0);
  void* d_x = gu_malloc(&ctx, n * sizeof(float));
  void* d_y = gu_malloc(&ctx, n * sizeof(float));
  gu_h2d(&ctx, d_x, h_x, n * sizeof(float));
  gu_h2d(&ctx, d_y, h_y, n * sizeof(float));

  GU_KERNEL(&ctx, &k, gu_saxpy);  // unified: works for CUDA/HIP/OpenCL
  gu_arg(&k, n); gu_arg(&k, d_y); gu_arg(&k, d_x); gu_arg(&k, a);
  gu_run(&ctx, &k, (n + 255) / 256, 256, 0);

  gu_sync(&ctx);
  gu_d2h(&ctx, h_y, d_y, n * sizeof(float));

  gu_kernel_destroy(&k);
  gu_free(&ctx, d_x); gu_free(&ctx, d_y);
  gu_ctx_destroy(&ctx);
  return 0;
}
```

**OpenCL build step:**

```bash
# Render kernel to .cl and generate source header
tools/render saxpy.gu.cu -o saxpy.cl --emit-header saxpy_cl.h
```

The `GU_KERNEL` macro:
- CUDA/HIP: uses function pointer directly (header defines `gu_<name>_cl_source` as no-op)
- OpenCL: uses `gu_<name>_cl_source` string from the generated header

The generated header is safe to include unconditionally—no `#if defined(GUH_OPENCL)` needed.

## Not in Scope (v1)

- Warp/subgroup intrinsics for correctness (`__shfl*`, cooperative groups): use `__shared__/__local + __syncthreads()`.
- Float atomics as a required feature: use fixed-point(Q32.32) helpers.
- CUDA-only features (tensor cores/WMMA, dynamic parallelism, inline PTX, textures/surfaces).
- “Big library” layers (FFT/BLAS/Thrust-like APIs): bind external libs per backend if needed.

## License

MIT (see `LICENSE`).
