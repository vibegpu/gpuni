# gpuni

gpuni is **AI-first** and built for **cross-platform many-core GPU computing**: a small, explicit CUDA-truth kernel dialect that targets CUDA, HIP, and OpenCL C 1.2.

Start here:
- Write kernels as `*.pk.cu` and follow **Dialect contract (must)** below.
- For AI coding (Codex/Claude Code), load/activate the `gpuni` skill: `skills/gpuni/SKILL.md` (prompt: “use `$gpuni`”).

**Package:** `gpuni.h` + `tools/render.c` (+ optional `skills/`).

## Why gpuni

- **One kernel source:** write once in CUDA style, reuse across backends.
- **OpenCL 1.2 as baseline:** forces the “portable surface” (explicit address spaces, uniform barriers).
- **No kernel `#ifdef` maze:** backend differences live in `gpuni.h` and the render step.

## Quickstart

Write a kernel (typically `*.pk.cu`):

```cpp
#include "gpuni.h"

PK_EXTERN_C __global__ void pk_saxpy(int n,
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
tools/render my_kernel.pk.cu -o my_kernel.cl
```

Optional sanity checks (no runtime required):

```bash
nvcc  -I. -c my_kernel.pk.cu
hipcc -I. -c my_kernel.pk.cu
clang -x cl -cl-std=CL1.2 -fsyntax-only my_kernel.cl  # optional
```

## Dialect contract (must)

- **Entry point:** `PK_EXTERN_C __global__ void pk_<name>(...)`
- **Include:** only `#include "gpuni.h"` in dialect kernels (avoid other includes on the OpenCL path)
- **C-like subset:** no templates/classes/overloads/references/exceptions/RTTI/`new`/`delete`/standard library
- **CUDA/C99 spellings in kernels:** use `sinf/expf/...` and `atomicAdd/atomicCAS/...`; use `pk_*` only for real OpenCL 1.2 gaps.
- **Explicit pointer address spaces (OpenCL 1.2):** every non-private pointer must be `__global/__local/__constant` (params + aliases + helper args). Legacy synonyms: `PK_GLOBAL/PK_LOCAL/PK_CONSTANT`, `PK_*_PTR(T)`.
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

## What `gpuni.h` provides (PK_DIALECT_VERSION=1)

- **Backends/caps:** `PK_BACKEND_{CUDA,HIP,OPENCL,HOST}`, `PK_HAS_{FP64,I64_ATOMICS,LOCAL_ATOMICS}`
- **Types:** `pk_{i32,u32,i64,u64}`, `pk_real` (default `float`; define `PK_USE_DOUBLE` and check `PK_REAL_IS_*`)
- **Builtins:** `threadIdx`, `blockIdx`, `blockDim`, `gridDim` (`x/y/z`)
- **Keywords:** `__global__`, `__device__`, `__host__`, `__shared__`, `__constant__`, `__launch_bounds__(t,b)`
- **Address spaces:** `__global/__local/__constant`, plus legacy `PK_*` helpers
- **Utilities:** `PK_RESTRICT`, `PK_INLINE`, `PK_EXTERN_C`, `PK_BIND_DYNAMIC_SMEM(ptr)`
- **Math:** CUDA-style `*f` float math is mapped for OpenCL

### Atomics (portable baseline)

OpenCL 1.2 core atomics are **32-bit integer**. For portable float accumulation, prefer **fixed-point(Q32.32) + integer atomics**.

Provided APIs:
- `atomicAdd/atomicSub/atomicExch/atomicMin/atomicMax/atomicAnd/atomicOr/atomicXor/atomicCAS` (CUDA-style; **`int`/`unsigned int` only** on the portable OpenCL 1.2 baseline)
- `pk_atomic_add_u64` (returns `void`; OpenCL may require int64 atomics support, else uses a portable accumulation fallback)
- `pk_atomic_add_f32` (OpenCL 1.2 CAS fallback; correctness-first, slower than fixed-point)
- `pk_real_to_fixed_q32_32`, `pk_fixed_q32_32_to_real`, `pk_atomic_add_fixed_q32_32`

Minimal usage pattern:

```cpp
// acc points to a Q32.32 buffer (pk_u64 per element)
pk_atomic_add_fixed_q32_32(acc + i, value);
```

### Dynamic shared memory (portable ABI)

OpenCL needs an explicit `__local` kernel argument; CUDA/HIP use `extern __shared__`:

```cpp
PK_EXTERN_C __global__ void pk_reduce_sum(/* ... */, __local float* pk_smem) {
  PK_BIND_DYNAMIC_SMEM(pk_smem);  // OpenCL: no-op; CUDA/HIP: binds extern __shared__
  __local float* s = pk_smem;
  /* ... */
}
```

Host-side contract:
- CUDA/HIP: set `smem_bytes` as the dynamic shared size; pass `NULL` for `pk_smem`
- OpenCL: `clSetKernelArg(pk_smem_arg_index, smem_bytes, NULL)`

Note (important):
- Prefer a **typed** local parameter (`__local float*`, `__local int*`, ...) for dynamic shared memory.
  Some OpenCL drivers may only guarantee alignment based on the pointee type; `__local unsigned char*` + cast can crash.

## Not in Scope (v1)

- Warp/subgroup intrinsics for correctness (`__shfl*`, cooperative groups): use `__shared__/__local + __syncthreads()`.
- Float atomics as a required feature: use fixed-point(Q32.32) helpers.
- CUDA-only features (tensor cores/WMMA, dynamic parallelism, inline PTX, textures/surfaces).
- “Big library” layers (FFT/BLAS/Thrust-like APIs): bind external libs per backend if needed.

## License

MIT (see `LICENSE`).
