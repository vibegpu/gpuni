---
name: polykernel
description: >
  Write, refactor, and review PolyKernel CUDA-truth dialect GPU kernels that compile as CUDA/HIP
  and render to OpenCL C 1.2.

  Use when: creating/editing `.pk.cu` kernels; adding OpenCL 1.2 address-space annotations
  (`__global/__local/__constant`) on pointer params/aliases; ensuring uniform barriers
  (`__syncthreads()`); implementing portable atomics (`pk_atomic_*`); using/debugging the renderer
  (`tools/render.c`) and OpenCL address-space errors.

  Keywords: polykernel, dialect CUDA, 方言CUDA, OpenCL 1.2, HIP, 渲染, render, pk.cu, polykernel.h.
---

# PolyKernel

## Overview

Use the minimal PolyKernel dialect to write one CUDA-style kernel source (source-of-truth) that:
- Compiles directly with `nvcc` and `hipcc` (no translation step)
- Can be rendered into a single-file OpenCL C 1.2 source via the PolyKernel renderer

## Repo Guardrails (this repo)

- Don’t run `git` commands in this environment.
- Treat `tmp/` as read-only reference snapshots; don’t copy/paste code from `tmp/` into publishable code.

## Workflow (AI-safe)

1) Locate the PolyKernel package root (contains `polykernel.h` and `tools/render.c`). In this repo it is `polykernel/` (then run commands from there).  
2) Write a kernel as CUDA (file typically `*.pk.cu`) and include `#include "polykernel.h"`.  
3) Apply the dialect checklist below before iterating.  
4) Validate by compiling (CUDA/HIP) and rendering (OpenCL).

## Quickstart (render to OpenCL C 1.2)

Run from the PolyKernel package root (the directory that contains `polykernel.h`).

```bash
mkdir -p tmp/bin
cc -O2 -std=c99 -o tmp/bin/pk_render tools/render.c
tmp/bin/pk_render path/to/kernel.pk.cu -o tmp/kernel.cl
```

## Dialect Checklist (must)

### Entry point shape

- Use `PK_EXTERN_C __global__ void pk_<name>(...)` for every kernel entry.
- Keep kernel code “CUDA-like and C-like” so it can become OpenCL C 1.2 after rendering.

### OpenCL 1.2 address spaces (most important)

OpenCL C 1.2 has **no generic pointer**: any pointer that is not private must carry an address space.

- Annotate every non-private pointer type (kernel params, pointer aliases, helper function args) with:
  - `__global` / `__local` / `__constant`
- Do not confuse:
  - `__global__` (CUDA kernel qualifier) vs `__global` (OpenCL address-space for pointers)

Examples:

```cpp
PK_EXTERN_C __global__ void pk_saxpy(int n,
                                    __global float* y,
                                    __global const float* x,
                                    float a) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) y[i] = a * x[i] + y[i];
}

// Pointer alias must keep address space (OpenCL would default to __private otherwise)
__global const float* p = x + off;

__shared__ float tile[256];
__local float* t = tile;
```

### Common Mistakes (fast rejects)

#### Missing address space on pointer aliases

```cpp
// WRONG (OpenCL): p becomes __private float*, cannot point to __global memory
float* p = x + off;

// CORRECT
__global float* p = x + off;
```

#### Divergent barrier

```cpp
// WRONG: not all threads reach the barrier
if (threadIdx.x < 128) __syncthreads();

// CORRECT: all threads reach the barrier
__syncthreads();
if (threadIdx.x < 128) { /* ... */ }
```

### Prefer CUDA/C99 spellings

- Prefer CUDA/C99 spellings in kernels (math `sinf/expf/...`, integer atomics `atomicAdd/atomicCAS/...`).
- Use `pk_*` only when OpenCL 1.2 needs extra code to emulate missing semantics (e.g. float atomics, `u64` add).

### Synchronization correctness

- Ensure every `__syncthreads()` is reached uniformly by the whole block/work-group (no divergent barrier, no early return before a barrier).

### Keep the subset portable

- Avoid templates/classes/overloads/references/exceptions/RTTI/`new`/`delete`/standard library usage.
- Avoid warp/subgroup intrinsics for correctness (`__shfl*`, `__ballot*`, `__syncwarp`, cooperative groups).
- Avoid storing `float3/int3/...3` in global/constant buffers or structs (layout/ABI pitfalls); use `float4` or SoA.

## Dynamic shared memory pattern (portable ABI)

- Put dynamic shared/local memory as an explicit kernel argument: `__local T* pk_smem`
- Call `PK_BIND_DYNAMIC_SMEM(pk_smem)` to bind CUDA/HIP `extern __shared__` (OpenCL is a no-op).

```cpp
PK_EXTERN_C __global__ void pk_reduce_sum(/* ... */, __local float* pk_smem) {
  PK_BIND_DYNAMIC_SMEM(pk_smem);
  __local float* sdata = pk_smem;
  /* ... */
}
```

## Validation (fast)

- CUDA/HIP compile (if toolchains exist): make sure `-I` points to the directory that contains `polykernel.h`.
  - Example (this repo): `nvcc  -Ipolykernel -c path/to/kernel.pk.cu`
  - Example (this repo): `hipcc -Ipolykernel -c path/to/kernel.pk.cu`
- OpenCL render: `tmp/bin/pk_render path/to/kernel.pk.cu -o tmp/kernel.cl`
  - Optional syntax-check: `clang -x cl -cl-std=CL1.2 -fsyntax-only tmp/kernel.cl`

## Review Rules (what to reject)

- Reject any change that makes kernels “not valid CUDA without a CUDA-side translation step”.
- Reject any attempt to “infer” OpenCL address spaces automatically for local pointer aliases; require explicit annotation.
- Reject any barrier placed in control flow that some threads may skip.

## Where to look (progressive disclosure)

- Dialect contract + API list: `README.md` (search: "Dialect contract", "Atomics", "Dynamic shared")
- Exact mappings/atomics: `polykernel.h` (search: "PK_BACKEND_", "atomic", "fixed_q32_32")
- Renderer behavior/options: `tools/render.c` (search: "Usage", "render")

## Files

- Public package: `polykernel.h`, `tools/render.c`, `README.md`, `skills/polykernel/SKILL.md`
