---
name: polykernel
description: Write, refactor, and review PolyKernel CUDA-truth dialect GPU kernels and shims that must compile as CUDA/HIP and be renderable to OpenCL C 1.2. Use for authoring `.pk.cu` kernels with explicit OpenCL 1.2 address spaces (`__global/__local/__constant`), uniform barriers, C-like subset restrictions, and for generating OpenCL-ready sources via `tools/render.c`.
---

# PolyKernel

## Overview

Use the minimal PolyKernel dialect to write one CUDA-style kernel source (source-of-truth) that:
- Compiles directly with `nvcc` and `hipcc` (no translation step)
- Can be rendered into a single-file OpenCL C 1.2 source via the PolyKernel renderer

## Workflow (AI-safe)

1) Find the PolyKernel package root (it contains `polykernel.h` and `tools/render.c`).  
2) Write a kernel as CUDA (file typically `*.pk.cu`) and include `#include "polykernel.h"`.  
3) Apply the dialect checklist below before iterating.  
4) Validate by compiling (CUDA/HIP) and rendering (OpenCL).

Build and run the renderer (run from the directory that contains `polykernel.h`):

```bash
cc -O2 -std=c99 -o tools/render tools/render.c
tools/render my_kernel.pk.cu -o my_kernel.cl
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

### Synchronization correctness

- Ensure every `__syncthreads()` is reached uniformly by the whole block/work-group (no divergent barrier, no early return before a barrier).

### Keep the subset portable

- Avoid templates/classes/overloads/references/exceptions/RTTI/`new`/`delete`/standard library usage.
- Avoid warp/subgroup intrinsics for correctness (`__shfl*`, `__ballot*`, `__syncwarp`, cooperative groups).

## Dynamic shared memory pattern (portable ABI)

- Put dynamic shared/local memory as an explicit kernel argument: `__local unsigned char* pk_smem`
- Call `PK_BIND_DYNAMIC_SMEM(pk_smem)` to bind CUDA/HIP `extern __shared__` (OpenCL is a no-op).

```cpp
PK_EXTERN_C __global__ void pk_reduce_sum(/* ... */, __local unsigned char* pk_smem) {
  PK_BIND_DYNAMIC_SMEM(pk_smem);
  __local float* sdata = (__local float*)pk_smem;
  /* ... */
}
```

## Review Rules (what to reject)

- Reject any change that makes kernels “not valid CUDA without a CUDA-side translation step”.
- Reject any attempt to “infer” OpenCL address spaces automatically for local pointer aliases; require explicit annotation.
- Reject any barrier placed in control flow that some threads may skip.

## Files

- Public package: `polykernel.h`, `tools/render.c`, `README.md`, `skills/polykernel/SKILL.md`
