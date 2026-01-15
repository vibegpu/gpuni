---
name: gpuni
description: >-
  Write cross-platform GPU compute kernels using gpuni CUDA-truth dialect (*.gu.cu).
  Kernels compile natively as CUDA/HIP and render to OpenCL C 1.2.
  Use when: (1) creating/editing *.gu.cu files, (2) writing portable GPU parallel code,
  (3) using address-space qualifiers (__global/__local/__constant),
  (4) implementing portable atomics (atomicAddFloat, atomicAddFixed),
  (5) using dynamic shared memory (bindSharedMem),
  (6) reviewing CUDA/HIP/OpenCL portability,
  (7) debugging OpenCL render errors.
---

# gpuni

Write portable GPU kernels in CUDA-truth dialect. Compiles as CUDA/HIP, renders to OpenCL C 1.2.

## Critical Rules (always apply)

```cpp
// 0. Dialect file basics
// - Only include "gpuni.h" (no <...> system includes)

// 1. Entry signature
extern "C" __global__ void kernel_name(...)

// 2. All pointers AND aliases must have address space
__global float* output,           // param
__global float* p = output;       // alias - MUST keep __global
__local float* tile = smem + n;   // alias - MUST keep __local

// 3. Barriers must be uniform (all threads reach it)
__syncthreads();  // ✓ outside if
if (cond) __syncthreads();  // ✗ divergent = deadlock
```

## Workflow

1. Apply Critical Rules above for simple kernels
2. Need API details? Read `references/README.md`
3. Need code templates? Read `references/examples.md`
4. Compile error? Check `references/dialect.md` (especially for OpenCL)

## Review Checklist

- [ ] All pointers have `__global`/`__local`/`__constant`
- [ ] All pointer aliases retain address-space qualifier
- [ ] `__syncthreads()` reachable by all threads (no divergent barriers)
- [ ] Entry uses `extern "C" __global__ void`
- [ ] Dynamic smem param is last + `bindSharedMem()` called
- [ ] No warp intrinsics (`__shfl*`, `__ballot*`)
- [ ] No system includes in kernels (no `<...>`; only `"gpuni.h"` or `"gpuni/..."`)

## References

| File | When to read |
|------|--------------|
| `references/README.md` | Dialect rules + API (kernel & host) |
| `references/examples.md` | Need complete code templates |
| `references/dialect.md` | Compile errors (OpenCL error → fix mapping) |

## Package

Source: **https://github.com/vibegpu/gpuni**

| File | Purpose |
|------|---------|
| `gpuni.h` | Core header (backend detection, macros, portable APIs) |
| `tools/render.c` | Offline CUDA→OpenCL renderer |
| `README.md` | Usage guide |
| `skills/gpuni/` | AI skill definitions |
