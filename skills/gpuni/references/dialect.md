# gpuni Dialect Rules (reference)

Read this when writing/reviewing `*.pk.cu` kernels or diagnosing OpenCL 1.2 compile errors.

## 1) File shape (OpenCL render path)

- Kernel files should include **only** `#include "gpuni.h"` (avoid extra includes on the OpenCL path).
- Keep kernel code “CUDA-like and C-like” so it can become OpenCL C 1.2 after rendering.

## 2) Kernel entry points

- Every kernel entry point must be:

```cpp
PK_EXTERN_C __global__ void pk_<name>(/* params */) { /* ... */ }
```

## 3) OpenCL 1.2 address spaces (critical)

OpenCL C 1.2 has **no generic pointers**. Any pointer without an address space defaults to
`__private`, which breaks aliases and helper function args.

### 3.1 Kernel parameters

Annotate every non-private pointer parameter:

```cpp
PK_EXTERN_C __global__ void pk_saxpy(int n,
                                    __global float* y,
                                    __global const float* x,
                                    float a) {
  /* ... */
}
```

### 3.2 Pointer aliases / temporaries

If a pointer points into `__global`/`__local`/`__constant`, the alias must carry the same address
space in its type.

```cpp
// WRONG for OpenCL: p becomes __private float*
float* p = x + off;

// CORRECT
__global float* p = x + off;
```

### 3.3 Shared/local pointers

```cpp
__shared__ float tile[256];

// Alias must be __local in OpenCL (CUDA: __shared__ maps to __local via gpuni.h)
__local float* t = tile;
```

### 3.4 Helper function pointer arguments

Same rule for helper functions:

```cpp
static PK_INLINE void helper(__global float* x) { /* ... */ }
```

## 4) Barriers must be uniform

OpenCL has only work-group barriers: if you call `barrier()` (via `__syncthreads()`), **every**
work-item in the work-group must reach it.

```cpp
// WRONG: divergent barrier
if (threadIdx.x < 128) __syncthreads();

// CORRECT
__syncthreads();
if (threadIdx.x < 128) { /* ... */ }
```

Rule of thumb: no early return / break / continue that can skip a later `__syncthreads()`.

## 5) Allowed subset (keep it renderable)

Avoid C++ features that cannot become OpenCL C 1.2:

- No templates, classes, overloads, references
- No exceptions/RTTI
- No `new`/`delete`
- No standard library usage

## 6) Warp/subgroup intrinsics

Correctness-first baseline: don’t use warp/subgroup intrinsics:
`__shfl*`, `__ballot*`, `__syncwarp`, cooperative groups.

If an algorithm depends on them, rewrite using `__shared__`/`__local` + `__syncthreads()` (may be slower).

## 7) Atomics (portable baseline)

- Portable OpenCL 1.2 baseline provides **32-bit integer atomics** only.
- Use CUDA spellings (`atomicAdd`, `atomicCAS`, …) in kernels; `gpuni.h` maps them for OpenCL.
- Use `pk_*` only when you need semantics not present in OpenCL 1.2 (e.g. float atomics).

Recommended pattern for portable float accumulation:
- Use fixed-point Q32.32 accumulation (`pk_u64` buffer) + `pk_atomic_add_fixed_q32_32`.

## 8) Dynamic shared/local memory

OpenCL needs an explicit `__local` argument; CUDA/HIP use `extern __shared__`.

```cpp
PK_EXTERN_C __global__ void pk_reduce_sum(/* ... */, __local float* pk_smem) {
  PK_BIND_DYNAMIC_SMEM(pk_smem);  // OpenCL: no-op; CUDA/HIP: binds extern __shared__
  __local float* s = pk_smem;
  /* ... */
}
```

## 9) Quick diagnosis

- OpenCL error like “pointer without address space” / “cannot convert `__global T*` to `__private T*`”:
  add `__global/__local/__constant` to the pointer alias or helper argument type.
- OpenCL hang / deadlock:
  look for divergent `__syncthreads()` (barrier) or early exit before a barrier.
