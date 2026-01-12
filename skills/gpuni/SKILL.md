---
name: gpuni
description: "Write/refactor/review gpuni CUDA-truth dialect kernels (*.pk.cu) that compile as CUDA/HIP and render to OpenCL C 1.2 via tools/render.c; use for OpenCL 1.2 address spaces (__global/__local/__constant), uniform __syncthreads, portable atomics (atomic*, pk_atomic_*), and edits to gpuni.h or the renderer; keywords: gpuni, gpuni.h, render.c, OpenCL 1.2, HIP, CUDA dialect."
---

# gpuni

## Core idea (CUDA-truth + OpenCL 1.2 baseline)

Write one CUDA-style kernel source (source-of-truth) that:
- Compiles directly with `nvcc` and `hipcc` (no CUDA-side translation)
- Renders into a single-file OpenCL C 1.2 source via `tools/render.c`

## Workflow (AI-safe)

Canonical public repo (release): `git@github.com:vibegpu/gpuni.git`.

1) Locate the gpuni package root (contains `gpuni.h` and `tools/render.c`). In this dev repo it is `gpuni/`.  
2) Choose the task type:
   - New/edited kernel (`*.pk.cu`): write CUDA, include only `#include "gpuni.h"`, then apply the dialect checklist below.
   - OpenCL build error: first check address spaces (`__global/__local/__constant`) on pointer aliases + helper args, then check barriers.
   - Portability gap: add a mapping/implementation in `gpuni.h` (prefer CUDA spellings; introduce `pk_*` only when OpenCL 1.2 truly lacks it).
3) Validate:
   - CUDA/HIP: compile the `.pk.cu` directly (no translation).
   - OpenCL: render with the C99 renderer and (optionally) syntax-check with `clang -cl-std=CL1.2`.

## Quickstart (render to OpenCL C 1.2)

Run from the gpuni package root (the directory that contains `gpuni.h`).

```bash
mkdir -p tmp
cc -O2 -std=c99 -o tools/render tools/render.c
tools/render --help
tools/render path/to/kernel.pk.cu -o tmp/kernel.cl
```

## Dialect rules (short)

- Entry point: `PK_EXTERN_C __global__ void pk_<name>(...)`
- OpenCL 1.2 address spaces: every non-private pointer must be annotated (`__global/__local/__constant`), including pointer aliases.
- Uniform barriers only: every `__syncthreads()` must be reached by the whole block/work-group.
- Keep kernels C-like: no templates/classes/overloads/refs/exceptions/RTTI/`new`/`delete`/standard library.
- Correctness-first: avoid warp/subgroup intrinsics (`__shfl*`, `__ballot*`, `__syncwarp`, cooperative groups).
- Dynamic shared/local: use explicit `__local T* pk_smem` param + `PK_BIND_DYNAMIC_SMEM(pk_smem)`.
- Prefer CUDA/C99 spellings in code; use `pk_*` only for real OpenCL 1.2 portability gaps.

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

- CUDA/HIP compile (if toolchains exist): make sure `-I` points to the directory that contains `gpuni.h`.
  - Example (package root): `nvcc  -I. -c path/to/kernel.pk.cu`
  - Example (package root): `hipcc -I. -c path/to/kernel.pk.cu`
  - Example (dev repo root): `nvcc  -Igpuni -c path/to/kernel.pk.cu`
  - Example (dev repo root): `hipcc -Igpuni -c path/to/kernel.pk.cu`
- OpenCL render: `tools/render path/to/kernel.pk.cu -o tmp/kernel.cl`
  - Optional syntax-check: `clang -x cl -cl-std=CL1.2 -fsyntax-only tmp/kernel.cl`

## Review Rules (what to reject)

- Reject any change that makes kernels “not valid CUDA without a CUDA-side translation step”.
- Reject any attempt to “infer” OpenCL address spaces automatically for local pointer aliases; require explicit annotation.
- Reject any barrier placed in control flow that some threads may skip.

## Where to look (progressive disclosure)

- Full dialect rules + examples: `references/dialect.md`
- Dialect contract + API list: `README.md` (search: "Dialect contract", "Atomics", "Dynamic shared")
- Exact mappings/atomics: `gpuni.h` (search: "PK_BACKEND_", "atomic", "fixed_q32_32")
- Renderer behavior/options: `tools/render.c` (search: "Usage", "render")

## Files

- Public package: `gpuni.h`, `tools/render.c`, `README.md`, `skills/gpuni/SKILL.md`
