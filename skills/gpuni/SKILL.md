---
name: gpuni
description: >-
  Write, refactor, and review gpuni CUDA-truth dialect kernels (*.pk.cu)
  that compile as CUDA/HIP and render to OpenCL C 1.2 via tools/render.c.
  Use when: (1) Creating/editing *.pk.cu kernels, (2) Adding OpenCL 1.2
  address spaces (__global/__local/__constant), (3) Implementing portable
  atomics (atomicAdd, pk_atomic_add_fixed_q32_32), (4) Using dynamic shared
  memory (PK_BIND_DYNAMIC_SMEM), (5) Editing gpuni.h or tools/render.c.
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

## Dialect rules (essential)

Read `README.md` section **"Dialect contract (must)"** for the full contract. Key points:

1. Entry: `PK_EXTERN_C __global__ void pk_<name>(...)`
2. Address spaces: every non-private pointer → `__global/__local/__constant`
3. Uniform barriers: all threads must reach `__syncthreads()`
4. C-like only: no templates/classes/refs/exceptions/new/delete
5. Dynamic smem: `__local T* pk_smem` + `PK_BIND_DYNAMIC_SMEM(pk_smem)`

**Detailed rules + examples**: See `references/dialect.md`

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

## References (load as needed)

| Need | Where to look |
|------|---------------|
| Dialect contract, API overview | `README.md` |
| Address space / barrier errors | `references/dialect.md` |
| Exact macro mappings | `gpuni.h` (search: `PK_BACKEND_`, `atomic`) |
| Atomics / fixed-point Q32.32 | `gpuni.h` (search: `pk_atomic_`, `fixed_q32_32`) |
| Render tool options | `tools/render --help` or `tools/render.c` |

## Package files

`gpuni.h`, `tools/render.c`, `README.md`, `skills/gpuni/SKILL.md`
