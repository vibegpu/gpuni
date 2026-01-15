---
name: gpuni
description: >-
  Write, refactor, and review gpuni CUDA-truth dialect kernels (*.gu.cu)
  that compile as CUDA/HIP and render to OpenCL C 1.2. Use when working on:
  (1) *.gu.cu kernels, (2) OpenCL 1.2 address spaces (__global/__local/__constant),
  (3) portable atomics (atomic* + atomicAddFloat/MinFloat/MaxFloat + atomicAddFixed), (4) dynamic shared memory (bindSharedMem),
  or (5) portability reviews for CUDA/HIP/OpenCL consistency.
---

# gpuni

Canonical repo: `git@github.com:vibegpu/gpuni.git`

If the gpuni package is not available locally, ask the user to provide it (or clone it if appropriate).

**Source of truth:** `README.md` (or `gpuni/README.md` in dev repo). Do not read `gpuni.h` during normal work.

## Workflow

**Writing/editing kernel (`*.gu.cu`):**
1. Read `references/kernel-api.md` for types, indexing, atomics, smem
2. Apply dialect rules: `extern "C"`, address spaces on all pointers + aliases, uniform barriers
3. If OpenCL fails, check `references/dialect.md` for error → fix mapping

**Writing/editing host code:**
1. Read `references/host-api.md` for memory, launch, streams, events
2. Include `kernel.gu.h` for OpenCL JIT (CUDA/HIP don't need it)

**Need complete patterns?** Read `references/examples.md`

## Review Checklist

- [ ] All pointers have `__global`/`__local`/`__constant`
- [ ] All pointer aliases retain address-space qualifier
- [ ] `__syncthreads()` reachable by all threads (no divergent barriers)
- [ ] Entry uses `extern "C" __global__ void`
- [ ] Dynamic smem param is last + `bindSharedMem()` called
- [ ] No warp intrinsics (`__shfl*`, `__ballot*`)

## References

| File | When to read |
|------|--------------|
| `references/kernel-api.md` | Writing kernel code (`*.gu.cu`) |
| `references/host-api.md` | Writing host code (C++) |
| `references/examples.md` | Need complete kernel+host patterns |
| `references/dialect.md` | OpenCL compilation fails |

## Package

`gpuni.h`, `tools/render.c`, `README.md`, `skills/gpuni/`
