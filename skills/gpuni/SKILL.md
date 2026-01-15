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

## Workflow

1. Read `references/README.md` for dialect rules and API reference
2. Apply: `extern "C"`, address spaces on all pointers + aliases, uniform barriers
3. If need code templates, read `references/examples.md`
4. If OpenCL fails, check `references/dialect.md` for error → fix mapping

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
| `references/README.md` | Dialect rules + API (kernel & host) |
| `references/examples.md` | Need complete code templates |
| `references/dialect.md` | OpenCL compilation fails |

## Package

`gpuni.h`, `tools/render.c`, `README.md`, `skills/gpuni/`
