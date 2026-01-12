---
name: gpuni
description: >-
  Write, refactor, and review gpuni CUDA-truth dialect kernels (*.gu.cu)
  that compile as CUDA/HIP and render to OpenCL C 1.2. Use when working on:
  (1) *.gu.cu kernels, (2) OpenCL 1.2 address spaces (GU_GLOBAL/GU_LOCAL/GU_CONSTANT),
  (3) portable atomics (atomic* + gu_atomic_*), (4) dynamic shared memory (GU_BIND_DYNAMIC_SMEM),
  or (5) portability reviews for CUDA/HIP/OpenCL consistency.
---

# gpuni

Canonical repo: `git@github.com:vibegpu/gpuni.git`

If the gpuni package is not available locally, ask the user to provide it (or clone it if appropriate).

Read the dialect contract first:
- Dev workspace: `gpuni/README.md`
- Released package: `README.md`

Treat the README as the source of truth. Do not read `gpuni.h` or renderer sources during normal kernel work.

## Workflow

1) Identify the kernel(s): `*.gu.cu`.
2) Apply the README "Dialect Rules" (entry signature, address spaces on every pointer + alias, C subset, uniform barriers, no warp/subgroup intrinsics).
3) If OpenCL compilation fails, use `references/dialect.md` to map errors → fixes (usually pointer aliases or divergent barriers).
4) Validate using the repo's smoke/lint scripts (see README "Verification").

## Review Rules

Reject:
- Any change making kernels "not valid CUDA without CUDA-side translation"
- Missing address-space qualifiers on pointer aliases (require explicit annotation)
- Barriers in control flow that some threads may skip

## References

- Dialect contract, API, examples: `README.md` (or `gpuni/README.md` in this dev repo)
- Error diagnosis (AI): `references/dialect.md`

## Package

`gpuni.h`, `tools/render.c`, `README.md`, `skills/gpuni/`
