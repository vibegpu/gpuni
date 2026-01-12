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

Canonical repo: `git@github.com:vibegpu/gpuni.git`

Read `README.md` for dialect contract, API reference, and code examples.

## Workflow

1) Locate gpuni package root (contains `gpuni.h` and `tools/render.c`). In this dev repo: `gpuni/`.
2) Task types:
   - New/edited kernel: write CUDA in `*.pk.cu`, apply dialect contract from README.
   - OpenCL build error: check address spaces on pointer aliases + helper args, then barriers.
   - Portability gap: add mapping in `gpuni.h` (prefer CUDA spellings; `pk_*` only when OpenCL 1.2 lacks it).
3) Validate: see README "Verification" section.

## Review Rules

Reject:
- Any change making kernels "not valid CUDA without CUDA-side translation"
- Automatic inference of OpenCL address spaces for pointer aliases (require explicit annotation)
- Barriers in control flow that some threads may skip

## References

| Need | Where |
|------|-------|
| Dialect contract, API, examples | `README.md` |
| Address space / barrier errors | `references/dialect.md` |
| Macro mappings, atomics | `gpuni.h` |
| Render tool | `tools/render --help` |

## Package

`gpuni.h`, `tools/render.c`, `README.md`, `skills/gpuni/SKILL.md`
