# gpuni Dialect Diagnosis (AI reference)

Read `README.md` for full dialect rules. This file is for **error diagnosis**.

## Error → Fix mapping

| OpenCL error message | Cause | Fix |
|---------------------|-------|-----|
| "pointer without address space" | Unqualified pointer alias | Add `__global/__local/__constant` |
| "cannot convert `__global T*` to `__private T*`" | Missing address space on alias | Copy address space from source pointer |
| "cannot convert `__local T*` to `__private T*`" | Missing `__local` on shared alias | Use `__local float* t = tile;` |
| Hang / deadlock | Divergent barrier | Ensure all threads reach `__syncthreads()` |
| "undeclared identifier 'sinf'" | Missing gpuni.h include | Add `#include "gpuni.h"` |
| "use of undeclared identifier 'threadIdx'" | Missing gpuni.h or wrong backend | Check include and backend detection |

## Common mistakes checklist

When reviewing `.pk.cu` for OpenCL compatibility:

1. **Pointer aliases** — every `T* p = ...` pointing to `__global/__local/__constant` needs qualifier
2. **Helper function args** — same rule applies to `__device__` helper parameters
3. **Barriers** — no `__syncthreads()` inside `if` that some threads skip
4. **vec3 storage** — no `float3`/`int3` in global arrays or structs

## See also

- Full dialect contract: `README.md` § "Dialect contract (must)"
- Address space examples: `README.md` § "OpenCL 1.2 pointer rule"
- Atomics API: `README.md` § "Atomics (portable baseline)"
