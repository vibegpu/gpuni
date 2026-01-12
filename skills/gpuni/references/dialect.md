# gpuni Dialect Diagnosis (optional)

This file is **not required** for normal gpuni kernel work.

Use it only when OpenCL C 1.2 compilation/runtime fails, after following the dialect rules in `README.md`.

## Error → Fix mapping

| OpenCL error message | Cause | Fix |
|---------------------|-------|-----|
| "pointer without address space" | Unqualified pointer alias | Add `__global/__local/__constant` |
| "cannot convert `__global T*` to `__private T*`" | Missing address space on alias | Copy address space from source pointer |
| "cannot convert `__local T*` to `__private T*`" | Missing `__local` on shared alias | Use `__local float* t = tile;` |
| Hang / deadlock | Divergent barrier | Ensure all threads reach `__syncthreads()` |
| "undeclared identifier 'sinf'" | Missing dialect mapping | Include `gpuni.h` (maps CUDA spellings to OpenCL) |
| "use of undeclared identifier 'threadIdx'" | Wrong backend context | Ensure OpenCL source is renderer output and includes `gpuni.h` |

## Common mistakes checklist

1. **Pointer aliases:** every alias to global/local/constant must keep the address space qualifier.
2. **Helper args:** apply the same rule to `__device__` helper function parameters.
3. **Uniform barriers:** never put `__syncthreads()` behind divergent control flow.
4. **Vector storage:** avoid `float3`/`int3` in buffers (use `float4` or SoA).

## See also

- Dialect contract and examples: `README.md`
