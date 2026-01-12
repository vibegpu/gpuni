# gpuni

A small CUDA-truth kernel dialect for cross-platform many-core GPU compute (CUDA, HIP, OpenCL C 1.2).

**For AI coding (Codex/Claude Code):** load the `gpuni` skill at `skills/gpuni/SKILL.md` (prompt: use `$gpuni`).

**Package:** `gpuni.h` + `tools/render.c`

## Kernel Example

Write `*.gu.cu`:

```cpp
#include "gpuni.h"

GU_EXTERN_C __global__ void gu_saxpy(int n,
                                    GU_GLOBAL float* y,
                                    GU_GLOBAL const float* x,
                                    float a) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) y[i] = a * x[i] + y[i];
}
```

Build:

```bash
cc -O2 -std=c99 -o gpuni-render tools/render.c

nvcc  -I. -c saxpy.gu.cu                      # CUDA
hipcc -I. -c saxpy.gu.cu                      # HIP
./gpuni-render saxpy.gu.cu -o saxpy.cl        # OpenCL
```

## Dialect Rules

1. **Entry:** `GU_EXTERN_C __global__ void gu_<name>(...)`
2. **Address spaces:** annotate every pointer with `GU_GLOBAL/GU_LOCAL/GU_CONSTANT` (including aliases)

   | Keyword | Meaning | Note |
   |---------|---------|------|
   | `__global__` | kernel entry | CUDA native |
   | `GU_GLOBAL` | global memory pointer | no-op in CUDA/HIP |
   | `GU_LOCAL` | local/shared memory pointer | for dynamic shared memory |
   | `GU_CONSTANT` | constant memory pointer | no-op in CUDA/HIP |
   | `__shared__` | shared array declaration | use for `__shared__ float arr[N]` |

   ```cpp
   GU_GLOBAL const float* p = x + off;  // alias must keep GU_GLOBAL

   __shared__ float tile[256];
   GU_LOCAL float* t = tile;            // alias to shared must keep GU_LOCAL
   ```
3. **C subset only:** no templates/classes/overloads/exceptions/new/delete
4. **Uniform barriers:** `__syncthreads()` must be reached by all threads (no divergent barrier)
5. **No `float3` in buffers:** use `float4` or SoA
6. **No warp/subgroup intrinsics:** avoid `__shfl*`, `__ballot*`, cooperative groups; use `__shared__` + `__syncthreads()`

## Types and Helpers

Prefer CUDA/C99 spellings in kernels (e.g. `rsqrtf`, `fmaf`, `atomicAdd`).

```cpp
// Builtins (always available): threadIdx/blockIdx/blockDim/gridDim (.x/.y/.z)

// Portable integer types
gu_i32, gu_u32, gu_i64, gu_u64

// Device helper function
__device__ float my_helper(GU_GLOBAL const float* p) { return p[0] * 2.0f; }

// Optional: double precision (define before include)
#define GU_USE_DOUBLE
#include "gpuni.h"
gu_real x;  // float by default, double if GU_USE_DOUBLE and GU_HAS_FP64
```

## Host API

Enable host API by defining `GUH_CUDA`, `GUH_HIP`, or `GUH_OPENCL` before including `gpuni.h`.

```cpp
#define GUH_CUDA  // or GUH_HIP, GUH_OPENCL
#include "gpuni.h"
#include "saxpy.gu.h"

#if defined(GUH_CUDA) || defined(GUH_HIP)
extern "C" __global__ void gu_saxpy(int, float*, const float*, float);
#endif

int main() {
  gu_ctx ctx; gu_kernel k;
  int n = 1024; float a = 2.0f;
  float *h_x, *h_y;  // host arrays

  gu_ctx_init(&ctx, 0);
  void* d_x = gu_malloc(&ctx, n * sizeof(float));
  void* d_y = gu_malloc(&ctx, n * sizeof(float));
  gu_h2d(&ctx, d_x, h_x, n * sizeof(float));
  gu_h2d(&ctx, d_y, h_y, n * sizeof(float));

  GU_KERNEL(&ctx, &k, gu_saxpy);
  gu_arg(&k, n); gu_arg(&k, d_y); gu_arg(&k, d_x); gu_arg(&k, a);
  gu_run(&ctx, &k, (n + 255) / 256, 256, 0);

  gu_sync(&ctx);
  gu_d2h(&ctx, h_y, d_y, n * sizeof(float));

  gu_kernel_destroy(&k);
  gu_free(&ctx, d_x); gu_free(&ctx, d_y);
  gu_ctx_destroy(&ctx);
}
```

Render for OpenCL: `./gpuni-render saxpy.gu.cu -o saxpy.cl --emit-header saxpy.gu.h`
Host build: `nvcc -DGUH_CUDA host.cu` / `hipcc -DGUH_HIP host.cu` / `cc -DGUH_OPENCL host.c -lOpenCL`

Also: `gu_d2d(&ctx, dst, src, n)` for device-to-device copy.

## Atomics

For portable float accumulation, use fixed-point:

```cpp
// acc is gu_u64* buffer
gu_atomic_add_fixed_q32_32(acc + i, value);

// convert back: gu_fixed_q32_32_to_real(acc[i])
```

Also available: `atomicAdd/atomicCAS/...` (int/uint only), `gu_atomic_add_f32` (float).

## Dynamic Shared Memory

```cpp
GU_EXTERN_C __global__ void gu_reduce(/* ... */, GU_LOCAL float* gu_smem) {
  GU_BIND_DYNAMIC_SMEM(gu_smem);
  GU_LOCAL float* s = gu_smem;
  // ...
}
```

Host: pass `smem_bytes` to `gu_run()`, `NULL` for `gu_smem` argument.
