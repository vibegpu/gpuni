# gpuni Kernel API

Reference for writing `*.gu.cu` kernel code.

## Types

| Type | Description |
|------|-------------|
| `int`, `uint` | 32-bit signed/unsigned |
| `int64`, `uint64` | 64-bit signed/unsigned |
| `float`, `double` | IEEE 754 (double requires device support) |

## Indexing

| API | Description |
|-----|-------------|
| `threadIdx.x/y/z` | Thread index within block |
| `blockIdx.x/y/z` | Block index within grid |
| `blockDim.x/y/z` | Block dimensions |
| `gridDim.x/y/z` | Grid dimensions |

## Address-Space Qualifiers

| Qualifier | Usage |
|-----------|-------|
| `__global` | Device memory pointers |
| `__local` | Shared memory pointers |
| `__constant` | Constant memory pointers |
| `__restrict__` | No-alias hint (optional) |

**Critical:** Every pointer AND alias must have explicit qualifier:
```cpp
__global float* p = output;      // ✓ alias keeps __global
__local float* tile = smem + n;  // ✓ alias keeps __local
float* bad = output;             // ✗ OpenCL error
```

## Synchronization

| API | Description |
|-----|-------------|
| `__syncthreads()` | Block-level barrier (**must be uniform**) |

## Integer Atomics

Pointer type: `__global int*` or `__global uint*`

| API | Operation |
|-----|-----------|
| `atomicAdd(p, v)` | `*p += v`, returns old |
| `atomicSub(p, v)` | `*p -= v`, returns old |
| `atomicExch(p, v)` | `*p = v`, returns old |
| `atomicMin(p, v)` | `*p = min(*p, v)`, returns old |
| `atomicMax(p, v)` | `*p = max(*p, v)`, returns old |
| `atomicCAS(p, cmp, v)` | if `*p == cmp` then `*p = v`, returns old |
| `atomicAnd(p, v)` | `*p &= v`, returns old |
| `atomicOr(p, v)` | `*p |= v`, returns old |
| `atomicXor(p, v)` | `*p ^= v`, returns old |

## Float Atomics

Pointer type: `__global float*`

| API | Operation |
|-----|-----------|
| `atomicAddFloat(p, v)` | `*p += v` |
| `atomicMinFloat(p, v)` | `*p = min(*p, v)` |
| `atomicMaxFloat(p, v)` | `*p = max(*p, v)` |

## Q32.32 Fixed-Point (double precision)

| API | Description |
|-----|-------------|
| `atomicAddFixed(__global int64* acc, double v)` | Accumulate ~9 decimal digits |

Range: ±2^31, precision ~9 digits. See `references/host-api.md` for host conversion.

## Dynamic Shared Memory

| API | Description |
|-----|-------------|
| `bindSharedMem(smem)` | Bind `__local T* smem` (call first in kernel) |

Pattern:
```cpp
extern "C" __global__ void kernel(..., __local float* smem) {  // MUST be last param
  bindSharedMem(smem);  // MUST be first line

  // Multi-array: partition via pointer arithmetic
  __local float* arr1 = smem;
  __local float* arr2 = smem + size1;  // alias keeps __local
}
```

## Math Functions

CUDA-style names work directly:

| Category | Functions |
|----------|-----------|
| Trig | `sinf`, `cosf`, `tanf`, `asinf`, `acosf`, `atanf`, `atan2f` |
| Exp/Log | `expf`, `logf`, `log2f`, `log10f`, `powf` |
| Sqrt | `sqrtf`, `rsqrtf` |
| Round | `floorf`, `ceilf`, `roundf`, `truncf` |
| Misc | `fabsf`, `fminf`, `fmaxf`, `fmaf`, `fmodf` |

## Entry Signature

```cpp
extern "C" __global__ void kernel_name(
    int n,                                    // scalars first
    __global float* __restrict__ output,      // output pointers
    __global const float* __restrict__ input, // input pointers (const)
    float param,                              // more scalars
    __local float* smem)                      // dynamic smem LAST (optional)
{
  bindSharedMem(smem);  // if using smem
  // ...
}
```
