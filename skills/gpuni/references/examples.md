# gpuni Examples

Progressive examples from simple to advanced patterns.

## 1. Element-wise (no shared memory)

```cpp
#include "gpuni.h"

extern "C" __global__ void vec_add(int n,
                                   __global float* __restrict__ c,
                                   __global const float* __restrict__ a,
                                   __global const float* __restrict__ b) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) c[i] = a[i] + b[i];
}
```

## 2. Shared Memory Tiling

```cpp
#include "gpuni.h"

#define TILE 256

extern "C" __global__ void smooth(int n,
                                  __global float* __restrict__ out,
                                  __global const float* __restrict__ in,
                                  __local float* smem) {
  bindSharedMem(smem);
  int tid = threadIdx.x;
  int gid = (int)(blockIdx.x * blockDim.x + tid);

  // Load with halo (simplified: clamp boundary)
  int load_idx = (gid > 0) ? gid - 1 : 0;
  smem[tid] = (load_idx < n) ? in[load_idx] : 0.0f;
  __syncthreads();

  if (gid < n && tid > 0 && tid < TILE - 1) {
    out[gid] = 0.25f * smem[tid-1] + 0.5f * smem[tid] + 0.25f * smem[tid+1];
  }
}
```

## 3. Atomic Float Accumulation

```cpp
#include "gpuni.h"

extern "C" __global__ void histogram_float(int n,
                                           __global const float* __restrict__ vals,
                                           __global const int* __restrict__ bins,
                                           __global float* __restrict__ hist,
                                           int num_bins) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    int b = bins[i];
    if (b >= 0 && b < num_bins) {
      atomicAddFloat(&hist[b], vals[i]);  // portable float atomic
    }
  }
}
```

## 4. Block Reduction with Shared Memory

```cpp
#include "gpuni.h"

extern "C" __global__ void block_sum(int n,
                                     __global const float* __restrict__ in,
                                     __global float* __restrict__ block_sums,
                                     __local float* smem) {
  bindSharedMem(smem);
  int tid = threadIdx.x;
  int gid = (int)(blockIdx.x * blockDim.x + tid);

  // Load
  smem[tid] = (gid < n) ? in[gid] : 0.0f;
  __syncthreads();

  // Tree reduction (power-of-2 block size assumed)
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();  // uniform: all threads reach this
  }

  if (tid == 0) {
    block_sums[blockIdx.x] = smem[0];
  }
}
```

## 5. Q32.32 Fixed-Point Accumulation (double precision)

For high-precision reduction without `atomicAdd(double*)`:

```cpp
#include "gpuni.h"

extern "C" __global__ void precise_sum(int n,
                                       __global const double* __restrict__ in,
                                       __global int64* __restrict__ acc) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    atomicAddFixed(acc, in[i]);  // Q32.32 accumulator
  }
}
```

Host:
```cpp
int64 h_acc = DoubleToFixed(0.0);
int64* d_acc = Malloc<int64>(1);
Memcpy(d_acc, &h_acc, sizeof(int64), H2D);
Launch(GetKernel(precise_sum), grid, block, n, d_in, d_acc);
DeviceSync();
Memcpy(&h_acc, d_acc, sizeof(int64), D2H);
double result = FixedToDouble(h_acc);  // ~9 decimal digits precision
```

## 6. 2D Grid Indexing

```cpp
#include "gpuni.h"

extern "C" __global__ void transpose(int rows, int cols,
                                     __global float* __restrict__ out,
                                     __global const float* __restrict__ in) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);

  if (x < cols && y < rows) {
    out[x * rows + y] = in[y * cols + x];
  }
}
```

Host:
```cpp
dim3 block(16, 16);
dim3 grid((cols + 15) / 16, (rows + 15) / 16);
Launch(GetKernel(transpose), grid, block, rows, cols, d_out, d_in);
```

## 7. Multiple Shared Arrays

```cpp
#include "gpuni.h"

extern "C" __global__ void dual_buffer(int n,
                                       __global float* __restrict__ out,
                                       __global const float* __restrict__ a,
                                       __global const float* __restrict__ b,
                                       __local float* smem) {
  bindSharedMem(smem);
  int tid = threadIdx.x;
  int gid = (int)(blockIdx.x * blockDim.x + tid);

  // Split shared memory manually
  __local float* sa = smem;                    // first half
  __local float* sb = smem + blockDim.x;      // second half (alias keeps __local)

  sa[tid] = (gid < n) ? a[gid] : 0.0f;
  sb[tid] = (gid < n) ? b[gid] : 0.0f;
  __syncthreads();

  if (gid < n) {
    out[gid] = sa[tid] * sb[tid];
  }
}
// Host: smem_bytes = 2 * block * sizeof(float)
```

## Common Mistakes

| Pattern | Wrong | Correct |
|---------|-------|---------|
| Pointer alias | `float* p = x;` | `__global float* p = x;` |
| Shared alias | `float* t = smem;` | `__local float* t = smem;` |
| Divergent barrier | `if (cond) __syncthreads();` | Move barrier outside `if` |
| Missing bindSharedMem | forget call | `bindSharedMem(smem);` first line |
