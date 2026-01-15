# gpuni Examples

Core patterns from simple to advanced.

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

## 2. Block Reduction with Shared Memory

```cpp
#include "gpuni.h"

extern "C" __global__ void block_sum(int n,
                                     __global const float* __restrict__ in,
                                     __global float* __restrict__ block_sums,
                                     __local float* smem) {
  bindSharedMem(smem);
  int tid = threadIdx.x;
  int gid = (int)(blockIdx.x * blockDim.x + tid);

  smem[tid] = (gid < n) ? in[gid] : 0.0f;
  __syncthreads();

  // Tree reduction (power-of-2 block size)
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
      atomicAddFloat(&hist[b], vals[i]);
    }
  }
}
```

## 4. 2D Grid Indexing

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

## Common Mistakes

| Pattern | Wrong | Correct |
|---------|-------|---------|
| Pointer alias | `float* p = x;` | `__global float* p = x;` |
| Shared alias | `float* t = smem;` | `__local float* t = smem;` |
| Divergent barrier | `if (cond) __syncthreads();` | Move barrier outside `if` |
| Missing bindSharedMem | forget call | `bindSharedMem(smem);` first line |
