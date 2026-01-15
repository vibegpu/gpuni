# gpuni Host API

Reference for writing host-side C++ code.

**Namespace:** `using namespace gu;` (recommended)

## Device Management

| API | Description |
|-----|-------------|
| `SetDevice(id)` | Select device (**call first**) |
| `GetDevice()` | Get current device ID |
| `GetDeviceCount()` | Get number of devices |
| `DeviceSync()` | Wait for all operations |

## Memory

| API | Description |
|-----|-------------|
| `Malloc<T>(n)` | Allocate `n` elements on device |
| `Free(p)` | Free device memory |
| `Memset(p, v, bytes)` | Set device memory |
| `MallocHost<T>(n)` | Allocate pinned host memory |
| `FreeHost(p)` | Free pinned host memory |

## Data Transfer

| API | Description |
|-----|-------------|
| `Memcpy(dst, src, bytes, kind)` | Synchronous copy |
| `MemcpyAsync(dst, src, bytes, kind, stream)` | Async copy |

**MemcpyKind:** `H2D`, `D2H`, `D2D`, `H2H`

## Kernel Execution

| API | Description |
|-----|-------------|
| `GetKernel(fn)` | Get kernel handle (auto-cached) |
| `Launch(k, grid, block, args...)` | Launch without smem |
| `Launch(k, grid, block, smem, args...)` | Launch with dynamic smem |
| `Launch(k, grid, block, smem, stream, args...)` | Launch with smem + stream |

**Argument order:** `smem` (size_t) comes before `stream`.

Example:
```cpp
auto k = GetKernel(saxpy);
Launch(k, grid, block, smem_bytes, n, d_y, d_x, a);  // smem before kernel args
```

## Dimensions

| API | Description |
|-----|-------------|
| `dim3(x, y, z)` | 3D grid/block dimensions |
| `int` | 1D grid/block dimensions |

Example:
```cpp
// 1D
Launch(k, 64, 256, ...);

// 2D
dim3 block(16, 16);
dim3 grid((cols + 15) / 16, (rows + 15) / 16);
Launch(k, grid, block, ...);
```

## Streams

| API | Description |
|-----|-------------|
| `stream s;` | Create stream |
| `s.sync()` | Wait for stream |
| `StreamSynchronize(s)` | Wait for stream (functional) |

## Events

| API | Description |
|-----|-------------|
| `event e;` | Create event |
| `e.record(s)` | Record in stream |
| `e.sync()` | Wait for event |
| `EventRecord(e, s)` | Record (functional) |
| `EventSynchronize(e)` | Wait (functional) |
| `ElapsedTime(e1, e2)` | Time between events (ms) |

## Error Handling

| API | Description |
|-----|-------------|
| `GetLastError()` | Get last error code |
| `GetErrorString(e)` | Error code to string |
| `Check(expr)` | Assert success or print error |
| `Success` | Success error code |

## Q32.32 Host Conversion

For `atomicAddFixed` results:

| API | Description |
|-----|-------------|
| `DoubleToFixed(double v)` | Convert double → int64 (before kernel) |
| `FixedToDouble(int64 acc)` | Convert int64 → double (after kernel) |

Example:
```cpp
int64 h_acc = DoubleToFixed(0.0);          // init
Memcpy(d_acc, &h_acc, sizeof(int64), H2D);
Launch(k, grid, block, n, d_in, d_acc);    // kernel uses atomicAddFixed
DeviceSync();
Memcpy(&h_acc, d_acc, sizeof(int64), D2H);
double result = FixedToDouble(h_acc);      // ~9 decimal digits
```

## Build Commands

```bash
# 1. Build render tool (once)
cc -O2 -std=c99 -o gpuni-render tools/render.c

# 2. OpenCL (render + JIT)
./gpuni-render kernel.gu.cu -o kernel.gu.h
c++ -I. host.cpp -lOpenCL

# 3. CUDA (direct)
nvcc -I. host.cpp kernel.gu.cu

# 4. HIP (direct)
hipcc -I. host.cpp kernel.gu.cu
```

## Minimal Host Template

```cpp
#include "gpuni.h"
#include "kernel.gu.h"  // required for OpenCL JIT
using namespace gu;

int main() {
  SetDevice(0);

  float* d_x = Malloc<float>(n);
  float* h_x = MallocHost<float>(n);
  // ... init h_x ...
  Memcpy(d_x, h_x, n * sizeof(float), H2D);

  auto k = GetKernel(kernel_name);
  Launch(k, grid, block, /* smem, */ args...);

  DeviceSync();
  Memcpy(h_x, d_x, n * sizeof(float), D2H);

  Free(d_x);
  FreeHost(h_x);
}
```
