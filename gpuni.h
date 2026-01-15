#ifndef GPUNI_H
#define GPUNI_H

/* gpuni CUDA-truth kernel dialect.
 *
 * Naming:
 * - Kernel API: camelCase (`atomicAddFloat`, `atomicAddFixed`, `bindSharedMem`)
 * - Host API: PascalCase (`SetDevice`, `GetKernel`, `Launch`, `Check`)
 * - Internal: underscore prefix (`_atomicAddF32Impl`, `_gpuni_smem_`)
 */

#define GU_DIALECT_VERSION 1

#if defined(__OPENCL_VERSION__) || defined(__OPENCL_C_VERSION__)
#  define GU_BACKEND_OPENCL 1
#elif defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#  define GU_BACKEND_HIP 1
#elif defined(__CUDACC__)
#  define GU_BACKEND_CUDA 1
#else
#  define GU_BACKEND_HOST 1
#endif

#if defined(GU_BACKEND_OPENCL)
#  if defined(cl_khr_fp64) || defined(cl_amd_fp64)
#    define GU_HAS_FP64 1
#  else
#    define GU_HAS_FP64 0
#  endif
#  if defined(cl_khr_int64_base_atomics) && !defined(GU_DISABLE_OPENCL_INT64_ATOMICS)
#    define GU_HAS_I64_ATOMICS 1
#  else
#    define GU_HAS_I64_ATOMICS 0
#  endif
#  if defined(cl_khr_local_int32_base_atomics) || defined(cl_khr_local_int32_extended_atomics)
#    define GU_HAS_LOCAL_ATOMICS 1
#  else
#    define GU_HAS_LOCAL_ATOMICS 0
#  endif
#else
#  define GU_HAS_FP64 1
#  define GU_HAS_I64_ATOMICS 1
#  define GU_HAS_LOCAL_ATOMICS 1
#endif

#if defined(GU_BACKEND_OPENCL)
#  ifdef cl_khr_fp64
#    pragma OPENCL EXTENSION cl_khr_fp64 : enable
#  endif
#  ifdef cl_amd_fp64
#    pragma OPENCL EXTENSION cl_amd_fp64 : enable
#  endif
#endif

#if defined(GU_BACKEND_OPENCL)
/* OpenCL C: int/uint/long/ulong built-in */
typedef long int64;
typedef ulong uint64;
#else
/* CUDA/HIP/Host: short aliases */
typedef unsigned int uint;
typedef long long int64;
typedef unsigned long long uint64;
#endif

/* Types int, uint, int64, uint64 are used directly (no aliases needed) */

#define GU_FIXED_Q32_32_SCALE_F 4294967296.0f
#define GU_FIXED_Q32_32_INV_SCALE_F 2.3283064365386963e-10f /* 2^-32 */
#define GU_FIXED_Q32_32_SCALE_D 4294967296.0
#define GU_FIXED_Q32_32_INV_SCALE_D 2.3283064365386963e-10 /* 2^-32 */

#if defined(GU_BACKEND_HIP)
#  include <hip/hip_runtime.h>
#endif

#if defined(GU_BACKEND_OPENCL)

#  if defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ < 120
#    error "gpuni requires OpenCL C 1.2+"
#  elif defined(__OPENCL_VERSION__) && __OPENCL_VERSION__ < 120
#    error "gpuni requires OpenCL C 1.2+"
#  endif

/* Enable atomics extensions when available (OpenCL 1.2 core still works). */
#  ifdef cl_khr_global_int32_base_atomics
#    pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#  endif
#  ifdef cl_khr_local_int32_base_atomics
#    pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#  endif
#  ifdef cl_khr_global_int32_extended_atomics
#    pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#  endif
#  ifdef cl_khr_local_int32_extended_atomics
#    pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#  endif
#  ifdef cl_khr_int64_base_atomics
#    pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#  endif
#  ifdef cl_khr_int64_extended_atomics
#    pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#  endif

#  define __host__
#  define __device__
#  define __global__ __kernel
#  define __shared__ __local
#  define __shared __local
#  define __constant__ __constant
#  define __launch_bounds__(t, b)

#  define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)

#  define threadIdx ((uint3)(get_local_id(0), get_local_id(1), get_local_id(2)))
#  define blockIdx  ((uint3)(get_group_id(0), get_group_id(1), get_group_id(2)))
#  define blockDim  ((uint3)(get_local_size(0), get_local_size(1), get_local_size(2)))
#  define gridDim   ((uint3)(get_num_groups(0), get_num_groups(1), get_num_groups(2)))

#  define GU_GLOBAL __global
#  define GU_LOCAL __local
#  define GU_CONSTANT __constant

/* Pointer helper macros (address space only; add `const` in `T` when needed). */
#  define GU_GLOBAL_PTR(T) GU_GLOBAL T*
#  define GU_LOCAL_PTR(T) GU_LOCAL T*
#  define GU_CONSTANT_PTR(T) GU_CONSTANT T*

/* Map CUDA __restrict__ to OpenCL restrict */
#  define __restrict__ restrict

#  define GU_INLINE inline

/* Dynamic shared memory: OpenCL uses kernel param, no binding needed */
#  define bindSharedMem(ptr) /* no-op */

/* CUDA/C99 float math aliases for OpenCL (so kernels can stay CUDA-like). */
#  ifndef rsqrtf
#    define rsqrtf(x) rsqrt((float)(x))
#  endif
#  ifndef sqrtf
#    define sqrtf(x) sqrt((float)(x))
#  endif
#  ifndef fabsf
#    define fabsf(x) fabs((float)(x))
#  endif
#  ifndef fminf
#    define fminf(x, y) fmin((float)(x), (float)(y))
#  endif
#  ifndef fmaxf
#    define fmaxf(x, y) fmax((float)(x), (float)(y))
#  endif
#  ifndef fmaf
#    define fmaf(a, b, c) fma((float)(a), (float)(b), (float)(c))
#  endif
#  ifndef sinf
#    define sinf(x) sin((float)(x))
#  endif
#  ifndef cosf
#    define cosf(x) cos((float)(x))
#  endif
#  ifndef tanf
#    define tanf(x) tan((float)(x))
#  endif
#  ifndef asinf
#    define asinf(x) asin((float)(x))
#  endif
#  ifndef acosf
#    define acosf(x) acos((float)(x))
#  endif
#  ifndef atanf
#    define atanf(x) atan((float)(x))
#  endif
#  ifndef atan2f
#    define atan2f(y, x) atan2((float)(y), (float)(x))
#  endif
#  ifndef sinhf
#    define sinhf(x) sinh((float)(x))
#  endif
#  ifndef coshf
#    define coshf(x) cosh((float)(x))
#  endif
#  ifndef tanhf
#    define tanhf(x) tanh((float)(x))
#  endif
#  ifndef expf
#    define expf(x) exp((float)(x))
#  endif
#  ifndef exp2f
#    define exp2f(x) exp2((float)(x))
#  endif
#  ifndef logf
#    define logf(x) log((float)(x))
#  endif
#  ifndef log2f
#    define log2f(x) log2((float)(x))
#  endif
#  ifndef log10f
#    define log10f(x) log10((float)(x))
#  endif
#  ifndef powf
#    define powf(x, y) pow((float)(x), (float)(y))
#  endif
#  ifndef floorf
#    define floorf(x) floor((float)(x))
#  endif
#  ifndef ceilf
#    define ceilf(x) ceil((float)(x))
#  endif
#  ifndef truncf
#    define truncf(x) trunc((float)(x))
#  endif
#  ifndef roundf
#    define roundf(x) round((float)(x))
#  endif
#  ifndef fmodf
#    define fmodf(x, y) fmod((float)(x), (float)(y))
#  endif
#  ifndef copysignf
#    define copysignf(x, y) copysign((float)(x), (float)(y))
#  endif
#  ifndef hypotf
#    define hypotf(x, y) hypot((float)(x), (float)(y))
#  endif
#  ifndef cbrtf
#    define cbrtf(x) cbrt((float)(x))
#  endif
#  ifndef erff
#    define erff(x) erf((float)(x))
#  endif
#  ifndef erfcf
#    define erfcf(x) erfc((float)(x))
#  endif

/* CUDA-style 32-bit atomics (int/uint only) */
#  ifndef atomicAdd
#    define atomicAdd atomic_add
#  endif
#  ifndef atomicSub
#    define atomicSub atomic_sub
#  endif
#  ifndef atomicExch
#    define atomicExch atomic_xchg
#  endif
#  ifndef atomicMin
#    define atomicMin atomic_min
#  endif
#  ifndef atomicMax
#    define atomicMax atomic_max
#  endif
#  ifndef atomicAnd
#    define atomicAnd atomic_and
#  endif
#  ifndef atomicOr
#    define atomicOr atomic_or
#  endif
#  ifndef atomicXor
#    define atomicXor atomic_xor
#  endif
#  ifndef atomicCAS
#    define atomicCAS atomic_cmpxchg
#  endif

/* Atomics (OpenCL C 1.2 legacy atomics + optional int64 extensions)
   Notes:
   - Use int32 atomics for counters/indices.
   - Use fixed-point(Q32.32)+u64 atomic add for portable float accumulation.
   - If cl_khr_int64_base_atomics is unavailable, u64 add falls back to 2x u32
     atomics + carry (correct for accumulation; not a full 64-bit RMW API). */

static GU_INLINE void _atomicAddU64Impl(GU_GLOBAL uint64* p, uint64 val) {
#  if defined(cl_khr_int64_base_atomics) && !defined(GU_DISABLE_OPENCL_INT64_ATOMICS)
  (void)atom_add((volatile __global ulong*)p, (ulong)val);
#  else
  volatile __global uint* word = (volatile __global uint*)p;
#    ifdef __ENDIAN_LITTLE__
  const int low = 0;
#    else
  const int low = 1;
#    endif
  const uint lower = (uint)val;
  uint upper = (uint)(val >> 32);
  /* Fast-path: when the low-word increment is zero, there is no carry. */
  if (lower == 0u) {
    if (upper != 0u) atomic_add(&word[1 - low], upper);
    return;
  }
  const uint old_lower = atomic_add(&word[low], lower);
  const uint sum = old_lower + lower;
  upper += (sum < old_lower) ? 1u : 0u;
  if (upper != 0u) atomic_add(&word[1 - low], upper);
#  endif
}

/* Q32.32 fixed-point accumulator: high-throughput atomic add without CAS contention.
   Use for summing many values; convert result back to double after kernel completes.
   Precision: 32-bit integer + 32-bit fraction (~9 decimal digits). */
static GU_INLINE void atomicAddFixed(GU_GLOBAL int64* p, double x) {
  _atomicAddU64Impl((GU_GLOBAL uint64*)p, (uint64)(int64)(x * GU_FIXED_Q32_32_SCALE_D));
}

static GU_INLINE int64 doubleToFixed(double x) {
  return (int64)(x * GU_FIXED_Q32_32_SCALE_D);
}

static GU_INLINE double fixedToDouble(int64 x) {
  return (double)x * GU_FIXED_Q32_32_INV_SCALE_D;
}

/* Float atomic add (OpenCL 1.2 has no atomic_add(float); emulate via CAS on u32 bits).
   Correctness-first; prefer fixed-point(Q32.32) for high-throughput accumulation. */
static GU_INLINE float _atomicAddF32Impl(GU_GLOBAL float* p, float x) {
  volatile GU_GLOBAL uint* u = (volatile GU_GLOBAL uint*)p;
  uint old = atomic_add(u, (uint)0);
  for (;;) {
    uint assumed = old;
    uint desired = as_uint(as_float(assumed) + x);
    old = atomic_cmpxchg(u, assumed, desired);
    if (old == assumed) return as_float(assumed);
  }
}

/* Unified atomics for the gpuni dialect.
   Notes:
   - OpenCL C 1.2 cannot overload by type for user-defined functions; keep names explicit.
   - Float min/max/add use CAS on u32 bits for OpenCL portability. */
static GU_INLINE float _atomicMinF32Impl(GU_GLOBAL float* p, float x) {
  volatile GU_GLOBAL uint* u = (volatile GU_GLOBAL uint*)p;
  uint old = atomicAdd((volatile GU_GLOBAL uint*)u, (uint)0);
  for (;;) {
    uint assumed = old;
    float assumed_f = as_float(assumed);
    float desired_f = fminf(assumed_f, x);
    uint desired = as_uint(desired_f);
    old = atomicCAS((volatile GU_GLOBAL uint*)u, assumed, desired);
    if (old == assumed) return assumed_f;
  }
}

static GU_INLINE float _atomicMaxF32Impl(GU_GLOBAL float* p, float x) {
  volatile GU_GLOBAL uint* u = (volatile GU_GLOBAL uint*)p;
  uint old = atomicAdd((volatile GU_GLOBAL uint*)u, (uint)0);
  for (;;) {
    uint assumed = old;
    float assumed_f = as_float(assumed);
    float desired_f = fmaxf(assumed_f, x);
    uint desired = as_uint(desired_f);
    old = atomicCAS((volatile GU_GLOBAL uint*)u, assumed, desired);
    if (old == assumed) return assumed_f;
  }
}

/* Float atomic aliases */
#define atomicAddFloat _atomicAddF32Impl
#define atomicMinFloat _atomicMinF32Impl
#define atomicMaxFloat _atomicMaxF32Impl

/* OpenCL is C, no extern "C" needed */
#  define EXTERN_C

#else

/* OpenCL address-space keywords (used in dialect for pointer types).
   In CUDA/HIP/host they are no-ops; in OpenCL they are language keywords.
   Note: HIP toolchains may predefine __global/__local/__constant as addrspace
   attributes; we explicitly neutralize them here for CUDA-truth sources. */
#  ifdef __global
#    undef __global
#  endif
#  define __global
#  ifdef __local
#    undef __local
#  endif
#  define __local
#  ifdef __constant
#    undef __constant
#  endif
#  define __constant
#  define __shared __shared__

#  define GU_GLOBAL
#  define GU_LOCAL
#  define GU_CONSTANT

/* Pointer helper macros (address space only; add `const` in `T` when needed). */
#  define GU_GLOBAL_PTR(T) T*
#  define GU_LOCAL_PTR(T) T*
#  define GU_CONSTANT_PTR(T) T*

/* __restrict__ is native in CUDA/HIP/GCC/Clang; MSVC uses __restrict */
#  if defined(_MSC_VER)
#    define __restrict__ __restrict
#  endif

#  if defined(GU_BACKEND_CUDA) || defined(GU_BACKEND_HIP)
#    define GU_INLINE __forceinline__
#  else
#    define GU_INLINE inline
#  endif

/* Dynamic shared memory: CUDA/HIP binds to extern __shared__ */
#  define bindSharedMem(ptr) \
     extern __shared__ unsigned char _gpuni_smem_[]; \
     (ptr) = (decltype(ptr))(&_gpuni_smem_[0])

#  if defined(GU_BACKEND_CUDA) || defined(GU_BACKEND_HIP)
static __device__ GU_INLINE void _atomicAddU64Impl(GU_GLOBAL uint64* p, uint64 val) {
  (void)atomicAdd((unsigned long long*)p, (unsigned long long)val);
}

static __device__ GU_INLINE float _atomicAddF32Impl(GU_GLOBAL float* p, float x) {
  return atomicAdd((float*)p, x);
}

static __device__ GU_INLINE uint _bitcastU32FromF32(float x) {
  union {
    float f;
    uint u;
  } v;
  v.f = x;
  return v.u;
}

static __device__ GU_INLINE float _bitcastF32FromU32(uint x) {
  union {
    float f;
    uint u;
  } v;
  v.u = x;
  return v.f;
}

static __device__ GU_INLINE float _atomicMinF32Impl(GU_GLOBAL float* p, float x) {
  uint* u = (uint*)p;
  uint old = atomicCAS((unsigned int*)u, 0u, 0u);
  for (;;) {
    uint assumed = old;
    float assumed_f = _bitcastF32FromU32(assumed);
    float desired_f = fminf(assumed_f, x);
    uint desired = _bitcastU32FromF32(desired_f);
    old = atomicCAS((unsigned int*)u, (unsigned int)assumed, (unsigned int)desired);
    if (old == assumed) return assumed_f;
  }
}

static __device__ GU_INLINE float _atomicMaxF32Impl(GU_GLOBAL float* p, float x) {
  uint* u = (uint*)p;
  uint old = atomicCAS((unsigned int*)u, 0u, 0u);
  for (;;) {
    uint assumed = old;
    float assumed_f = _bitcastF32FromU32(assumed);
    float desired_f = fmaxf(assumed_f, x);
    uint desired = _bitcastU32FromF32(desired_f);
    old = atomicCAS((unsigned int*)u, (unsigned int)assumed, (unsigned int)desired);
    if (old == assumed) return assumed_f;
  }
}

/* Float atomic aliases */
#define atomicAddFloat _atomicAddF32Impl
#define atomicMinFloat _atomicMinF32Impl
#define atomicMaxFloat _atomicMaxF32Impl

/* Q32.32 fixed-point accumulator: high-throughput atomic add without CAS contention.
   Use for summing many values; convert result back to double after kernel completes.
   Precision: 32-bit integer + 32-bit fraction (~9 decimal digits). */
static __device__ GU_INLINE void atomicAddFixed(GU_GLOBAL int64* p, double x) {
  _atomicAddU64Impl((GU_GLOBAL uint64*)p, (uint64)(int64)(x * GU_FIXED_Q32_32_SCALE_D));
}

static __device__ GU_INLINE int64 doubleToFixed(double x) {
  return (int64)(x * GU_FIXED_Q32_32_SCALE_D);
}

static __device__ GU_INLINE double fixedToDouble(int64 x) {
  return (double)x * GU_FIXED_Q32_32_INV_SCALE_D;
}
#  endif

/* Prevent C++ name mangling for kernel symbols */
#  define EXTERN_C extern "C"

#endif

/* ============================================================
 * Host API (enabled by GUH_CUDA / GUH_HIP / GUH_OPENCL)
 * ============================================================ */

/* Auto-detect backend if not explicitly set.
 * Priority: CUDA (nvcc) > HIP (hipcc) > OpenCL (fallback for plain C/C++).
 * Skip auto-detection when compiling OpenCL kernels (GU_BACKEND_OPENCL is set). */
#if !defined(GUH_CUDA) && !defined(GUH_HIP) && !defined(GUH_OPENCL) && !defined(GU_BACKEND_OPENCL)
#  if defined(__CUDACC__) || defined(CUDA_VERSION)
#    define GUH_CUDA 1
#  elif defined(__HIPCC__)
#    define GUH_HIP 1
#  else
#    define GUH_OPENCL 1
#  endif
#endif

#if defined(GUH_CUDA) || defined(GUH_HIP) || defined(GUH_OPENCL)

#include <stddef.h>

#if defined(GUH_CUDA)
#  include <cuda_runtime.h>
#elif defined(GUH_HIP)
#  include <hip/hip_runtime.h>
#elif defined(GUH_OPENCL)
#  ifdef __APPLE__
#    include <OpenCL/cl.h>
#  else
#    include <CL/cl.h>
#  endif
#  include <stdio.h>
#  include <stdlib.h>
#else
#  include <stdlib.h>
#  include <string.h>
#endif

#ifndef GUH_MAX_ARGS
#  define GUH_MAX_ARGS 24
#endif

#ifndef GUH_OPENCL_BUILD_OPTIONS
#  define GUH_OPENCL_BUILD_OPTIONS "-cl-std=CL1.2"
#endif

#endif /* GUH_CUDA || GUH_HIP || GUH_OPENCL */

/* ============================================================
 * C++ Level 1 API (namespace gu)
 * cudaXxx -> gu::Xxx (CUDA-style, cross-platform)
 * ============================================================ */
#if defined(__cplusplus) && (defined(GUH_CUDA) || defined(GUH_HIP) || defined(GUH_OPENCL))

#include <cstring>
#include <cstdio>
#if defined(GUH_OPENCL)
#include <unordered_map>
#include <vector>
#include <utility>
#endif

/* Unify CUDA/HIP API names */
#if defined(GUH_CUDA) || defined(GUH_HIP)
#  define GUH_NATIVE 1
#  if defined(GUH_CUDA)
#    define GU_API(name) cuda##name
#    define GU_API_T(name) cuda##name##_t
#    define GU_MCPY(k) cudaMemcpy##k
#  else
#    define GU_API(name) hip##name
#    define GU_API_T(name) hip##name##_t
#    define GU_MCPY(k) hipMemcpy##k
#  endif
#endif

namespace gu {

using Error_t = int;
constexpr Error_t Success = 0;

/* Q32.32 fixed-point conversion (host-side). */
inline int64 DoubleToFixed(double x) { return (int64)(x * GU_FIXED_Q32_32_SCALE_D); }
inline double FixedToDouble(int64 x) { return (double)x * GU_FIXED_Q32_32_INV_SCALE_D; }

namespace detail {
#if defined(GUH_NATIVE)
  static Error_t g_last_error = 0;
#elif defined(GUH_OPENCL)
  static Error_t g_last_error = CL_SUCCESS;
  static cl_context g_context = nullptr;
  static cl_command_queue g_queue = nullptr;
  static cl_device_id g_device = nullptr;
  static int g_current_device = -1;
  static std::vector<std::pair<cl_platform_id, cl_device_id>> g_all_devices;
  static std::unordered_map<void*, cl_mem> g_pinned_map;

  static inline void init_device_list() {
    if (!g_all_devices.empty()) return;
    cl_uint np = 0;
    clGetPlatformIDs(0, nullptr, &np);
    if (np == 0) return;
    std::vector<cl_platform_id> plats(np);
    clGetPlatformIDs(np, plats.data(), nullptr);
    for (auto& p : plats) {
      cl_uint nd = 0;
      if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) == CL_SUCCESS && nd > 0) {
        std::vector<cl_device_id> devs(nd);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
        for (auto& d : devs) g_all_devices.push_back({p, d});
      }
    }
    if (g_all_devices.empty()) {
      for (auto& p : plats) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) == CL_SUCCESS && nd > 0) {
          std::vector<cl_device_id> devs(nd);
          clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
          for (auto& d : devs) g_all_devices.push_back({p, d});
        }
      }
    }
  }
#endif
} // namespace detail

static inline Error_t GetLastError() {
#if defined(GUH_NATIVE)
  Error_t e = (Error_t)GU_API(GetLastError)();
  if (e != 0) detail::g_last_error = e;
#endif
  return detail::g_last_error;
}

static inline const char* GetErrorString(Error_t e) {
#if defined(GUH_NATIVE)
  return GU_API(GetErrorString)((GU_API_T(Error))e);
#elif defined(GUH_OPENCL)
  switch (e) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    default: return "CL_UNKNOWN_ERROR";
  }
#else
  (void)e; return "Unknown";
#endif
}

#define Check(expr) do { \
  gu::Error_t _e = (expr); \
  if (_e != gu::Success) fprintf(stderr, "gpuni error %d: %s at %s:%d\n", _e, gu::GetErrorString(_e), __FILE__, __LINE__); \
} while(0)

/* ---- Device ---- */
static inline int GetDeviceCount() {
#if defined(GUH_NATIVE)
  int n = 0; detail::g_last_error = (Error_t)GU_API(GetDeviceCount)(&n); return n;
#elif defined(GUH_OPENCL)
  detail::init_device_list();
  return (int)detail::g_all_devices.size();
#else
  return 1;
#endif
}

static inline void SetDevice(int id) {
#if defined(GUH_NATIVE)
  detail::g_last_error = (Error_t)GU_API(SetDevice)(id);
#elif defined(GUH_OPENCL)
  detail::init_device_list();
  if (id < 0 || id >= (int)detail::g_all_devices.size()) { detail::g_last_error = CL_INVALID_DEVICE; return; }
  if (detail::g_current_device == id) return;
  if (detail::g_queue) { clFinish(detail::g_queue); clReleaseCommandQueue(detail::g_queue); }
  if (detail::g_context) clReleaseContext(detail::g_context);
  cl_device_id dev = detail::g_all_devices[id].second;
  cl_int e;
  detail::g_device = dev;
  detail::g_context = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
  if (e != CL_SUCCESS) { detail::g_last_error = e; return; }
  detail::g_queue = clCreateCommandQueue(detail::g_context, dev, CL_QUEUE_PROFILING_ENABLE, &e);
  detail::g_last_error = e;
  detail::g_current_device = id;
#endif
}

static inline int GetDevice() {
#if defined(GUH_NATIVE)
  int id = 0; detail::g_last_error = (Error_t)GU_API(GetDevice)(&id); return id;
#elif defined(GUH_OPENCL)
  return detail::g_current_device;
#else
  return 0;
#endif
}

static inline void DeviceSync() {
#if defined(GUH_NATIVE)
  detail::g_last_error = (Error_t)GU_API(DeviceSynchronize)();
#elif defined(GUH_OPENCL)
  if (detail::g_queue) detail::g_last_error = clFinish(detail::g_queue);
#endif
}

/* ---- Memory ---- */
template<typename T> static inline T* Malloc(size_t count) {
  T* p = nullptr;
#if defined(GUH_NATIVE)
  detail::g_last_error = (Error_t)GU_API(Malloc)(&p, count * sizeof(T));
#elif defined(GUH_OPENCL)
  cl_int e;
  p = (T*)clCreateBuffer(detail::g_context, CL_MEM_READ_WRITE, count * sizeof(T), nullptr, &e);
  detail::g_last_error = e;
#else
  p = (T*)malloc(count * sizeof(T));
#endif
  return p;
}

static inline void Free(void* p) {
#if defined(GUH_NATIVE)
  detail::g_last_error = (Error_t)GU_API(Free)(p);
#elif defined(GUH_OPENCL)
  if (p) clReleaseMemObject((cl_mem)p);
#else
  free(p);
#endif
}

static inline void Memset(void* p, int val, size_t bytes) {
#if defined(GUH_NATIVE)
  detail::g_last_error = (Error_t)GU_API(Memset)(p, val, bytes);
#elif defined(GUH_OPENCL)
  cl_uchar pattern = (cl_uchar)val;
  detail::g_last_error = clEnqueueFillBuffer(detail::g_queue, (cl_mem)p, &pattern, 1, 0, bytes, 0, nullptr, nullptr);
#else
  memset(p, val, bytes);
#endif
}

template<typename T> static inline T* MallocHost(size_t count) {
  T* p = nullptr;
#if defined(GUH_CUDA)
  detail::g_last_error = (Error_t)cudaMallocHost(&p, count * sizeof(T));
#elif defined(GUH_HIP)
  detail::g_last_error = (Error_t)hipHostMalloc(&p, count * sizeof(T));
#elif defined(GUH_OPENCL)
  size_t bytes = count * sizeof(T);
  cl_int e;
  cl_mem buf = clCreateBuffer(detail::g_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, &e);
  if (e != CL_SUCCESS) { detail::g_last_error = e; return nullptr; }
  p = (T*)clEnqueueMapBuffer(detail::g_queue, buf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, bytes, 0, nullptr, nullptr, &e);
  if (e != CL_SUCCESS) { clReleaseMemObject(buf); detail::g_last_error = e; return nullptr; }
  detail::g_pinned_map[p] = buf;
#else
  p = (T*)malloc(count * sizeof(T));
#endif
  return p;
}

static inline void FreeHost(void* p) {
#if defined(GUH_CUDA)
  detail::g_last_error = (Error_t)cudaFreeHost(p);
#elif defined(GUH_HIP)
  detail::g_last_error = (Error_t)hipHostFree(p);
#elif defined(GUH_OPENCL)
  auto it = detail::g_pinned_map.find(p);
  if (it != detail::g_pinned_map.end()) {
    clEnqueueUnmapMemObject(detail::g_queue, it->second, p, 0, nullptr, nullptr);
    clReleaseMemObject(it->second);
    detail::g_pinned_map.erase(it);
  }
#else
  free(p);
#endif
}

/* ---- Memcpy ---- */
enum MemcpyKind {
  H2D = 1, D2H = 2, D2D = 3, H2H = 4,
  MemcpyHostToDevice = 1, MemcpyDeviceToHost = 2,
  MemcpyDeviceToDevice = 3, MemcpyHostToHost = 4
};

static inline void Memcpy(void* dst, const void* src, size_t bytes, MemcpyKind kind) {
#if defined(GUH_NATIVE)
  auto ck = (kind == H2D) ? GU_MCPY(HostToDevice) : (kind == D2H) ? GU_MCPY(DeviceToHost) :
            (kind == D2D) ? GU_MCPY(DeviceToDevice) : GU_MCPY(HostToHost);
  detail::g_last_error = (Error_t)GU_API(Memcpy)(dst, src, bytes, ck);
#elif defined(GUH_OPENCL)
  switch (kind) {
    case H2D: detail::g_last_error = clEnqueueWriteBuffer(detail::g_queue, (cl_mem)dst, CL_TRUE, 0, bytes, src, 0, nullptr, nullptr); break;
    case D2H: detail::g_last_error = clEnqueueReadBuffer(detail::g_queue, (cl_mem)src, CL_TRUE, 0, bytes, dst, 0, nullptr, nullptr); break;
    case D2D: detail::g_last_error = clEnqueueCopyBuffer(detail::g_queue, (cl_mem)src, (cl_mem)dst, 0, 0, bytes, 0, nullptr, nullptr); clFinish(detail::g_queue); break;
    case H2H: std::memcpy(dst, src, bytes); break;
  }
#else
  std::memcpy(dst, src, bytes);
#endif
}

/* ---- stream ---- */
class stream {
#if defined(GUH_NATIVE)
  GU_API_T(Stream) s_ = nullptr;
#elif defined(GUH_OPENCL)
  cl_command_queue q_ = nullptr;
#endif
public:
  stream() {
#if defined(GUH_NATIVE)
    detail::g_last_error = (Error_t)GU_API(StreamCreate)(&s_);
#elif defined(GUH_OPENCL)
    cl_int e;
    q_ = clCreateCommandQueue(detail::g_context, detail::g_device, CL_QUEUE_PROFILING_ENABLE, &e);
    detail::g_last_error = e;
#endif
  }
  ~stream() {
#if defined(GUH_NATIVE)
    if (s_) detail::g_last_error = (Error_t)GU_API(StreamDestroy)(s_);
#elif defined(GUH_OPENCL)
    if (q_) clReleaseCommandQueue(q_);
#endif
  }
  stream(const stream&) = delete;
  stream& operator=(const stream&) = delete;
  void sync() {
#if defined(GUH_NATIVE)
    detail::g_last_error = (Error_t)GU_API(StreamSynchronize)(s_);
#elif defined(GUH_OPENCL)
    if (q_) detail::g_last_error = clFinish(q_);
#endif
  }
#if defined(GUH_NATIVE)
  GU_API_T(Stream) native() const { return s_; }
#elif defined(GUH_OPENCL)
  cl_command_queue native() const { return q_; }
#endif
};

static inline void StreamSynchronize(stream& s) { s.sync(); }

static inline void MemcpyAsync(void* dst, const void* src, size_t bytes, MemcpyKind kind, stream& s) {
#if defined(GUH_NATIVE)
  auto ck = (kind == H2D) ? GU_MCPY(HostToDevice) : (kind == D2H) ? GU_MCPY(DeviceToHost) :
            (kind == D2D) ? GU_MCPY(DeviceToDevice) : GU_MCPY(HostToHost);
  detail::g_last_error = (Error_t)GU_API(MemcpyAsync)(dst, src, bytes, ck, s.native());
#elif defined(GUH_OPENCL)
  switch (kind) {
    case H2D: detail::g_last_error = clEnqueueWriteBuffer(s.native(), (cl_mem)dst, CL_FALSE, 0, bytes, src, 0, nullptr, nullptr); break;
    case D2H: detail::g_last_error = clEnqueueReadBuffer(s.native(), (cl_mem)src, CL_FALSE, 0, bytes, dst, 0, nullptr, nullptr); break;
    case D2D: detail::g_last_error = clEnqueueCopyBuffer(s.native(), (cl_mem)src, (cl_mem)dst, 0, 0, bytes, 0, nullptr, nullptr); break;
    case H2H: std::memcpy(dst, src, bytes); break;
  }
#else
  (void)s; std::memcpy(dst, src, bytes);
#endif
}

/* ---- event ---- */
class event {
#if defined(GUH_NATIVE)
  GU_API_T(Event) e_ = nullptr;
#elif defined(GUH_OPENCL)
  cl_event e_ = nullptr;
#endif
public:
  event() {
#if defined(GUH_NATIVE)
    detail::g_last_error = (Error_t)GU_API(EventCreate)(&e_);
#endif
  }
  ~event() {
#if defined(GUH_NATIVE)
    if (e_) detail::g_last_error = (Error_t)GU_API(EventDestroy)(e_);
#elif defined(GUH_OPENCL)
    if (e_) clReleaseEvent(e_);
#endif
  }
  event(const event&) = delete;
  event& operator=(const event&) = delete;
  void record(stream& s) {
#if defined(GUH_NATIVE)
    detail::g_last_error = (Error_t)GU_API(EventRecord)(e_, s.native());
#elif defined(GUH_OPENCL)
    if (e_) clReleaseEvent(e_);
    detail::g_last_error = clEnqueueMarkerWithWaitList(s.native(), 0, nullptr, &e_);
#endif
  }
  void sync() {
#if defined(GUH_NATIVE)
    detail::g_last_error = (Error_t)GU_API(EventSynchronize)(e_);
#elif defined(GUH_OPENCL)
    if (e_) clWaitForEvents(1, &e_);
#endif
  }
#if defined(GUH_NATIVE)
  GU_API_T(Event) native() const { return e_; }
#elif defined(GUH_OPENCL)
  cl_event native() const { return e_; }
#endif
};

static inline void EventRecord(event& e, stream& s) { e.record(s); }
static inline void EventSynchronize(event& e) { e.sync(); }

static inline float ElapsedTime(event& start, event& end) {
#if defined(GUH_NATIVE)
  float ms = 0; detail::g_last_error = (Error_t)GU_API(EventElapsedTime)(&ms, start.native(), end.native()); return ms;
#elif defined(GUH_OPENCL)
  start.sync(); end.sync();
  cl_ulong t0 = 0, t1 = 0;
  clGetEventProfilingInfo(start.native(), CL_PROFILING_COMMAND_END, sizeof(t0), &t0, nullptr);
  clGetEventProfilingInfo(end.native(), CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
  return (float)(t1 - t0) / 1e6f;
#else
  (void)start; (void)end; return 0.0f;
#endif
}

/* ---- dim3 ---- */
#if defined(GUH_NATIVE)
using dim3 = ::dim3;  /* alias to system dim3 for consistent gu::dim3 usage */
#else
struct dim3 {
  unsigned x, y, z;
  dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
#endif

/* ---- Kernel & Launch ---- */
namespace detail {
  struct kernel_ref {
#if defined(GUH_NATIVE)
    void* func = nullptr;
#elif defined(GUH_OPENCL)
    cl_kernel k = nullptr;
    cl_program p = nullptr;
#endif
  };

#if defined(GUH_NATIVE)
  static void* g_args[GUH_MAX_ARGS];
  template<typename T>
  static inline void set_arg(kernel_ref&, int& idx, const T& v) {
    static thread_local T storage[GUH_MAX_ARGS];
    storage[idx] = v;
    g_args[idx++] = &storage[idx - 1];
  }
#elif defined(GUH_OPENCL)
  template<typename T>
  static inline void set_arg(kernel_ref& kr, int& idx, const T& v) {
    clSetKernelArg(kr.k, (cl_uint)idx++, sizeof(T), &v);
  }

  /* Kernel cache: auto-caches compiled kernels by source pointer */
  struct kernel_cache_entry {
    const char* src;
    kernel_ref kr;
    kernel_cache_entry* next;
  };

  inline kernel_ref& get_or_compile(const char* src, const char* name) {
    static kernel_cache_entry* head = nullptr;
    for (auto* e = head; e; e = e->next) {
      if (e->src == src) return e->kr;
    }
    auto* entry = new kernel_cache_entry{src, {}, head};
    cl_int err;
    entry->kr.p = clCreateProgramWithSource(g_context, 1, &src, nullptr, &err);
    if (err == CL_SUCCESS) {
      err = clBuildProgram(entry->kr.p, 1, &g_device, "-cl-std=CL1.2", nullptr, nullptr);
      if (err == CL_SUCCESS) entry->kr.k = clCreateKernel(entry->kr.p, name, &err);
    }
    g_last_error = err;
    head = entry;
    return entry->kr;
  }
#endif

  template<typename... Args>
  static inline void set_args(kernel_ref& kr, int& idx, const Args&... args) {
    int dummy[] = {0, (set_arg(kr, idx, args), 0)...};
    (void)dummy;
  }
} // namespace detail

#if defined(GUH_NATIVE)
#define GetKernel(fn) (fn)
#elif defined(GUH_OPENCL)
#define GetKernel(fn) gu::detail::get_or_compile(fn##_gpuni_src, #fn)
#endif

#if defined(GUH_NATIVE)
template<typename Kernel, typename... Args>
static inline void Launch(Kernel kernel, int grid, int block, const Args&... args) {
  kernel<<<grid, block>>>(args...);
}

template<typename Kernel, typename... Args>
static inline void Launch(Kernel kernel, dim3 grid, dim3 block, const Args&... args) {
  kernel<<<grid, block>>>(args...);
}

template<typename Kernel, typename... Args>
static inline void Launch(Kernel kernel, dim3 grid, dim3 block, size_t smem, const Args&... args) {
  kernel<<<grid, block, smem>>>(args...);
}

template<typename Kernel, typename... Args>
static inline void Launch(Kernel kernel, dim3 grid, dim3 block, stream& s, const Args&... args) {
  kernel<<<grid, block, 0, s.native()>>>(args...);
}

template<typename Kernel, typename... Args>
static inline void Launch(Kernel kernel, dim3 grid, dim3 block, size_t smem, stream& s, const Args&... args) {
  kernel<<<grid, block, smem, s.native()>>>(args...);
}

template<typename Kernel, typename... Args>
static inline void Launch(Kernel kernel, int grid, int block, size_t smem, const Args&... args) {
  kernel<<<grid, block, smem>>>(args...);
}

template<typename Kernel, typename... Args>
static inline void Launch(Kernel kernel, int grid, int block, stream& s, const Args&... args) {
  kernel<<<grid, block, 0, s.native()>>>(args...);
}

template<typename Kernel, typename... Args>
static inline void Launch(Kernel kernel, int grid, int block, size_t smem, stream& s, const Args&... args) {
  kernel<<<grid, block, smem, s.native()>>>(args...);
}
#elif defined(GUH_OPENCL)
/* Launch overloads */
template<typename... Args>
static inline void Launch(detail::kernel_ref kr, int grid, int block, const Args&... args) {
  int idx = 0; detail::set_args(kr, idx, args...);
  size_t global = (size_t)grid * (size_t)block, local = (size_t)block;
  clEnqueueNDRangeKernel(detail::g_queue, kr.k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
}

template<typename... Args>
static inline void Launch(detail::kernel_ref kr, dim3 grid, dim3 block, const Args&... args) {
  int idx = 0; detail::set_args(kr, idx, args...);
  size_t global[3] = {grid.x * block.x, grid.y * block.y, grid.z * block.z};
  size_t local[3] = {block.x, block.y, block.z};
  clEnqueueNDRangeKernel(detail::g_queue, kr.k, 3, nullptr, global, local, 0, nullptr, nullptr);
}

template<typename... Args>
static inline void Launch(detail::kernel_ref kr, dim3 grid, dim3 block, size_t smem, const Args&... args) {
  int idx = 0; detail::set_args(kr, idx, args...);
  if (smem > 0) clSetKernelArg(kr.k, (cl_uint)idx, smem, nullptr);
  size_t global[3] = {grid.x * block.x, grid.y * block.y, grid.z * block.z};
  size_t local[3] = {block.x, block.y, block.z};
  clEnqueueNDRangeKernel(detail::g_queue, kr.k, 3, nullptr, global, local, 0, nullptr, nullptr);
}

template<typename... Args>
static inline void Launch(detail::kernel_ref kr, dim3 grid, dim3 block, stream& s, const Args&... args) {
  int idx = 0; detail::set_args(kr, idx, args...);
  size_t global[3] = {grid.x * block.x, grid.y * block.y, grid.z * block.z};
  size_t local[3] = {block.x, block.y, block.z};
  clEnqueueNDRangeKernel(s.native(), kr.k, 3, nullptr, global, local, 0, nullptr, nullptr);
}

template<typename... Args>
static inline void Launch(detail::kernel_ref kr, dim3 grid, dim3 block, size_t smem, stream& s, const Args&... args) {
  int idx = 0; detail::set_args(kr, idx, args...);
  if (smem > 0) clSetKernelArg(kr.k, (cl_uint)idx, smem, nullptr);
  size_t global[3] = {grid.x * block.x, grid.y * block.y, grid.z * block.z};
  size_t local[3] = {block.x, block.y, block.z};
  clEnqueueNDRangeKernel(s.native(), kr.k, 3, nullptr, global, local, 0, nullptr, nullptr);
}

template<typename... Args>
static inline void Launch(detail::kernel_ref kr, int grid, int block, size_t smem, const Args&... args) {
  int idx = 0; detail::set_args(kr, idx, args...);
  if (smem > 0) clSetKernelArg(kr.k, (cl_uint)idx, smem, nullptr);
  size_t global = (size_t)grid * (size_t)block, local = (size_t)block;
  clEnqueueNDRangeKernel(detail::g_queue, kr.k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
}

template<typename... Args>
static inline void Launch(detail::kernel_ref kr, int grid, int block, stream& s, const Args&... args) {
  int idx = 0; detail::set_args(kr, idx, args...);
  size_t global = (size_t)grid * (size_t)block, local = (size_t)block;
  clEnqueueNDRangeKernel(s.native(), kr.k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
}

template<typename... Args>
static inline void Launch(detail::kernel_ref kr, int grid, int block, size_t smem, stream& s, const Args&... args) {
  int idx = 0; detail::set_args(kr, idx, args...);
  if (smem > 0) clSetKernelArg(kr.k, (cl_uint)idx, smem, nullptr);
  size_t global = (size_t)grid * (size_t)block, local = (size_t)block;
  clEnqueueNDRangeKernel(s.native(), kr.k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
}
#endif

} // namespace gu

#endif /* __cplusplus && (GUH_CUDA || GUH_HIP || GUH_OPENCL) */

#endif
