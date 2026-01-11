#ifndef POLYKERNEL_H
#define POLYKERNEL_H

/* PolyKernel CUDA-truth kernel dialect.
 *
 * Naming principle:
 * - Prefer CUDA/C99 spellings in kernels (math `*f`, `atomicAdd/atomicCAS/...`, etc.).
 * - Use `pk_*` only for real portability gaps where OpenCL 1.2 needs extra code
 *   to emulate missing semantics (e.g. fixed-point float accumulation, `u64` add).
 */

#define PK_DIALECT_VERSION 1

#if defined(__OPENCL_VERSION__) || defined(__OPENCL_C_VERSION__)
#  define PK_BACKEND_OPENCL 1
#elif defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#  define PK_BACKEND_HIP 1
#elif defined(__CUDACC__)
#  define PK_BACKEND_CUDA 1
#else
#  define PK_BACKEND_HOST 1
#endif

#if defined(PK_BACKEND_OPENCL)
typedef int pk_i32;
typedef uint pk_u32;
typedef long pk_i64;
typedef ulong pk_u64;
#else
typedef int pk_i32;
typedef unsigned int pk_u32;
typedef long long pk_i64;
typedef unsigned long long pk_u64;
#endif

#define PK_FIXED_Q32_32_SCALE_F 4294967296.0f
#define PK_FIXED_Q32_32_INV_SCALE_F 2.3283064365386963e-10f /* 2^-32 */

#if defined(PK_BACKEND_HIP)
#  include <hip/hip_runtime.h>
#endif

#if defined(PK_BACKEND_OPENCL)

#  if defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ < 120
#    error "PolyKernel requires OpenCL C 1.2+"
#  elif defined(__OPENCL_VERSION__) && __OPENCL_VERSION__ < 120
#    error "PolyKernel requires OpenCL C 1.2+"
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
#  define __device__ inline
#  define __global__ __kernel
#  define __shared__ __local
#  define __constant__ __constant
#  define __launch_bounds__(t, b)

#  define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)

#  define threadIdx ((uint3)(get_local_id(0), get_local_id(1), get_local_id(2)))
#  define blockIdx  ((uint3)(get_group_id(0), get_group_id(1), get_group_id(2)))
#  define blockDim  ((uint3)(get_local_size(0), get_local_size(1), get_local_size(2)))
#  define gridDim   ((uint3)(get_num_groups(0), get_num_groups(1), get_num_groups(2)))

#  define PK_GLOBAL __global
#  define PK_LOCAL __local
#  define PK_CONSTANT __constant

#  define PK_GLOBAL_PTR(T) PK_GLOBAL T*
#  define PK_LOCAL_PTR(T) PK_LOCAL T*
#  define PK_CONSTANT_PTR(T) PK_CONSTANT T*

#  define PK_RESTRICT restrict
#  define PK_INLINE inline

/* Dynamic shared memory: OpenCL uses kernel param, no binding needed */
#  define PK_BIND_DYNAMIC_SMEM(ptr) /* no-op */

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

static PK_INLINE void pk_atomic_add_u64(PK_GLOBAL pk_u64* p, pk_u64 val) {
#  if defined(cl_khr_int64_base_atomics) && !defined(PK_DISABLE_OPENCL_INT64_ATOMICS)
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
  const uint old_lower = atomic_add(&word[low], lower);
  const uint sum = old_lower + lower;
  upper += (sum < old_lower) ? 1u : 0u;
  if (upper != 0u) atomic_add(&word[1 - low], upper);
#  endif
}

static PK_INLINE pk_i64 pk_real_to_fixed_q32_32(float x) {
  return (pk_i64)(x * PK_FIXED_Q32_32_SCALE_F);
}

static PK_INLINE float pk_fixed_q32_32_to_real(pk_i64 x) {
  return (float)x * PK_FIXED_Q32_32_INV_SCALE_F;
}

static PK_INLINE void pk_atomic_add_fixed_q32_32(PK_GLOBAL pk_u64* p, float x) {
  pk_atomic_add_u64(p, (pk_u64)pk_real_to_fixed_q32_32(x));
}

/* Float atomic add (OpenCL 1.2 has no atomic_add(float); emulate via CAS on u32 bits).
   Correctness-first; prefer fixed-point(Q32.32) for high-throughput accumulation. */
static PK_INLINE float pk_atomic_add_f32(PK_GLOBAL float* p, float x) {
  volatile PK_GLOBAL pk_u32* u = (volatile PK_GLOBAL pk_u32*)p;
  pk_u32 old = atomic_add(u, (pk_u32)0);
  for (;;) {
    pk_u32 assumed = old;
    pk_u32 desired = as_uint(as_float(assumed) + x);
    old = atomic_cmpxchg(u, assumed, desired);
    if (old == assumed) return as_float(assumed);
  }
}

/* OpenCL is C, no extern "C" needed */
#  define PK_EXTERN_C

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

#  define PK_GLOBAL __global
#  define PK_LOCAL __local
#  define PK_CONSTANT __constant

#  define PK_GLOBAL_PTR(T) T*
#  define PK_LOCAL_PTR(T) T*
#  define PK_CONSTANT_PTR(T) T*

#  if defined(_MSC_VER)
#    define PK_RESTRICT __restrict
#  else
#    define PK_RESTRICT __restrict__
#  endif
#  if defined(PK_BACKEND_CUDA) || defined(PK_BACKEND_HIP)
#    define PK_INLINE __forceinline__
#  else
#    define PK_INLINE inline
#  endif

/* Dynamic shared memory: CUDA/HIP binds to extern __shared__ */
#  define PK_BIND_DYNAMIC_SMEM(ptr) \
     extern __shared__ unsigned char _pk_smem_[]; \
     (ptr) = (decltype(ptr))(&_pk_smem_[0])

#  if defined(PK_BACKEND_CUDA) || defined(PK_BACKEND_HIP)
static __device__ PK_INLINE void pk_atomic_add_u64(PK_GLOBAL pk_u64* p, pk_u64 val) {
  (void)atomicAdd((unsigned long long*)p, (unsigned long long)val);
}

static __device__ PK_INLINE float pk_atomic_add_f32(PK_GLOBAL float* p, float x) {
  return atomicAdd((float*)p, x);
}

static __device__ PK_INLINE pk_i64 pk_real_to_fixed_q32_32(float x) {
  return (pk_i64)(x * PK_FIXED_Q32_32_SCALE_F);
}

static __device__ PK_INLINE float pk_fixed_q32_32_to_real(pk_i64 x) {
  return (float)x * PK_FIXED_Q32_32_INV_SCALE_F;
}

static __device__ PK_INLINE void pk_atomic_add_fixed_q32_32(PK_GLOBAL pk_u64* p, float x) {
  pk_atomic_add_u64(p, (pk_u64)pk_real_to_fixed_q32_32(x));
}
#  endif

/* Prevent C++ name mangling for kernel symbols */
#  define PK_EXTERN_C extern "C"

#endif

#endif
