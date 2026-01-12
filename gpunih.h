/* gpunih.h - gpuni host-side API
 * Unified context, memory, kernel launch for CUDA/HIP/OpenCL.
 */
#ifndef GPUNIH_H
#define GPUNIH_H

#include <stddef.h>

/* Backend detection */
#if !defined(GUH_CUDA) && !defined(GUH_HIP) && !defined(GUH_OPENCL)
#  if defined(__CUDACC__) || defined(CUDA_VERSION)
#    define GUH_CUDA 1
#  elif defined(__HIPCC__)
#    define GUH_HIP 1
#  endif
#endif

/* Includes */
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

/* ============================================================
 * Context
 * ============================================================ */
typedef struct gu_ctx {
#if defined(GUH_CUDA)
    int device;
    cudaStream_t stream;
#elif defined(GUH_HIP)
    int device;
    hipStream_t stream;
#elif defined(GUH_OPENCL)
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
#else
    int dummy;
#endif
} gu_ctx;

static inline int gu_ctx_init(gu_ctx* c, int dev) {
#if defined(GUH_CUDA)
    c->device = dev;
    if (cudaSetDevice(dev) != cudaSuccess) return -1;
    return (int)cudaStreamCreate(&c->stream);
#elif defined(GUH_HIP)
    c->device = dev;
    if (hipSetDevice(dev) != hipSuccess) return -1;
    return (int)hipStreamCreate(&c->stream);
#elif defined(GUH_OPENCL)
    (void)dev;
    c->context = NULL;
    c->queue = NULL;
    c->device = NULL;
    cl_platform_id plat; cl_int e;
    e = clGetPlatformIDs(1, &plat, NULL);
    if (e != CL_SUCCESS) return (int)e;
    e = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &c->device, NULL);
    if (e != CL_SUCCESS)
        e = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &c->device, NULL);
    if (e != CL_SUCCESS) return (int)e;
    c->context = clCreateContext(NULL, 1, &c->device, NULL, NULL, &e);
    if (e != CL_SUCCESS) return (int)e;
    c->queue = clCreateCommandQueue(c->context, c->device, 0, &e);
    return (int)e;
#else
    (void)dev; c->dummy = 0; return 0;
#endif
}

static inline void gu_ctx_destroy(gu_ctx* c) {
#if defined(GUH_CUDA)
    cudaStreamDestroy(c->stream);
#elif defined(GUH_HIP)
    (void)hipStreamDestroy(c->stream);
#elif defined(GUH_OPENCL)
    if (c->queue) clReleaseCommandQueue(c->queue);
    if (c->context) clReleaseContext(c->context);
#else
    (void)c;
#endif
}

static inline void gu_sync(gu_ctx* c) {
#if defined(GUH_CUDA)
    cudaStreamSynchronize(c->stream);
#elif defined(GUH_HIP)
    (void)hipStreamSynchronize(c->stream);
#elif defined(GUH_OPENCL)
    clFinish(c->queue);
#else
    (void)c;
#endif
}

/* ============================================================
 * Memory
 * ============================================================ */
static inline void* gu_malloc(gu_ctx* c, size_t n) {
    void* p = NULL;
#if defined(GUH_CUDA)
    (void)c; cudaMalloc(&p, n);
#elif defined(GUH_HIP)
    (void)c; (void)hipMalloc(&p, n);
#elif defined(GUH_OPENCL)
    p = (void*)clCreateBuffer(c->context, CL_MEM_READ_WRITE, n, NULL, NULL);
#else
    (void)c; p = malloc(n);
#endif
    return p;
}

static inline void gu_free(gu_ctx* c, void* p) {
#if defined(GUH_CUDA)
    (void)c; cudaFree(p);
#elif defined(GUH_HIP)
    (void)c; (void)hipFree(p);
#elif defined(GUH_OPENCL)
    (void)c; if (p) clReleaseMemObject((cl_mem)p);
#else
    (void)c; free(p);
#endif
}

/* ============================================================
 * Memcpy
 * ============================================================ */
static inline void gu_h2d(gu_ctx* c, void* d, const void* s, size_t n) {
#if defined(GUH_CUDA)
    cudaMemcpyAsync(d, s, n, cudaMemcpyHostToDevice, c->stream);
#elif defined(GUH_HIP)
    (void)hipMemcpyAsync(d, s, n, hipMemcpyHostToDevice, c->stream);
#elif defined(GUH_OPENCL)
    clEnqueueWriteBuffer(c->queue, (cl_mem)d, CL_FALSE, 0, n, s, 0, NULL, NULL);
#else
    (void)c; memcpy(d, s, n);
#endif
}

static inline void gu_d2h(gu_ctx* c, void* d, const void* s, size_t n) {
#if defined(GUH_CUDA)
    cudaMemcpyAsync(d, s, n, cudaMemcpyDeviceToHost, c->stream);
#elif defined(GUH_HIP)
    (void)hipMemcpyAsync(d, s, n, hipMemcpyDeviceToHost, c->stream);
#elif defined(GUH_OPENCL)
    clEnqueueReadBuffer(c->queue, (cl_mem)s, CL_FALSE, 0, n, d, 0, NULL, NULL);
#else
    (void)c; memcpy(d, s, n);
#endif
}

static inline void gu_d2d(gu_ctx* c, void* d, const void* s, size_t n) {
#if defined(GUH_CUDA)
    cudaMemcpyAsync(d, s, n, cudaMemcpyDeviceToDevice, c->stream);
#elif defined(GUH_HIP)
    (void)hipMemcpyAsync(d, s, n, hipMemcpyDeviceToDevice, c->stream);
#elif defined(GUH_OPENCL)
    clEnqueueCopyBuffer(c->queue, (cl_mem)s, (cl_mem)d, 0, 0, n, 0, NULL, NULL);
#else
    (void)c; memcpy(d, s, n);
#endif
}

/* ============================================================
 * Kernel
 * ============================================================ */
typedef struct gu_kernel {
    int nargs;
#if defined(GUH_CUDA) || defined(GUH_HIP)
    void* func;
    void* args[GUH_MAX_ARGS];
#elif defined(GUH_OPENCL)
    cl_kernel kernel;
    cl_program program;
#else
    void* func;
#endif
} gu_kernel;

static inline int gu_kernel_create(gu_ctx* c, gu_kernel* k, void* func, const char* src, const char* name) {
    k->nargs = 0;
#if defined(GUH_CUDA) || defined(GUH_HIP)
    (void)c; (void)src; (void)name;
    k->func = func;
    return 0;
#elif defined(GUH_OPENCL)
    (void)func;
    cl_int e;
    k->kernel = NULL;
    k->program = NULL;
    k->program = clCreateProgramWithSource(c->context, 1, &src, NULL, &e);
    if (e != CL_SUCCESS) return (int)e;
    e = clBuildProgram(k->program, 1, &c->device, GUH_OPENCL_BUILD_OPTIONS, NULL, NULL);
    if (e != CL_SUCCESS) {
        size_t len = 0;
        clGetProgramBuildInfo(k->program, c->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        if (len > 1) {
            char* log = (char*)malloc(len);
            if (log) {
                clGetProgramBuildInfo(k->program, c->device, CL_PROGRAM_BUILD_LOG, len, log, NULL);
                fprintf(stderr, "OpenCL build error:\n%s\n", log);
                free(log);
            }
        }
        clReleaseProgram(k->program);
        k->program = NULL;
        return (int)e;
    }
    k->kernel = clCreateKernel(k->program, name, &e);
    return (int)e;
#else
    (void)c; (void)func; (void)src; (void)name;
    return 0;
#endif
}

static inline void gu_kernel_destroy(gu_kernel* k) {
#if defined(GUH_OPENCL)
    if (k->kernel) clReleaseKernel(k->kernel);
    if (k->program) clReleaseProgram(k->program);
#else
    (void)k;
#endif
}

/* Add argument */
static inline void gu_arg_impl(gu_kernel* k, const void* ptr, size_t sz) {
#if defined(GUH_CUDA) || defined(GUH_HIP)
    if (k->nargs < GUH_MAX_ARGS) k->args[k->nargs++] = (void*)ptr;
    (void)sz;
#elif defined(GUH_OPENCL)
    if (k->nargs < GUH_MAX_ARGS) clSetKernelArg(k->kernel, (cl_uint)k->nargs++, sz, ptr);
#else
    (void)k; (void)ptr; (void)sz;
#endif
}

#define gu_arg(k, v) gu_arg_impl(k, &(v), sizeof(v))

static inline void gu_args_reset(gu_kernel* k) { k->nargs = 0; }

/* Launch kernel */
static inline void gu_run(gu_ctx* c, gu_kernel* k, int grid, int block, int smem) {
#if defined(GUH_CUDA)
    cudaLaunchKernel(k->func, dim3(grid), dim3(block), k->args, (size_t)smem, c->stream);
#elif defined(GUH_HIP)
    (void)hipLaunchKernel(k->func, dim3(grid), dim3(block), k->args, (size_t)smem, c->stream);
#elif defined(GUH_OPENCL)
    if (smem > 0) clSetKernelArg(k->kernel, (cl_uint)k->nargs, (size_t)smem, NULL);
    size_t global = (size_t)grid * (size_t)block;
    size_t local = (size_t)block;
    clEnqueueNDRangeKernel(c->queue, k->kernel, 1, NULL, &global, &local, 0, NULL, NULL);
#else
    (void)c; (void)k; (void)grid; (void)block; (void)smem;
#endif
    gu_args_reset(k);
}

/* ============================================================
 * Unified kernel creation macro
 * ============================================================
 * Usage: GU_KERNEL(&ctx, &k, gu_saxpy);
 *
 * Include the generated .gu.h header before using this macro.
 */
#if defined(GUH_CUDA) || defined(GUH_HIP)
#define GU_KERNEL(ctx, k, name) \
    gu_kernel_create(ctx, k, (void*)(name), NULL, NULL)
#elif defined(GUH_OPENCL)
#define GU_KERNEL(ctx, k, name) \
    gu_kernel_create(ctx, k, NULL, name##_gu_source, #name)
#else
#define GU_KERNEL(ctx, k, name) \
    gu_kernel_create(ctx, k, NULL, NULL, NULL)
#endif

#endif /* GPUNIH_H */
