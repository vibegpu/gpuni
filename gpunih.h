/* gpunih.h - gpuni host-side API
 * Unified context, memory, kernel launch for CUDA/HIP/OpenCL.
 */
#ifndef GPUNIH_H
#define GPUNIH_H

#include <stddef.h>

/* Backend detection */
#if !defined(PKH_CUDA) && !defined(PKH_HIP) && !defined(PKH_OPENCL)
#  if defined(__CUDACC__) || defined(CUDA_VERSION)
#    define PKH_CUDA 1
#  elif defined(__HIPCC__)
#    define PKH_HIP 1
#  endif
#endif

/* Includes */
#if defined(PKH_CUDA)
#  include <cuda_runtime.h>
#elif defined(PKH_HIP)
#  include <hip/hip_runtime.h>
#elif defined(PKH_OPENCL)
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

#define PKH_MAX_ARGS 24

/* ============================================================
 * Context
 * ============================================================ */
typedef struct pk_ctx {
#if defined(PKH_CUDA)
    int device;
    cudaStream_t stream;
#elif defined(PKH_HIP)
    int device;
    hipStream_t stream;
#elif defined(PKH_OPENCL)
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
#else
    int dummy;
#endif
} pk_ctx;

static inline int pk_ctx_init(pk_ctx* c, int dev) {
#if defined(PKH_CUDA)
    c->device = dev;
    if (cudaSetDevice(dev) != cudaSuccess) return -1;
    return (int)cudaStreamCreate(&c->stream);
#elif defined(PKH_HIP)
    c->device = dev;
    if (hipSetDevice(dev) != hipSuccess) return -1;
    return (int)hipStreamCreate(&c->stream);
#elif defined(PKH_OPENCL)
    (void)dev;
    cl_platform_id plat; cl_int e;
    clGetPlatformIDs(1, &plat, NULL);
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

static inline void pk_ctx_destroy(pk_ctx* c) {
#if defined(PKH_CUDA)
    cudaStreamDestroy(c->stream);
#elif defined(PKH_HIP)
    hipStreamDestroy(c->stream);
#elif defined(PKH_OPENCL)
    if (c->queue) clReleaseCommandQueue(c->queue);
    if (c->context) clReleaseContext(c->context);
#else
    (void)c;
#endif
}

static inline void pk_sync(pk_ctx* c) {
#if defined(PKH_CUDA)
    cudaStreamSynchronize(c->stream);
#elif defined(PKH_HIP)
    hipStreamSynchronize(c->stream);
#elif defined(PKH_OPENCL)
    clFinish(c->queue);
#else
    (void)c;
#endif
}

/* ============================================================
 * Memory
 * ============================================================ */
static inline void* pk_malloc(pk_ctx* c, size_t n) {
    void* p = NULL;
#if defined(PKH_CUDA)
    (void)c; cudaMalloc(&p, n);
#elif defined(PKH_HIP)
    (void)c; hipMalloc(&p, n);
#elif defined(PKH_OPENCL)
    p = (void*)clCreateBuffer(c->context, CL_MEM_READ_WRITE, n, NULL, NULL);
#else
    (void)c; p = malloc(n);
#endif
    return p;
}

static inline void pk_free(pk_ctx* c, void* p) {
#if defined(PKH_CUDA)
    (void)c; cudaFree(p);
#elif defined(PKH_HIP)
    (void)c; hipFree(p);
#elif defined(PKH_OPENCL)
    (void)c; if (p) clReleaseMemObject((cl_mem)p);
#else
    (void)c; free(p);
#endif
}

/* ============================================================
 * Memcpy
 * ============================================================ */
static inline void pk_h2d(pk_ctx* c, void* d, const void* s, size_t n) {
#if defined(PKH_CUDA)
    cudaMemcpyAsync(d, s, n, cudaMemcpyHostToDevice, c->stream);
#elif defined(PKH_HIP)
    hipMemcpyAsync(d, s, n, hipMemcpyHostToDevice, c->stream);
#elif defined(PKH_OPENCL)
    clEnqueueWriteBuffer(c->queue, (cl_mem)d, CL_FALSE, 0, n, s, 0, NULL, NULL);
#else
    (void)c; memcpy(d, s, n);
#endif
}

static inline void pk_d2h(pk_ctx* c, void* d, const void* s, size_t n) {
#if defined(PKH_CUDA)
    cudaMemcpyAsync(d, s, n, cudaMemcpyDeviceToHost, c->stream);
#elif defined(PKH_HIP)
    hipMemcpyAsync(d, s, n, hipMemcpyDeviceToHost, c->stream);
#elif defined(PKH_OPENCL)
    clEnqueueReadBuffer(c->queue, (cl_mem)s, CL_FALSE, 0, n, d, 0, NULL, NULL);
#else
    (void)c; memcpy(d, s, n);
#endif
}

static inline void pk_d2d(pk_ctx* c, void* d, const void* s, size_t n) {
#if defined(PKH_CUDA)
    cudaMemcpyAsync(d, s, n, cudaMemcpyDeviceToDevice, c->stream);
#elif defined(PKH_HIP)
    hipMemcpyAsync(d, s, n, hipMemcpyDeviceToDevice, c->stream);
#elif defined(PKH_OPENCL)
    clEnqueueCopyBuffer(c->queue, (cl_mem)s, (cl_mem)d, 0, 0, n, 0, NULL, NULL);
#else
    (void)c; memcpy(d, s, n);
#endif
}

/* ============================================================
 * Kernel
 * ============================================================ */
typedef struct pk_kernel {
    int nargs;
#if defined(PKH_CUDA)
    void* func;
    void* args[PKH_MAX_ARGS];
#elif defined(PKH_HIP)
    void* func;
    void* args[PKH_MAX_ARGS];
#elif defined(PKH_OPENCL)
    cl_kernel kernel;
    cl_program program;
#else
    void* func;
#endif
} pk_kernel;

static inline int pk_kernel_create(pk_ctx* c, pk_kernel* k, void* func, const char* src, const char* name) {
    k->nargs = 0;
#if defined(PKH_CUDA) || defined(PKH_HIP)
    (void)c; (void)src; (void)name;
    k->func = func;
    return 0;
#elif defined(PKH_OPENCL)
    (void)func;
    cl_int e;
    k->program = clCreateProgramWithSource(c->context, 1, &src, NULL, &e);
    if (e != CL_SUCCESS) return (int)e;
    e = clBuildProgram(k->program, 1, &c->device, "-cl-std=CL1.2", NULL, NULL);
    if (e != CL_SUCCESS) {
        /* Print build log for debugging */
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
        return (int)e;
    }
    k->kernel = clCreateKernel(k->program, name, &e);
    return (int)e;
#else
    (void)c; (void)func; (void)src; (void)name;
    return 0;
#endif
}

static inline void pk_kernel_destroy(pk_kernel* k) {
#if defined(PKH_OPENCL)
    if (k->kernel) clReleaseKernel(k->kernel);
    if (k->program) clReleaseProgram(k->program);
#else
    (void)k;
#endif
}

/* Add argument */
static inline void pk_arg_impl(pk_kernel* k, const void* ptr, size_t sz) {
#if defined(PKH_CUDA) || defined(PKH_HIP)
    if (k->nargs < PKH_MAX_ARGS) k->args[k->nargs++] = (void*)ptr;
    (void)sz;
#elif defined(PKH_OPENCL)
    if (k->nargs < PKH_MAX_ARGS) clSetKernelArg(k->kernel, (cl_uint)k->nargs++, sz, ptr);
#else
    (void)k; (void)ptr; (void)sz;
#endif
}

#define pk_arg(k, v) pk_arg_impl(k, &(v), sizeof(v))

static inline void pk_args_reset(pk_kernel* k) { k->nargs = 0; }

/* Launch kernel */
static inline void pk_run(pk_ctx* c, pk_kernel* k, int grid, int block, int smem) {
#if defined(PKH_CUDA)
    cudaLaunchKernel(k->func, dim3(grid), dim3(block), k->args, (size_t)smem, c->stream);
#elif defined(PKH_HIP)
    hipLaunchKernel(k->func, dim3(grid), dim3(block), k->args, (size_t)smem, c->stream);
#elif defined(PKH_OPENCL)
    if (smem > 0) clSetKernelArg(k->kernel, (cl_uint)k->nargs, (size_t)smem, NULL);
    size_t global = (size_t)grid * (size_t)block;
    size_t local = (size_t)block;
    clEnqueueNDRangeKernel(c->queue, k->kernel, 1, NULL, &global, &local, 0, NULL, NULL);
#else
    (void)c; (void)k; (void)grid; (void)block; (void)smem;
#endif
    pk_args_reset(k);
}

#endif /* GPUNIH_H */
