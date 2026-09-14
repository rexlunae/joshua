// SPDX-License-Identifier: MIT
// OpenCL 3.0 backend smoke test for the Intel ICD / iGPU (joshua host).
// Proves the backend can BUILD + EXECUTE a kernel on the iGPU.
// Build: gcc -o /tmp/ocl_smoke ocl_smoke.c -lOpenCL    Run: /tmp/ocl_smoke
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>

static const char* SRC =
    "__kernel void vadd(__global const float* a, __global const float* b,"
    "                   __global float* c){ int i = get_global_id(0); c[i]=a[i]+b[i]; }";

int main() {
    cl_int err = 0; cl_uint np = 0; cl_platform_id pf[4];
    cl_uint nd = 0; cl_device_id dev = NULL; cl_context ctx = NULL; cl_command_queue q = NULL;

    if (clGetPlatformIDs(4, pf, &np) != 0) { puts("FAIL clGetPlatformIDs"); return 1; }
    printf("platforms: %u\n", np);
    if (np == 0) { puts("no platform"); return 1; }

    if (clGetDeviceIDs(pf[0], CL_DEVICE_TYPE_ALL, 1, &dev, &nd) != 0 || !dev) {
        puts("FAIL clGetDeviceIDs"); return 1;
    }
    char dname[64] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, NULL);
    printf("device: %s\n", dname);

    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (!ctx) { printf("FAIL clCreateContext rc=%d\n", err); return 1; }
    q = clCreateCommandQueue(ctx, dev, 0, &err);
    if (!q) { printf("FAIL clCreateCommandQueue rc=%d\n", err); return 1; }

    cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err);
    if (!prog) { printf("FAIL clCreateProgramWithSource rc=%d\n", err); return 1; }
    if (clBuildProgram(prog, 0, NULL, NULL, NULL, NULL) != 0) {
        puts("FAIL clBuildProgram"); return 1;
    }
    printf("program built\n");

    cl_kernel k = clCreateKernel(prog, "vadd", &err);
    if (!k) { printf("FAIL clCreateKernel rc=%d\n", err); return 1; }

    const int n = 8; float a[8], b[8], c[8];
    for (int i = 0; i < n; i++) { a[i] = (float)i; b[i] = 1.0f; c[i] = -1.0f; }

    cl_mem ma = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(a), NULL, &err);
    cl_mem mb = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(b), NULL, &err);
    cl_mem mc = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(c), NULL, &err);
    if (!ma || !mb || !mc) { printf("FAIL clCreateBuffer rc=%d\n", err); return 1; }

    if (clSetKernelArg(k, 0, sizeof(cl_mem), &ma) != 0) { puts("FAIL arg a"); return 1; }
    if (clSetKernelArg(k, 1, sizeof(cl_mem), &mb) != 0) { puts("FAIL arg b"); return 1; }
    if (clSetKernelArg(k, 2, sizeof(cl_mem), &mc) != 0) { puts("FAIL arg c"); return 1; }
    if (clEnqueueWriteBuffer(q, ma, CL_TRUE, 0, sizeof(a), a, 0, NULL, NULL) != 0) { puts("FAIL write a"); return 1; }
    if (clEnqueueWriteBuffer(q, mb, CL_TRUE, 0, sizeof(b), b, 0, NULL, NULL) != 0) { puts("FAIL write b"); return 1; }
    size_t g[1] = { (size_t)n };
    if (clEnqueueNDRangeKernel(q, k, 1, NULL, g, NULL, 0, NULL, NULL) != 0) { puts("FAIL exec"); return 1; }
    if (clEnqueueReadBuffer(q, mc, CL_TRUE, 0, sizeof(c), c, 0, NULL, NULL) != 0) { puts("FAIL read"); return 1; }
    clFinish(q);

    int pass = 1;
    for (int i = 0; i < n; i++) if (c[i] != (float)(i + 1)) pass = 0;
    printf("RESULT: c = [");
    for (int i = 0; i < n; i++) printf(" %.0f", c[i]);
    printf(" ]\n");
    printf("%s\n", pass ? "PASS: vector add ran on iGPU" : "FAIL: mismatch");
    return pass ? 0 : 2;
}