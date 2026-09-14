// SPDX-License-Identifier: MIT
// OpenCL buffer round-trip on the joshua host's Intel iGPU.
//
// Raw equivalent of candle's OpenClStorage::from_vec / to_cpu_storage:
// write f32 -> host -> device, read f32 -> device -> host, verify equality.
// Verified PASS on Intel(R) UHD Graphics 730 (see docs/opencl-backend-plan.md).
//
// Build: gcc -o /tmp/ocl_rt ocl_roundtrip.c -lOpenCL     Run: /tmp/ocl_rt
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
int main(){
    cl_int err=0; cl_uint np=0; cl_platform_id pf[4];
    cl_uint nd=0; cl_device_id dev=NULL;
    if(clGetPlatformIDs(4,pf,&np)!=0||np==0){puts("no platform");return 1;}
    if(clGetDeviceIDs(pf[0],CL_DEVICE_TYPE_ALL,1,&dev,&nd)!=0||!dev){puts("no dev");return 1;}
    cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&err);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&err);
    int n=16; float *src=malloc(sizeof(float)*n), *dst=malloc(sizeof(float)*n);
    for(int i=0;i<n;i++){src[i]=i*1.5f; dst[i]=-1.0f;}
    cl_mem buf=clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizeof(float)*n,NULL,&err);
    if(clEnqueueWriteBuffer(q,buf,CL_TRUE,0,sizeof(float)*n,src,0,NULL,NULL)!=0){puts("FAIL write");return 1;}
    if(clEnqueueReadBuffer(q,buf,CL_TRUE,0,sizeof(float)*n,dst,0,NULL,NULL)!=0){puts("FAIL read");return 1;}
    clFinish(q);
    int ok=1; for(int i=0;i<n;i++) if(dst[i]!=src[i]){ok=0;break;}
    printf("roundtrip: dst[3]=%.2f src[3]=%.2f -> %s\n",dst[3],src[3], ok?"PASS":"FAIL");
    return ok?0:2;
}