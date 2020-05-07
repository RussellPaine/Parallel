/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cfloat>

#include <helper_math.h>

// includes, cuda
#include <helper_cuda.h>

#include "Ray.h"
#include "hitable.h"
#include "hitable_list.h"
#include "sphere.h"
#include "vec3.h"

typedef unsigned int uint;
typedef unsigned char uchar;

//Sphere X velocity
__device__ static float sphereX = 1.1;
//Sphere Y velocity
__device__ static float sphereY = 0.11;
//Sphere Z velocity
__device__ static float sphereZ = 0.0;

__device__ static float stepSizeX = 0.01;
__device__ static float stepSizeY = 0.01;
__device__ static float stepSizeZ = 0.00;

//Sphere X velocity
__device__ static float sphere2X = 1.1;
//Sphere Y velocity
__device__ static float sphere2Y = 0.11;
//Sphere Z velocity
__device__ static float sphere2Z = 0.0;

__device__ static float stepSize2X = -0.01;
__device__ static float stepSize2Y = -0.01;
__device__ static float stepSize2Z = -0.00;

__device__ vec3 castRay(const ray &r, hitable **world);
__global__ void create_world(hitable **d_list, hitable **d_world);
//#include "bicubicTexture_kernel.cuh"

cudaArray *d_imageArray = 0;

extern "C" void initTexture(int imageWidth, int imageHeight, uchar *h_data)
{
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_imageArray;
}
extern "C" void freeTexture()
{
    checkCudaErrors(cudaFreeArray(d_imageArray));
}

__global__ void
d_render(uchar4 *d_output, uint width, uint height, hitable **d_world)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint i = y * width + x;
    float u = x / (float)width; //----> [0, 1]x[0, 1]
    float v = y / (float)height;
    u = 2.0 * u - 1.0; //---> [-1, 1]x[-1, 1]
    v = -(2.0 * v - 1.0);
    u *= width / (float)height;
    u *= 2.0;
    v *= 2.0;
    vec3 eye = vec3(0, 0.5, 1.5);
    float distFrEye2Img = 1.0;
    ;
    if ((x < width) && (y < height))
    {
        //for each pixel
        vec3 pixelPos = vec3(u, v, eye.z() - distFrEye2Img);
        //fire a ray:
        ray r;
        r.O = eye;
        r.Dir = pixelPos - eye; //view direction along negtive z-axis!
        vec3 col = castRay(r, d_world);
        float red = col.x();
        float green = col.y();
        float blue = col.z();
        d_output[i] = make_uchar4(red * 255, green * 255, blue * 255, 0);
    }
}

// render image using CUDA
extern "C" void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4 *output)
{
    /*d_render << <gridSize, blockSize >> > (output, width, height);
     // call CUDA kernel, writing results to PBO memory
     getLastCudaError("kernel failed");*/

    // make our world of hitables
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2 * sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    create_world<<<1, 1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    d_render<<<gridSize, blockSize>>>(output, width, height, d_world);
    getLastCudaError("kernel failed");
}

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << "at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
__device__ vec3 castRay(const ray &r, hitable **world)
{
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec))
    {
        return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f,
                           rec.normal.z() + 1.0f);
    }
    else
    {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }
}
__global__ void create_world(hitable **d_list, hitable **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {

        //Sphere
        //Static Ball*(d_list) = new sphere(vec3(0, 0, -1), 0.5);

        sphereX += stepSizeX;
        sphereY += stepSizeY;
        sphereZ += stepSizeZ;
        if (sphereX > 1.8)
        {
            stepSizeX = -0.01;
        }
        if (sphereY > 1.8)
        {
            stepSizeY = -0.01;
        }
        if (sphereZ > 1.8)
        {
            stepSizeZ = -0.01;
        }

        if (sphereX < -1.8)
        {
            stepSizeX = 0.01;
        }
        if (sphereY < -1.8)
        {
            stepSizeY = 0.01;
        }
        if (sphereZ < -1.8)
        {
            stepSizeZ = 0.01;
        }

        sphere2X += stepSize2X;
        sphere2Y += stepSize2Y;
        sphere2Z += stepSize2Z;
        if (sphere2X > 1.8)
        {
            stepSize2X = -0.01;
        }
        if (sphere2Y > 1.8)
        {
            stepSize2Y = -0.01;
        }
        if (sphere2Z > 1.8)
        {
            stepSize2Z = -0.01;
        }

        if (sphere2X < -1.8)
        {
            stepSize2X = 0.01;
        }
        if (sphere2Y < -1.8)
        {
            stepSize2Y = 0.01;
        }
        if (sphere2Z < -1.8)
        {
            stepSize2Z = 0.01;
        }

        *(d_list) = new sphere(vec3(sphereX, sphereY, sphereZ), 0.2);

        *(d_list + 1) = new sphere(vec3(sphere2X, sphere2Y, sphere2Z), 0.2);

        //Left Wall
        *(d_list + 2) = new sphere(vec3(-10002.0, 0, -3), 10000);
        //Right Wall
        *(d_list + 3) = new sphere(vec3(10002.0, 0, -3), 10000);
        //Top Wall
        *(d_list + 4) = new sphere(vec3(0, 10002.0, -3), 10000);
        //Bottom Wall
        *(d_list + 5) = new sphere(vec3(0, -10002.0, -3), 10000);
        //Back Wall
        *(d_list + 6) = new sphere(vec3(0, 0, -10002.0), 10000);

        *d_world = new hitable_list(d_list, 7);
    }
}
__global__ void free_world(hitable **d_list, hitable **d_world)
{
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
}

#endif
