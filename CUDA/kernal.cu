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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
#include "sphereTemplate.h"

#define NumOfParticles 100

hitable **d_world;
sphere **d_list;

typedef unsigned int uint;
typedef unsigned char uchar;

__global__ void create_world(sphere **d_list, hitable **d_world, sphereTemplate *particleList);

__device__ vec3 castRay(const ray &r, hitable **world);
__global__ void move_particles(hitable **world);
//#include "bicubicTexture_kernel.cuh"

cudaArray *d_imageArray = 0;

extern "C" void initParticles() {

	srand (static_cast <unsigned> (time(0)));

	sphereTemplate particleList[NumOfParticles];

	sphereTemplate *d_particleList;
	checkCudaErrors(cudaMalloc((void **)&d_particleList,  NumOfParticles * sizeof(tempSphere)));
	checkCudaErrors(cudaMemcpy(d_particleList, particleList, NumOfParticles *  sizeof(tempSphere), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **)&d_list, NumOfParticles * sizeof(sphere *)));
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));

	create_world<<<1, 1>>>(d_list, d_world, d_particleList);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

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

__global__ void d_render(uchar4 *d_output, uint width, uint height, hitable **d_world)
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


//    hitable_list *d_world =  new hitable_list();
//    d_world->list_size = NumOfParticles;
//
//    checkCudaErrors(cudaMalloc((void **)&d_world, NumOfParticles * sizeof(hitable)));


//    sphere *d_particleList;
//    checkCudaErrors(cudaMalloc((void **)&d_particleList,  NumOfParticles * sizeof(sphere)));
//    checkCudaErrors(cudaMemcpy(d_particleList, particleList, NumOfParticles *  sizeof(sphere), cudaMemcpyHostToDevice));
//
//    move_particles <<< 1, NumOfParticles >>> (d_particleList);
//
//    hitable_list *d_world;
//	checkCudaErrors(cudaMalloc((void **)&d_world,  sizeof(hitable_list)));
//	checkCudaErrors(cudaMemcpy(d_world, new hitable_list(d_particleList, NumOfParticles), sizeof(hitable_list), cudaMemcpyHostToDevice));
//
//




	move_particles <<< 1, NumOfParticles >>> (d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
////    // [32,32] [16,16]
	d_render <<< gridSize, blockSize >>> (output, width, height, d_world);
	getLastCudaError("kernel failed");
}


__global__ void create_world(sphere **d_list, hitable **d_world, sphereTemplate *particleList)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
    	for (int i = 0; i < NumOfParticles; i++) {
    		*(d_list + i) = new sphere(particleList[i].center, particleList[i].veloctiy, particleList[i].color, particleList[i].radius);
		}
    	*d_world = new hitable_list(d_list, NumOfParticles);
    }
}

__device__ vec3 castRay(const ray &r, hitable **world)
{
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec))
    {
        return rec.color;
    }
    else
    {
        return vec3(0, 0, 0);
    }
}

__global__ void move_particles(hitable **world)
{
    int i = threadIdx.x;
    (*world)->update(i);
}

#endif
