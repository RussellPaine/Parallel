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
#include "sphere.h"
#include "vec3.h"
#include "tempSphere.h"

#define NumOfParticles 10000
#define systemBounds vec3(10, 10, 10)

const int ThreadsPerBlock = 100;
const int blocks = NumOfParticles / ThreadsPerBlock;

__device__ int colorType = 1;
__device__ bool gravity = false;

sphere **d_list;
vec3 *d_centerOfMass;

typedef unsigned int uint;
typedef unsigned char uchar;

__global__ void create_world(sphere **d_list, tempSphere *particleList);

__device__ vec3 castRay(const ray &r, sphere **d_list);
__global__ void move_particles(sphere **d_list, int colorType, vec3 *centerOfMass);
__global__ void apply_gravity(sphere **d_list);
__global__ void get_CenterOfMass(sphere **d_list, vec3 *d_CenterOfMass);
//#include "bicubicTexture_kernel.cuh"

cudaArray *d_imageArray = 0;

extern "C" void initParticles() {
	srand (static_cast <unsigned> (time(0)));

	tempSphere particleList[NumOfParticles];

	tempSphere *d_particleList;
	checkCudaErrors(cudaMalloc((void **)&d_particleList,  NumOfParticles * sizeof(tempSphere)));
	checkCudaErrors(cudaMemcpy(d_particleList, particleList, NumOfParticles *  sizeof(tempSphere), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **)&d_list, NumOfParticles * sizeof(sphere *)));
	checkCudaErrors(cudaMalloc((void **)&d_centerOfMass,  sizeof(vec3)));

	create_world<<<1, 1>>>(d_list, d_particleList);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void toggleGravity() {
	gravity = !gravity;
}

extern "C" void handleColorChange(int num) {
	colorType = num;
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

__global__ void d_render(uchar4 *d_output, uint width, uint height, sphere **d_list)
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
    vec3 eye = vec3(0, 0, 7);
    float distFrEye2Img = 1.0;

    if ((x < width) && (y < height))
    {
        //for each pixel
        vec3 pixelPos = vec3(u, v, eye.z() - distFrEye2Img);
        //fire a ray:
        ray r;
        r.O = eye;
        r.Dir = pixelPos - eye; //view direction along negtive z-axis!
        vec3 col = castRay(r, d_list);
        float red = col.x();
        float green = col.y();
        float blue = col.z();
        d_output[i] = make_uchar4(red * 255, green * 255, blue * 255, 0);
    }
}



// render image using CUDA
extern "C" void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4 *output)
{
	if(gravity) {
        apply_gravity <<< blocks, ThreadsPerBlock >>> (d_list);
        checkCudaErrors(cudaGetLastError());
	    checkCudaErrors(cudaDeviceSynchronize());
    }
	//move the particles
	if (colorType == 4) {
    get_CenterOfMass<<< 1, 1 >>>(d_list, d_centerOfMass);
    checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
    }

	move_particles <<< blocks, ThreadsPerBlock >>> (d_list, colorType, d_centerOfMass);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// [32,32] [16,16] Scales automatically for the size of the window
	d_render <<< gridSize, blockSize >>> (output, width, height, d_list);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("kernel failed");
}

__global__ void get_CenterOfMass(sphere **d_list, vec3 *d_CenterOfMass)
{
    	for (int i = 0; i < NumOfParticles; i++) {
    		*d_CenterOfMass += d_list[i]->center;
    	}
    	*d_CenterOfMass /= NumOfParticles;
}

__global__ void create_world(sphere **d_list, tempSphere *particleList)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
    	for (int i = 0; i < NumOfParticles; i++) {
    		*(d_list + i) = new sphere(particleList[i].center, particleList[i].veloctiy, particleList[i].color, particleList[i].radius);
		}
    }
}

__device__ vec3 castRay(const ray &r, sphere **d_list)
{
    hit_record rec;

	bool hit_anything = false;
	float closest_so_far = FLT_MAX;
	for (int i = 0; i < NumOfParticles; i++) {
		if (d_list[i]->hit(r, 0.0, closest_so_far, rec)) {
			hit_anything = true;
			closest_so_far = rec.t;
		}
	}

    if (hit_anything) return rec.color;
    else return vec3(0, 0, 0);
}

__global__ void move_particles(sphere **d_list, int colorType, vec3 *centerOfMass)
{
    int i = blockIdx.x *blockDim.x + threadIdx.x;
   // (*world)->update(i, NumOfParticles ,systemBounds, colorType);
    d_list[i]->update_position(systemBounds);
    d_list[i]->update_color(colorType, centerOfMass, d_list, NumOfParticles);
}

__global__ void apply_gravity(sphere **d_list)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	d_list[i]->update_gravity();
}

#endif
