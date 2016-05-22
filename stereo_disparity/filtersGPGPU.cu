//#ifndef __CUDACC__  
//	#define __CUDACC__
//#endif

#include "image.h"
#include "cudaErrorCheck.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cassert>
#include <math.h>

#define BLOCK 16

void gradXWithCuda(float *host_out, const float *host_in, dim3 imSize, dim3 grid, dim3 block);
void boxFilterWithCuda(float *host_out, const float *host_in, dim3 imSize, dim3 grid, dim3 block);

__global__ void gradXKernel(float *dev_out, const float *dev_in, const int width, const int height)
{
	extern __shared__ float sharedMemory[];

	int i = threadIdx.x - 1 + blockIdx.x * (blockDim.x - 2);
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < 0) {
		i = 0;
	}

	if (i > width - 1) {
		i = width - 1;
	}

	const int inBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;
	const int globalIdx = i + j * width;
	sharedMemory[inBlockIdx] = dev_in[globalIdx];

	__syncthreads();

	if (threadIdx.x > 0 && threadIdx.x <= blockDim.x - 2) {
		dev_out[globalIdx] = .5f * (sharedMemory[inBlockIdx + 1] - sharedMemory[inBlockIdx - 1]);
	}

	__syncthreads();
}

__global__ void boxFilterKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

Image Image::gradXGPGPU() const {
	assert(w >= 2);
	Image D(w, h);
	float *out = D.tab;

	dim3 imSize(w, h);
	dim3 block(BLOCK + 2, BLOCK);
	dim3 grid(ceil(w/BLOCK), ceil(h/BLOCK));

	// cudaOccupancyMaxActiveBlocksPerMultiprocessor ??

	gradXWithCuda(out, tab, imSize, grid, block);

	CudaSafeCall(cudaDeviceReset());

	return D;
}

//Image Image::boxFilterGPGPU(int radius) const {
//}

void gradXWithCuda(float *host_out, const float *host_in, dim3 imSize, dim3 grid, dim3 block) {
	float *dev_in = 0;
	float *dev_out = 0;
	int size = imSize.x * imSize.y * sizeof(float);

	// Choose which GPU to run on, change this on a multi-GPU system.
	CudaSafeCall(cudaSetDevice(0));

	// Allocate GPU buffers for three vectors (two input, one output).
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));

	// Copy input vectors from host memory to GPU buffers.
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	// Launch a kernel on the GPU with one thread for each element.
	gradXKernel <<<grid, block, block.x * block.y * sizeof(float)>>>(dev_out, dev_in, imSize.x, imSize.y);
	CudaCheckError();

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CudaSafeCall(cudaDeviceSynchronize());

	// Copy output vector from GPU buffer to host memory.
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
}

void boxFilterWithCuda(float *host_out, const float *host_in, dim3 imSize, dim3 grid, dim3 block) {
}