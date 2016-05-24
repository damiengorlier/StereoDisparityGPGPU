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
//#define BLOCK_SCAN 128

void gradXWithCuda(float *host_out, const float *host_in, dim3 imSize, dim3 grid, dim3 block);
void scanWithCuda(float *host_out, const float *host_in, dim3 imSize, dim3 grid, dim3 blok, bool addOri);
void transposeWithCuda(float *host_out, const float *host_in, dim3 imSize, dim3 grid, dim3 blok);
void boxFilterWithCuda(float *host_out, const float *host_in, dim3 imSize, dim3 grid, dim3 block);

Image Image::gradXGPGPU() const {
	assert(w >= 2);
	Image D(w, h);
	float *out = D.tab;

	dim3 imSize(w, h);
	dim3 block(BLOCK + 2, BLOCK);
	dim3 grid(ceil(w / BLOCK), ceil(h / BLOCK));

	// TODO : chercher cudaOccupancyMaxActiveBlocksPerMultiprocessor

	gradXWithCuda(out, tab, imSize, grid, block);

	return D;
}

Image Image::integralGPGPU() const {
	Image D(w, h);
	float *out = D.tab;
	float *tmp1 = new float[w*h];
	float *tmp2 = new float[w*h];

	int a = (w > h) ? w : h;

	dim3 imSize(w, h);

	dim3 block(w, 1);
	dim3 grid(1, h);
	scanWithCuda(tmp1, tab, imSize, grid, block, true);

	block = dim3(BLOCK, BLOCK);
	grid = dim3(ceil(a / BLOCK), ceil(a / BLOCK));
	transposeWithCuda(tmp2, tmp1, imSize, grid, block);
	
	imSize = dim3(h, w);
	block = dim3(h, 1);
	grid = dim3(1, w);
	scanWithCuda(tmp1, tmp2, imSize, grid, block, true);

	block = dim3(BLOCK, BLOCK);
	grid = dim3(ceil(a / BLOCK), ceil(a / BLOCK));
	transposeWithCuda(out, tmp1, imSize, grid, block);

	return D;
}

Image Image::boxFilterGPGPU(int radius) const {
	Image D(w, h);
	float *out = D.tab;

	dim3 imSize(w, h);
	dim3 block(BLOCK, BLOCK);
	dim3 grid(ceil(w / BLOCK), ceil(h / BLOCK));

	boxFilterWithCuda(out, tab, imSize, grid, block);

	return D;
}

// Should be private. Only for test purpose
Image Image::scanGPGPU() const {
	Image D(w, h);
	float *out = D.tab;

	dim3 imSize(w, h);
	//dim3 block(BLOCK_SCAN, 1);
	//dim3 grid(w / BLOCK_SCAN, h);
	dim3 block(w, 1);
	dim3 grid(1, h);

	scanWithCuda(out, tab, imSize, grid, block, true);

	return D;
}

Image Image::transposeGPGPU() const {
	Image D(h, w);
	float *out = D.tab;
	int a = (w > h) ? w : h;

	dim3 imSize(w, h);
	dim3 block(BLOCK, BLOCK);
	dim3 grid(ceil(a / BLOCK), ceil(a / BLOCK));

	transposeWithCuda(out, tab, imSize, grid, block);

	return D;
}

__global__ void addKernel(float *dev_out, float *dev_in1, float *dev_in2, const int width, const int height) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i <= width) && (j < height)) {

		int globalIdx = i + j * width;

		dev_out[globalIdx] = dev_in1[globalIdx] + dev_in2[globalIdx];
	}
}

__global__ void gradXKernel(float *dev_out, const float *dev_in, const int width, const int height) {
	// 1 block recouvre une partie de l'image que l'on veut traiter avec un thread en plus de chaque côté vertical
	//
	//     X - - - - - - X
	//     X - - - - - - X
	//     X - - - - - - X
	//     X - - - - - - X
	//     X - - - - - - X
	//     X - - - - - - X
	//
	extern __shared__ float sharedMemory[];

	int i = threadIdx.x - 1 + blockIdx.x * (blockDim.x - 2);
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i <= width) && (j < height)) {

		if (i < 0) {
			i = 0;
		}

		if (i == width) {
			i = width - 1;
		}

		const int inBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;
		const int globalIdx = i + j * width;
		sharedMemory[inBlockIdx] = dev_in[globalIdx];

		__syncthreads();

		if (threadIdx.x > 0 && threadIdx.x <= blockDim.x - 2) {
			dev_out[globalIdx] = .5f * (sharedMemory[inBlockIdx + 1] - sharedMemory[inBlockIdx - 1]);
		}

		//__syncthreads();
	}
}

__global__ void scanKernel(float *dev_out, const float *dev_in, const int n, const int width, const int height) {
	// Faire un parallel scan sur une image de dimension quelconque est très long à implémenter:
	// L'algorithme est prévu pour des blocks dont la taille est une puissance de 2
	// => Si la longueur de l'image n'est pas une puissance de 2, il faut diviser la longueur en plusieurs blocks
	// Il y a une étape intermédiaire pour récupérer la somme totale sur toute la longueur de l'image:
	// Il construire une array intermédiaire contenant les sommes de chaque block, et ensuite refaire un scan sur cette array.
	// Mais si la taille de cette array n'est pas non plus une puissance de 2 => refaire l'étape intermédiaire => etc.
	// C'est donc un algorithme récursif qu'il faut implémenter (début avec "aux", mais c'est trop compliqué).
	// Voir http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

	extern __shared__ float sharedMemory[];

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < width) && (j < height)) {

		const int tdx = threadIdx.x;
		const int globalIdx = i + j * width;

		sharedMemory[tdx] = dev_in[globalIdx];

		int offset = 1;

		for (int d = n >> 1; d > 0; d >>= 1)
		{
			__syncthreads();
			if (tdx < d)
			{
				int ai = offset * (2 * tdx + 1) - 1;
				int bi = offset * (2 * tdx + 2) - 1;
				sharedMemory[bi] += sharedMemory[ai];
			}
			offset *= 2;
		}

		if (tdx == 0) {
			//if (dev_aux) {
			//	int inGridIdx = blockIdx.y * gridDim.y + blockIdx.x;
			//	dev_aux[inGridIdx] = sharedMemory[n - 1];
			//}
			sharedMemory[n - 1] = 0;
		}

		__syncthreads();

		for (int d = 1; d < n; d *= 2)
		{
			offset >>= 1;
			__syncthreads();
			if (tdx < d)
			{
				int ai = offset * (2 * tdx + 1) - 1;
				int bi = offset * (2 * tdx + 2) - 1;
				float t = sharedMemory[ai];
				sharedMemory[ai] = sharedMemory[bi];
				sharedMemory[bi] += t;
			}
		}

		__syncthreads();

		dev_out[globalIdx] = sharedMemory[tdx];
	}
}

__global__ void transposeKernel(float *dev_out, const float *dev_in, const int width, const int height) {
	extern __shared__ float sharedMemory[];

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	// 
	if ((i < width) && (j < height)) {
		int globalIdx = i + j * width;
		const int inBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;
		sharedMemory[inBlockIdx] = dev_in[globalIdx];
	}

	__syncthreads();

	int i_t = blockIdx.y * blockDim.y + threadIdx.x;
	int j_t = blockIdx.x * blockDim.x + threadIdx.y;

	if ((i_t < height) && (j_t < width)) {
		int globalIdx = i_t + j_t * height;
		const int inBlockIdx = threadIdx.x * blockDim.y + threadIdx.y;
		dev_out[globalIdx] = sharedMemory[inBlockIdx];
	}

	//__syncthreads();
}

__global__ void boxFilterKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

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
	gradXKernel<<<grid, block, block.x * block.y * sizeof(float)>>>(dev_out, dev_in, imSize.x, imSize.y);
	CudaCheckError();

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CudaSafeCall(cudaDeviceSynchronize());

	// Copy output vector from GPU buffer to host memory.
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));

	CudaSafeCall(cudaDeviceReset());
}

void scanWithCuda(float *host_out, const float *host_in, dim3 imSize, dim3 grid, dim3 block, bool addOri) {
	//float *host_aux = new float[grid.x * grid.y];
	float *dev_in = 0;
	float *dev_tmp = 0;
	float *dev_out = 0;
	//float *dev_aux = 0;
	int size = imSize.x * imSize.y * sizeof(float);
	//int sizeAux = grid.x * grid.y * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	//CudaSafeCall(cudaMalloc((void**)&dev_aux, sizeAux));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	scanKernel<<<grid, block, block.x * block.y * sizeof(float)>>>(dev_out, dev_in, block.x, imSize.x, imSize.y);
	//scanKernel<<<grid, block, block.x * block.y * sizeof(float)>>>(dev_out, dev_aux, dev_in, block.x, imSize.x, imSize.y);
	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());

	if (addOri) {
		CudaSafeCall(cudaMalloc((void**)&dev_tmp, size));
		addKernel<<<grid, block>>>(dev_tmp, dev_in, dev_out, imSize.x, imSize.y);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaMemcpy(host_out, dev_tmp, size, cudaMemcpyDeviceToHost));
	}
	else{
		CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	}
	//CudaSafeCall(cudaMemcpy(host_aux, dev_aux, sizeAux, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void transposeWithCuda(float *host_out, const float *host_in, dim3 imSize, dim3 grid, dim3 block) {
	float *dev_in = 0;
	float *dev_out = 0;
	int size = imSize.x * imSize.y * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	transposeKernel<<<grid, block, block.x * block.y * sizeof(float)>>>(dev_out, dev_in, imSize.x, imSize.y);

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void boxFilterWithCuda(float *host_out, const float *host_in, dim3 imSize, dim3 grid, dim3 block) {
}