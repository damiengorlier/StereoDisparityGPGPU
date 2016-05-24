//#ifndef __CUDACC__  
//	#define __CUDACC__
//#endif

#include "cudaKernels.cuh"
#include "cudaErrorCheck.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void addKernel(float *dev_out, float *dev_in1, float *dev_in2, const int width, const int height) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i <= width) && (j < height)) {

		const int globalIdx = i + j * width;

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

		__syncthreads();

		int offset = 1;

		for (int d = n >> 1; d > 0; d >>= 1)
		{
			if (tdx < d)
			{
				int ai = offset * (2 * tdx + 1) - 1;
				int bi = offset * (2 * tdx + 2) - 1;
				sharedMemory[bi] += sharedMemory[ai];
			}
			offset *= 2;
			__syncthreads();
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
			if (tdx < d)
			{
				int ai = offset * (2 * tdx + 1) - 1;
				int bi = offset * (2 * tdx + 2) - 1;
				float t = sharedMemory[ai];
				sharedMemory[ai] = sharedMemory[bi];
				sharedMemory[bi] += t;
			}
			__syncthreads();
		}

		dev_out[globalIdx] = sharedMemory[tdx];
	}
}

__global__ void transposeKernel(float *dev_out, const float *dev_in, const int width, const int height) {
	extern __shared__ float sharedMemory[];

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < width) && (j < height)) {
		int globalIdx = i + j * width;
		int inBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;
		sharedMemory[inBlockIdx] = dev_in[globalIdx];
	}

	__syncthreads();

	int i_t = blockIdx.y * blockDim.y + threadIdx.x;
	int j_t = blockIdx.x * blockDim.x + threadIdx.y;

	if ((i_t < height) && (j_t < width)) {
		int globalIdx = i_t + j_t * height;
		int inBlockIdx = threadIdx.x * blockDim.y + threadIdx.y;
		dev_out[globalIdx] = sharedMemory[inBlockIdx];
	}

	//__syncthreads();
}

__global__ void boxFilterKernel(float *dev_out, const float *dev_in, const int radius, const int width, const int height)
{
	// Avec une image intégrale, la somme des pixels dans le carré ABCD : sum = D - B - C + A
	// Donc pour avoir la moyenne des pixels autours de O (rayon = radius), il suffit de calculer sum / nbr de pixels
	//
	//    -----> x
	//   |   ______________________
	//   |  |    _|          _|    |
	// y v  |__ |_|A_ __ __ |_|B   |
	//      |     |           |    |
	//      |     |     _     |    |
	//      |     |    |_|O   |    |
	//      |    _|          _|    |
	//      |__ |_|C_ __ __ |_|D   |
	//      |                      |
	//      |______________________|
	//
	//
	//   Ax => Cx
	//   Ay => By
	//   Dx => Bx
	//   Dy => Cy
	//
	extern __shared__ float sharedMemory[];

	int tileDimX = blockDim.x - 2 * radius - 1;
	int tileDimY = blockDim.y - 2 * radius - 1;

	int i = threadIdx.x - radius - 1 + blockIdx.x * tileDimX;
	int j = threadIdx.y - radius - 1 + blockIdx.y * tileDimY;

	if (i >= 0 && i < width && j >= 0 && j < height) {
		const int inBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;
		const int globalIdx = i + j * width;

		sharedMemory[inBlockIdx] = dev_in[globalIdx];

		__syncthreads();

		if (threadIdx.x > radius &&
			threadIdx.x < blockDim.x - radius &&
			threadIdx.y > radius &&
			threadIdx.y < blockDim.y - radius
			) {

			float A, B, C, D;    // Values of ABCD
			int Ax, Ay, Dx, Dy;  // Coordinates of AD in block
			int iA, jA, iD, jD;  // Coordinates of AD in image

			iA = i - radius - 1;
			jA = j - radius - 1;

			Ax = iA + radius + 1 - blockIdx.x * tileDimX;
			Ay = jA + radius + 1 - blockIdx.y * tileDimY;

			iD = i + radius;
			jD = j + radius;

			if (iD >= width) {
				iD = width - 1;
			}
			if (jD >= height) {
				jD = height - 1;
			}

			Dx = iD + radius + 1 - blockIdx.x * tileDimX;
			Dy = jD + radius + 1 - blockIdx.y * tileDimY;

			if (iA >= 0 && jA >= 0) {
				A = sharedMemory[Ax + Ay * blockDim.x];
				B = sharedMemory[Dx + Ay * blockDim.x];
				C = sharedMemory[Ax + Dy * blockDim.x];
				D = sharedMemory[Dx + Dy * blockDim.x];
			}
			else if (iA < 0 && jA >= 0) {
				A = 0;
				B = sharedMemory[Dx + Ay * blockDim.x];
				C = 0;
				D = sharedMemory[Dx + Dy * blockDim.x];

				iA = 0;
			}
			else if (jA < 0 && iA >= 0) {
				A = 0;
				B = 0;
				C = sharedMemory[Ax + Dy * blockDim.x];
				D = sharedMemory[Dx + Dy * blockDim.x];

				jA = 0;
			}
			else if (iA < 0 && jA < 0) {
				A = 0;
				B = 0;
				C = 0;
				D = sharedMemory[Dx + Dy * blockDim.x];

				iA = 0;
				jA = 0;
			}

			float sum = D - B - C + A;
			int nbrPx = (iD - iA) * (jD - jA);
			//int side = 2 * radius + 1;
			//int nbrPx = side * side;
			dev_out[globalIdx] = sum / nbrPx;
		}
	}
}

void gradXWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim) {
	dim3 imSize(width, height);
	dim3 block(blockDim + 2, blockDim);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim));

	// TODO : chercher cudaOccupancyMaxActiveBlocksPerMultiprocessor
	float *dev_in = 0;
	float *dev_out = 0;
	int size = imSize.x * imSize.y * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	gradXKernel << <grid, block, block.x * block.y * sizeof(float) >> >(dev_out, dev_in, imSize.x, imSize.y);

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void scanWithCuda(float *host_out, const float *host_in, const int width, const int height, const bool addOri) {
	dim3 imSize(width, height);
	dim3 block(width, 1);
	dim3 grid(1, height);

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

	scanKernel << <grid, block, block.x * block.y * sizeof(float) >> >(dev_out, dev_in, block.x, imSize.x, imSize.y);
	//scanKernel<<<grid, block, block.x * block.y * sizeof(float)>>>(dev_out, dev_aux, dev_in, block.x, imSize.x, imSize.y);
	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());

	if (addOri) {
		CudaSafeCall(cudaMalloc((void**)&dev_tmp, size));
		addKernel << <grid, block >> >(dev_tmp, dev_in, dev_out, imSize.x, imSize.y);
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

void transposeWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim) {
	int a = (width > height) ? width : height;
	dim3 imSize(width, height);
	dim3 block(blockDim, blockDim);
	dim3 grid(ceil(a / blockDim), ceil(a / blockDim));

	float *dev_in = 0;
	float *dev_out = 0;
	int size = imSize.x * imSize.y * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	transposeKernel << <grid, block, block.x * block.y * sizeof(float) >> >(dev_out, dev_in, imSize.x, imSize.y);

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void boxFilterWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim, int radius) {
	// host_in must be the integral image

	dim3 imSize(width, height);
	dim3 block(blockDim + 2 * radius + 1, blockDim + 2 * radius + 1);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim));

	float *dev_in = 0;
	float *dev_out = 0;
	int size = imSize.x * imSize.y * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	boxFilterKernel << <grid, block, block.x * block.y * sizeof(float) >> >(dev_out, dev_in, radius, imSize.x, imSize.y);

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}
