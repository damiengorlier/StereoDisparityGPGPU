//#ifndef __CUDACC__  
//	#define __CUDACC__
//#endif

#include "cudaKernels.cuh"
#include "cudaErrorCheck.cuh"
#include "TimingGPU.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// TODO : - chercher cudaOccupancyMaxActiveBlocksPerMultiprocessor
//        - Dans le rapport, expliquer pourquoi on doit avoir des images carrées (voir scanKernel)
//        - rgb_to_gray en GPGPU ?

#define TIMING 1
#define MAX_FLOAT 3.402823466e+38f

#define ISPOW2(x) ((x) > 0 && !((x) & (x-1)))

__device__ void inverseSym(float *inverse, const float *matrix) {
	inverse[0] = matrix[4] * matrix[8] - matrix[5] * matrix[7];
	inverse[1] = matrix[2] * matrix[7] - matrix[1] * matrix[8];
	inverse[2] = matrix[1] * matrix[5] - matrix[2] * matrix[4];
	float det = matrix[0] * inverse[0] + matrix[3] * inverse[1] + matrix[6] * inverse[2];
	det = 1 / det;
	inverse[0] *= det;
	inverse[1] *= det;
	inverse[2] *= det;
	inverse[3] = inverse[1];
	inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * det;
	inverse[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * det;
	inverse[6] = inverse[2];
	inverse[7] = inverse[5];
	inverse[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * det;
}

__global__ void operatorKernel(float *dev_out, float *dev_in1, float *dev_in2, const int width, const int height, const int op) {
	//
	// op : 0 - PLUS
	//      1 - MINUS
	//      2 - MULTIPLY
	//

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i <= width) && (j < height)) {

		const int globalIdx = i + j * width;

		if (op == 0) {
			dev_out[globalIdx] = dev_in1[globalIdx] + dev_in2[globalIdx];
		}
		else if (op == 1) {
			dev_out[globalIdx] = dev_in1[globalIdx] - dev_in2[globalIdx];
		}
		else if (op == 2) {
			dev_out[globalIdx] = dev_in1[globalIdx] * dev_in2[globalIdx];
		}
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
	}
}

__global__ void scanKernel(float *dev_out, const float *dev_in, const int n, const int width, const int height) {
	// Faire un parallel scan sur une image de dimension quelconque est très long à implémenter:
	// L'algorithme est prévu pour des blocks dont la taille est une puissance de 2
	// => Si la longueur de l'image n'est pas une puissance de 2, il faut diviser la longueur en plusieurs blocks
	// Il y a une étape intermédiaire pour récupérer la somme totale sur toute la longueur de l'image:
	// Il construire une array intermédiaire contenant les sommes de chaque block, et ensuite refaire un scan sur cette array.
	// Mais si la taille de cette array n'est pas non plus une puissance de 2 => refaire l'étape intermédiaire => etc.
	// C'est donc un algorithme récursif qu'il faut implémenter.
	// Voir http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

	// Ici on utilise un "hack" (triche) : on augmente la taille du block à la puissance de 2 supérieure. Les threads en dehors
	// de l'image auront une valeur 0 => on aura donc un scan de la forme:
	// 1 2 3 4 5 0 0 0 => 1 3 6 10 15 15 15 15
	// Les valeurs finales en dehors de l'image ne sont pas reprises.
	// Plus l'image est grande, plus le nombre de threads inutils est grand = Perte de ressources.

	extern __shared__ float sharedMemory[];

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;


	const int tdx = threadIdx.x;
	const int globalIdx = i + j * width;

	if ((i < width) && (j < height)) {
		sharedMemory[tdx] = dev_in[globalIdx];
	}
	else {
		sharedMemory[tdx] = 0;
	}

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

	if ((i < width) && (j < height)) {
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

__global__ void costVolumeKernel(float *dev_cost_out,
								 const float *dev_im1R, const float *dev_im1G, const float *dev_im1B,
								 const float *dev_im2R, const float *dev_im2G, const float *dev_im2B,
								 const float *dev_grad1, const float *dev_grad2,
								 const int width, const int height, const int dispMin, const int dispMax,
								 const float colorTh, const float gradTh, const float alpha) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int d = blockIdx.z + dispMin;

	if (i < width && j < height) {

		//printf("(i, j) = (%d, %d) | d = %d\n", i, j, d);

		int globalIdx = i + j * width;
		float costColor = colorTh;
		float costGrad = gradTh;
		if (d <= dispMax && i + d >= 0 && i + d < width) {
			
			float col1[] = { dev_im1R[globalIdx], dev_im1G[globalIdx], dev_im1B[globalIdx] };
			float col2[] = { dev_im2R[globalIdx + d], dev_im2G[globalIdx + d], dev_im2B[globalIdx + d] };
			float cost = 0;
			for (int k = 0; k < 3; k++) {
				float tmp = col1[k] - col2[k];
				if (tmp < 0) tmp = -tmp;
				cost += tmp;
			}
			cost /= 3;
			if (cost < colorTh) costColor = cost;

			cost = dev_grad1[globalIdx] - dev_grad2[globalIdx + d];
			if (cost < 0) cost = -cost;
			if (cost < gradTh) costGrad = cost;
		}

		dev_cost_out[globalIdx + blockIdx.z * width * height] = (1 - alpha) * costColor + alpha * costGrad;
	}
}

__global__ void aCoeffSpaceKernel(float *dev_out,
								  const float *dev_varRR, const float *dev_varRG, const float *dev_varRB,
								  const float *dev_varGG, const float *dev_varGB, const float *dev_varBB,
								  const float *dev_covarRCost, const float *dev_covarGCost, const float *dev_covarBCost,
								  const int width, const int height, const float epsilon) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < width && j < height) {
		int globalIdx = i + j * width;

		float S1[9] = {
			dev_varRR[globalIdx] + epsilon,		dev_varRG[globalIdx],				dev_varRB[globalIdx],
			dev_varRG[globalIdx],				dev_varGG[globalIdx] + epsilon,		dev_varGB[globalIdx],
			dev_varRB[globalIdx],				dev_varGB[globalIdx],				dev_varBB[globalIdx] + epsilon
		};

		float S2[9];
		inverseSym(S2, S1);

		dev_out[globalIdx] = dev_covarRCost[globalIdx] * S2[0] + dev_covarGCost[globalIdx] * S2[1] + dev_covarBCost[globalIdx] * S2[2];
		dev_out[globalIdx + width * height] = dev_covarRCost[globalIdx] * S2[3] + dev_covarGCost[globalIdx] * S2[4] + dev_covarBCost[globalIdx] * S2[5];
		dev_out[globalIdx + 2 * width * height] = dev_covarRCost[globalIdx] * S2[6] + dev_covarGCost[globalIdx] * S2[7] + dev_covarBCost[globalIdx] * S2[8];
	}
}

__global__ void disparitySelectionKernel(float *dev_out, const float *dev_cost_volume,
										 const int width, const int height, const int dispMin, const int dispMax) {

	extern __shared__ float sharedMemory[];

	int i = blockIdx.x;
	int j = blockIdx.y;
	int k = threadIdx.z;

	if (i < width && j < height) {

		const int tdx = threadIdx.x;
		const int globalIdx = i + j * width + k * height * width;

		if (k + dispMin <= dispMax) {
			sharedMemory[tdx] = dev_cost_volume[globalIdx];
		}
		else {
			sharedMemory[tdx] = MAX_FLOAT;
		}

		__syncthreads();

		int offset = 1;

		for (int d = blockDim.z >> 1; d > 0; d >>= 1)
		{
			if (tdx < d)
			{
				int ai = offset * (2 * tdx + 1) - 1;
				int bi = offset * (2 * tdx + 2) - 1;
				sharedMemory[bi] = (sharedMemory[bi] < sharedMemory[ai]) ? sharedMemory[bi] : sharedMemory[ai];
			}
			offset *= 2;
			__syncthreads();
		}

		if (tdx == 0) {
			dev_out[i + j * width] = sharedMemory[blockDim.z - 1];
		}
	}
}

void operatorWithCuda(float *host_out, const float *host_in1, const float *host_in2, const int width, const int height, const int blockDim, const int op) {
	dim3 block(blockDim, blockDim);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim));

	float *dev_in1 = 0;
	float *dev_in2 = 0;
	float *dev_out = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in1, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in2, size));
	CudaSafeCall(cudaMemcpy(dev_in1, host_in1, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_in2, host_in2, size, cudaMemcpyHostToDevice));

	TimingGPU timer;
	timer.StartCounter();

	operatorKernel << < grid, block >> >(dev_out, dev_in1, dev_in2, width, height, op);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | operatorKernel (op=" << op << ") : " << time << " ms" << std::endl;
	}

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void gradXWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim) {
	dim3 block(blockDim + 2, blockDim);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim));

	float *dev_in = 0;
	float *dev_out = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	//printf("		gradXWithCuda : grid = (%d, %d) | block = (%d, %d) | size = %d\n", grid.x, grid.y, block.x, block.y, block.x * block.y * sizeof(float));

	TimingGPU timer;
	timer.StartCounter();

	gradXKernel << < grid, block, block.x * block.y * sizeof(float) >> >(dev_out, dev_in, width, height);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | gradXKernel : " << time << " ms" << std::endl;
	}

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void scanWithCuda(float *host_out, const float *host_in, const int width, const int height, const bool addOri) {

	int blockSize = width;

	//printf("		scanWithCuda : %d is power of 2 ? %s\n", blockSize, ISPOW2(blockSize) ? "true" : "false");

	if (!ISPOW2(blockSize)) {
		int x = blockSize;
		blockSize = 2;
		while (x >>= 1) blockSize <<= 1;
	}

	dim3 block(blockSize, 1);
	dim3 grid(1, height);

	float *dev_in = 0;
	float *dev_tmp = 0;
	float *dev_out = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	//printf("		scanWithCuda : grid = (%d, %d) | block = (%d, %d)\n", grid.x, grid.y, block.x, block.y);

	TimingGPU timer;
	timer.StartCounter();

	scanKernel << <grid, block, block.x * block.y * sizeof(float) >> >(dev_out, dev_in, block.x, width, height);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | scanKernel : " << time << " ms" << std::endl;
	}

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());

	if (addOri) {
		CudaSafeCall(cudaMalloc((void**)&dev_tmp, size));

		operatorKernel << < grid, block >> >(dev_tmp, dev_in, dev_out, width, height, 0);

		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaMemcpy(host_out, dev_tmp, size, cudaMemcpyDeviceToHost));
	}
	else{
		CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	}
	CudaSafeCall(cudaDeviceReset());
}

void transposeWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim) {
	int a = (width > height) ? width : height;
	dim3 block(blockDim, blockDim);
	dim3 grid(ceil(a / blockDim), ceil(a / blockDim));

	float *dev_in = 0;
	float *dev_out = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	TimingGPU timer;
	timer.StartCounter();

	if (TIMING) {
		transposeKernel << < grid, block, block.x * block.y * sizeof(float) >> >(dev_out, dev_in, width, height);
	}

	float time = timer.GetCounter();
	std::cout << "GPU | transposeKernel : " << time << " ms" << std::endl;

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void boxFilterWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim, int radius) {
	// host_in must be the integral image

	dim3 block(blockDim + 2 * radius + 1, blockDim + 2 * radius + 1);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim));

	float *dev_in = 0;
	float *dev_out = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	//printf("		boxfilterWithCuda : grid = (%d, %d) | block = (%d, %d)\n");

	TimingGPU timer;
	timer.StartCounter();

	boxFilterKernel << < grid, block, block.x * block.y * sizeof(float) >> >(dev_out, dev_in, radius, width, height);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | boxFilterKernel (r=" << radius << ") : " << time << " ms" << std::endl;
	}

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void costVolumeWithCuda(float *host_out,
						const float *host_im1R, const float *host_im1G, const float *host_im1B,
						const float *host_im2R, const float *host_im2G, const float *host_im2B,
						const float *host_grad1, float *host_grad2,
						const int width, const int height, const int blockDim,
						const int dispMin, const int dispMax, const float colorTh, const float gradTh, const float alpha) {
	
	int dispSize = dispMax - dispMin + 1;
	dim3 block(blockDim, blockDim);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim), dispSize);

	float *dev_out;
	float *dev_im1R = 0, *dev_im1G = 0, *dev_im1B = 0;
	float *dev_im2R = 0, *dev_im2G = 0, *dev_im2B = 0;
	float *dev_grad1 = 0, *dev_grad2 = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size * dispSize));
	CudaSafeCall(cudaMalloc((void**)&dev_im1R, size)); CudaSafeCall(cudaMalloc((void**)&dev_im1G, size)); CudaSafeCall(cudaMalloc((void**)&dev_im1B, size));
	CudaSafeCall(cudaMalloc((void**)&dev_im2R, size)); CudaSafeCall(cudaMalloc((void**)&dev_im2G, size)); CudaSafeCall(cudaMalloc((void**)&dev_im2B, size));
	CudaSafeCall(cudaMalloc((void**)&dev_grad1, size)); CudaSafeCall(cudaMalloc((void**)&dev_grad2, size));
	CudaSafeCall(cudaMemcpy(dev_im1R, host_im1R, size, cudaMemcpyHostToDevice)); CudaSafeCall(cudaMemcpy(dev_im1G, host_im1G, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_im1B, host_im1B, size, cudaMemcpyHostToDevice)); CudaSafeCall(cudaMemcpy(dev_im2R, host_im2R, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_im2G, host_im2G, size, cudaMemcpyHostToDevice)); CudaSafeCall(cudaMemcpy(dev_im2B, host_im2B, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_grad1, host_grad1, size, cudaMemcpyHostToDevice)); CudaSafeCall(cudaMemcpy(dev_grad2, host_grad2, size, cudaMemcpyHostToDevice));

	//printf("		costVolumeWithCuda : grid = (%d, %d) | block = (%d, %d)\n", grid.x, grid.y, block.x, block.y);

	TimingGPU timer;
	timer.StartCounter();

	costVolumeKernel << < grid, block >> >(dev_out,
										  dev_im1R, dev_im1G, dev_im1B,
										  dev_im2R, dev_im2G, dev_im2B,
										  dev_grad1, dev_grad2,
										  width, height, dispMin, dispMax, colorTh, gradTh, alpha);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | costVolumeKernel : " << time << " ms" << std::endl;
	}

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size * dispSize, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void aCoeffSpaceWithCuda(float *host_out,
						 const float *host_varRR, const float *host_varRG, const float *host_varRB,
						 const float *host_varGG, const float *host_varGB, const float *host_varBB,
						 const float *host_covarRCost, const float *host_covarGCost, const float *host_covarBCost,
						 const int width, const int height, const int blockDim,
						 const float epsilon) {

	dim3 block(blockDim, blockDim);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim));

	float *dev_out;
	float *dev_varRR = 0; float *dev_varRG = 0; float *dev_varRB = 0;
	float *dev_varGG = 0; float *dev_varGB = 0; float *dev_varBB = 0;
	float *dev_covarRCost = 0; float *dev_covarGCost = 0; float *dev_covarBCost = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size * 3));
	CudaSafeCall(cudaMalloc((void**)&dev_varRR, size)); CudaSafeCall(cudaMalloc((void**)&dev_varRG, size)); CudaSafeCall(cudaMalloc((void**)&dev_varRB, size));
	CudaSafeCall(cudaMalloc((void**)&dev_varGG, size)); CudaSafeCall(cudaMalloc((void**)&dev_varGB, size)); CudaSafeCall(cudaMalloc((void**)&dev_varBB, size));
	CudaSafeCall(cudaMalloc((void**)&dev_covarRCost, size));
	CudaSafeCall(cudaMalloc((void**)&dev_covarGCost, size));
	CudaSafeCall(cudaMalloc((void**)&dev_covarBCost, size));
	CudaSafeCall(cudaMemcpy(dev_varRR, host_varRR, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_varRG, host_varRG, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_varRB, host_varRB, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_varGG, host_varGG, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_varGB, host_varGB, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_varBB, host_varBB, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_covarRCost, host_covarRCost, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_covarGCost, host_covarGCost, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_covarBCost, host_covarBCost, size, cudaMemcpyHostToDevice));

	TimingGPU timer;
	timer.StartCounter();

	aCoeffSpaceKernel << < grid, block >> >(dev_out,
											dev_varRR, dev_varRG, dev_varRB, dev_varGG, dev_varGB, dev_varBB,
											dev_covarRCost, dev_covarGCost, dev_covarBCost,
											width, height, epsilon);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | aCoeffSpaceKernel : " << time << " ms" << std::endl;
	}

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size * 3, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void disparitySelectionWithCuda(float *host_out, const float *host_cost_volume,
								const int width, const int height, const int dispMin, const int dispMax) {

	int dispSize = dispMax - dispMin + 1;

	int blockSize = dispSize;

	if (!ISPOW2(blockSize)) {
		int x = blockSize;
		blockSize = 2;
		while (x >>= 1) blockSize <<= 1;
	}

	dim3 block(1, 1, blockSize);
	dim3 grid(width, height);

	float *dev_out = 0;
	float *dev_cost_volume = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_cost_volume, size * dispSize));
	CudaSafeCall(cudaMemcpy(dev_cost_volume, host_cost_volume, size * dispSize, cudaMemcpyHostToDevice));

	//printf("		disparitySelectionWithCuda : grid = (%d, %d) | block = (%d, %d, %d)\n", grid.x, grid.y, block.x, block.y, block.z);

	TimingGPU timer;
	timer.StartCounter();

	disparitySelectionKernel << < grid, block, block.z * sizeof(float) >> >(dev_out, dev_cost_volume, width, height, dispMin, dispMax);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | disparitySelectionKernel : " << time << " ms" << std::endl;
	}

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}
