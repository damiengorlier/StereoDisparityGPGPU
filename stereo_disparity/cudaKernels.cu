//#ifndef __CUDACC__  
//	#define __CUDACC__
//#endif

#include "cudaKernels.cuh"
#include "cudaErrorCheck.cuh"
#include "TimingGPU.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TIMING 1
#define MAX_FLOAT 3.402823466e+38f
#define SYNC 1

#define ISPOW2(x) ((x) > 0 && !((x) & (x-1)))

// ######################################
// #          KERNEL FUNCTIONS          #
// ######################################

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

__global__ void copyKernel(float *dev_out, const float *dev_in, const int width, const int height) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < width) && (j < height)) {
		const int globalIdx = i + j * width;

		dev_out[globalIdx] = dev_in[globalIdx];
	}
}

__global__ void rgbToGrayKernel(float *dev_out, const float *dev_imR, const float *dev_imG, const float *dev_imB, const int width, const int height) {
	// Y = (6969 * R + 23434 * G + 2365 * B) / 32768

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < width) && (j < height)) {
		const int globalIdx = i + j * width;

		dev_out[globalIdx] = (float)(6969 * dev_imR[globalIdx] + 23434 * dev_imG[globalIdx] + 2365 * dev_imB[globalIdx]) / 32768;
	}
}

__global__ void operatorKernel(float *dev_out, const float *dev_in1, const float *dev_in2, const int width, const int height, const int op) {
	//
	// op : 0 - PLUS
	//      1 - MINUS
	//      2 - MULTIPLY
	//

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < width) && (j < height)) {

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
	// Problème général lié à la taille du block (gradX et bloxFilter) : un block est limité à 1024 ou 512 threads en général.
	// Augmenter la taille du block standard à 32x32 = 1024 va créer un block (32+2)x32 pour le gradX et un encore plus grand pour le boxFilter.
	// On est donc limité à un block 16x16, ce qui limite aussi la taille du rayon du boxFilter.

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
	// Problème général lié à la taille du block (gradX et bloxFilter) : un block est limité à 1024 ou 512 threads en général.
	// Augmenter la taille du block standard à 32x32 = 1024 va créer un block (32+2)x32 pour le gradX et un encore plus grand pour le boxFilter.
	// On est donc limité à un block 16x16, ce qui limite aussi la taille du rayon du boxFilter.

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

	// In sharedMemory, the cost values are saved from 0 to blockDim.z - 1 and the disparity values are saved from blockDim.z to 2 * blockDim.z - 1
	extern __shared__ float sharedMemory[];

	int i = blockIdx.x;
	int j = blockIdx.y;
	int k = threadIdx.z;

	if (i < width && j < height) {

		const int globalIdx = i + j * width + k * height * width;

		if (k + dispMin <= dispMax) {
			sharedMemory[k] = dev_cost_volume[globalIdx];
			sharedMemory[k + blockDim.z] = k + dispMin;
		}
		else {
			sharedMemory[k] = MAX_FLOAT;
			sharedMemory[k + blockDim.z] = 404;
		}

		__syncthreads();

		int offset = 1;

		for (int d = blockDim.z >> 1; d > 0; d >>= 1)
		{
			if (k < d)
			{
				int ai = offset * (2 * k + 1) - 1;
				int bi = offset * (2 * k + 2) - 1;
				// <= pour avoir le même comportement que dans la version CPU
				// si seulement <, on peut avoir des différences entre les disparity maps si les coûts sont =
				bool isSmaller = sharedMemory[bi] <= sharedMemory[ai];
				sharedMemory[bi] = isSmaller ? sharedMemory[bi] : sharedMemory[ai];
				sharedMemory[bi + blockDim.z] = isSmaller ? sharedMemory[bi + blockDim.z] : sharedMemory[ai + blockDim.z];
			}
			offset *= 2;
			__syncthreads();
		}

		if (k == 0) {
			dev_out[i + j * width] = sharedMemory[2 * blockDim.z - 1];
		}
	}
}

__global__ void detectOcclusionKernel(float *dev_out, const float *dev_dispLeft, const float *dev_dispRight,
									  const int width, const int height, const float dOcclusion, const int tolDisp) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < width && j < height) {
		int globalIdx = i + j * width;

		int d = (int)dev_dispLeft[globalIdx];

		bool occlusion = (i + d < 0 || i + d >= width);

		if (!occlusion) {
			int d2 = d + (int)dev_dispRight[globalIdx + d];
			if (d2 < 0) d2 = -d2;
			occlusion = d2 > tolDisp;
		}

		if (occlusion) {
			dev_out[globalIdx] = dOcclusion;
		}
		else {
			dev_out[globalIdx] = (float)d;
		}
	}
}

// ###################################
// #          DEV FUNCTIONS          #
// ###################################

void copyWithCudaDev(float *dev_out, const float *dev_in, const int width, const int height, const int blockDim) {

	dim3 block(blockDim, blockDim);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim));

	copyKernel << < grid, block >> > (dev_out, dev_in, width, height);
	CudaCheckError();
	if (SYNC) CudaSafeCall(cudaDeviceSynchronize());
}

void rgbToGrayWithCudaDev(float *dev_out, const float *dev_imR, const float *dev_imG, const float *dev_imB,
						  const int width, const int height, const int blockDim) {
	dim3 block(blockDim, blockDim);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim));

	rgbToGrayKernel << < grid, block >> > (dev_out, dev_imR, dev_imG, dev_imB, width, height);
	CudaCheckError();
	if (SYNC) CudaSafeCall(cudaDeviceSynchronize());
}

void operatorWithCudaDev(float *dev_out, const float *dev_in1, const float *dev_in2, const int width, const int height, const int blockDim, const int op) {

	dim3 block(blockDim, blockDim);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim));

	operatorKernel << < grid, block >> >(dev_out, dev_in1, dev_in2, width, height, op);
	CudaCheckError();
	if (SYNC) CudaSafeCall(cudaDeviceSynchronize());
}

void gradXWithCudaDev(float *dev_out, const float *dev_in, const int width, const int height, const int blockDim) {

	dim3 block(blockDim + 2, blockDim);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim));

	gradXKernel << < grid, block, block.x * block.y * sizeof(float) >> >(dev_out, dev_in, width, height);
	CudaCheckError();
	if (SYNC) CudaSafeCall(cudaDeviceSynchronize());
}

void scanWithCudaDev(float *dev_out, const float *dev_in, const int width, const int height, const int blockDim, const bool addOri) {
	int blockSize = width;

	if (!ISPOW2(blockSize)) {
		int x = blockSize;
		blockSize = 2;
		while (x >>= 1) blockSize <<= 1;
	}

	dim3 block(blockSize, 1);
	dim3 grid(1, height);

	float *dev_tmp = 0;
	int size = width * height * sizeof(float);
	CudaSafeCall(cudaMalloc((void**)&dev_tmp, size));

	scanKernel << <grid, block, block.x * block.y * sizeof(float) >> >(dev_tmp, dev_in, block.x, width, height);
	CudaCheckError();
	if (SYNC) CudaSafeCall(cudaDeviceSynchronize());

	if (addOri) {
		operatorWithCudaDev(dev_out, dev_in, dev_tmp, width, height, blockDim, 0);
	}
	else {
		copyWithCudaDev(dev_out, dev_tmp, width, height, blockDim);
	}
}

void transposeWithCudaDev(float *dev_out, const float *dev_in, const int width, const int height, const int blockDim) {
	int a = (width > height) ? width : height;
	dim3 block(blockDim, blockDim);
	dim3 grid(ceil(a / blockDim), ceil(a / blockDim));

	transposeKernel << < grid, block, block.x * block.y * sizeof(float) >> >(dev_out, dev_in, width, height);
	CudaCheckError();
	if (SYNC) CudaSafeCall(cudaDeviceSynchronize());
}

void integralWithCudaDev(float *dev_out, const float *dev_in, const int width, const int height, const int blockDim) {

	float *dev_tmp1 = 0; float *dev_tmp2 = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaMalloc((void**)&dev_tmp1, size)); CudaSafeCall(cudaMalloc((void**)&dev_tmp2, size));

	scanWithCudaDev(dev_tmp1, dev_in, width, height, blockDim, true);
	transposeWithCudaDev(dev_tmp2, dev_tmp1, width, height, blockDim);
	scanWithCudaDev(dev_tmp1, dev_tmp2, height, width, blockDim, true);
	transposeWithCudaDev(dev_out, dev_tmp1, height, width, blockDim);

	CudaSafeCall(cudaFree(dev_tmp1)); CudaSafeCall(cudaFree(dev_tmp2));
}

void boxFilterWithCudaDev(float *dev_out, const float *dev_in, const int width, const int height, const int blockDim, int radius) {

	float *dev_integral = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaMalloc((void**)&dev_integral, size));

	integralWithCudaDev(dev_integral, dev_in, width, height, blockDim);

	dim3 block(blockDim + 2 * radius + 1, blockDim + 2 * radius + 1);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim));

	boxFilterKernel << < grid, block, block.x * block.y * sizeof(float) >> >(dev_out, dev_integral, radius, width, height);
	CudaCheckError();
	if (SYNC) CudaSafeCall(cudaDeviceSynchronize());

	CudaSafeCall(cudaFree(dev_integral));
}

void covarianceWithCudaDev(float *dev_out, const float *dev_im1, const float *dev_mean1, const float *dev_im2, const float *dev_mean2,
						   const int width, const int height, const int blockDim, int radius) {

	float *dev_im1xim2 = 0; float *dev_mean1xmean2 = 0; float *dev_meanIm1xim2 = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaMalloc((void**)&dev_im1xim2, size));
	CudaSafeCall(cudaMalloc((void**)&dev_mean1xmean2, size));
	CudaSafeCall(cudaMalloc((void**)&dev_meanIm1xim2, size));

	operatorWithCudaDev(dev_im1xim2, dev_im1, dev_im2, width, height, blockDim, 2);
	operatorWithCudaDev(dev_mean1xmean2, dev_mean1, dev_mean2, width, height, blockDim, 2);
	boxFilterWithCudaDev(dev_meanIm1xim2, dev_im1xim2, width, height, blockDim, radius);
	operatorWithCudaDev(dev_out, dev_meanIm1xim2, dev_mean1xmean2, width, height, blockDim, 1);

	CudaSafeCall(cudaFree(dev_im1xim2)); CudaSafeCall(cudaFree(dev_mean1xmean2)); CudaSafeCall(cudaFree(dev_meanIm1xim2));
}

void costVolumeWithCudaDev(float *dev_out,
						   const float *dev_im1R, const float *dev_im1G, const float *dev_im1B,
						   const float *dev_im2R, const float *dev_im2G, const float *dev_im2B,
						   const float *dev_grad1, float *dev_grad2,
						   const int width, const int height, const int blockDim,
						   const int dispMin, const int dispMax, const float colorTh, const float gradTh, const float alpha) {

	int dispSize = dispMax - dispMin + 1;
	dim3 block(blockDim, blockDim);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim), dispSize);

	costVolumeKernel << < grid, block >> >(dev_out,
										   dev_im1R, dev_im1G, dev_im1B,
										   dev_im2R, dev_im2G, dev_im2B,
										   dev_grad1, dev_grad2,
										   width, height, dispMin, dispMax, colorTh, gradTh, alpha);
	CudaCheckError();
	if (SYNC) CudaSafeCall(cudaDeviceSynchronize());
}

void aCoeffSpaceWithCudaDev(float *dev_out,
							const float *dev_varRR, const float *dev_varRG, const float *dev_varRB,
							const float *dev_varGG, const float *dev_varGB, const float *dev_varBB,
							const float *dev_covarRCost, const float *dev_covarGCost, const float *dev_covarBCost,
							const int width, const int height, const int blockDim,
							const float epsilon) {
	dim3 block(blockDim, blockDim);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim));

	aCoeffSpaceKernel << < grid, block >> >(dev_out,
											dev_varRR, dev_varRG, dev_varRB, dev_varGG, dev_varGB, dev_varBB,
											dev_covarRCost, dev_covarGCost, dev_covarBCost,
											width, height, epsilon);
	CudaCheckError();
	if (SYNC) CudaSafeCall(cudaDeviceSynchronize());
}

void bCoeffWithCudaDev(float *dev_out, const float *dev_meanCost,
					   const float *dev_mean1R, const float *dev_mean1G, const float *dev_mean1B,
					   const float *dev_aCoeffR, const float *dev_aCoeffG, const float *dev_aCoeffB,
					   const int width, const int height, const int blockDim, const int radius) {

	float *dev_aCoeffxmeanR = 0; float *dev_aCoeffxmeanG = 0; float *dev_aCoeffxmeanB = 0;
	float *dev_tmp1 = 0; float *dev_tmp2 = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaMalloc((void**)&dev_aCoeffxmeanR, size));
	CudaSafeCall(cudaMalloc((void**)&dev_aCoeffxmeanG, size));
	CudaSafeCall(cudaMalloc((void**)&dev_aCoeffxmeanB, size));
	CudaSafeCall(cudaMalloc((void**)&dev_tmp1, size)); CudaSafeCall(cudaMalloc((void**)&dev_tmp2, size));

	operatorWithCudaDev(dev_aCoeffxmeanR, dev_aCoeffR, dev_mean1R, width, height, blockDim, 2);
	operatorWithCudaDev(dev_aCoeffxmeanG, dev_aCoeffG, dev_mean1G, width, height, blockDim, 2);
	operatorWithCudaDev(dev_aCoeffxmeanB, dev_aCoeffB, dev_mean1B, width, height, blockDim, 2);

	operatorWithCudaDev(dev_tmp1, dev_meanCost, dev_aCoeffxmeanR, width, height, blockDim, 1);
	operatorWithCudaDev(dev_tmp2, dev_tmp1, dev_aCoeffxmeanG, width, height, blockDim, 1);
	operatorWithCudaDev(dev_tmp1, dev_tmp2, dev_aCoeffxmeanB, width, height, blockDim, 1);

	boxFilterWithCudaDev(dev_out, dev_tmp1, width, height, blockDim, radius);

	CudaSafeCall(cudaFree(dev_aCoeffxmeanR)); CudaSafeCall(cudaFree(dev_aCoeffxmeanG)); CudaSafeCall(cudaFree(dev_aCoeffxmeanB));
	CudaSafeCall(cudaFree(dev_tmp1)); CudaSafeCall(cudaFree(dev_tmp2));
}

void filteredCostWithCudaDev(float *dev_out, const float *dev_bCoeff,
							 const float *dev_aCoeffR, const float *dev_aCoeffG, const float *dev_aCoeffB,
							 const float *dev_im1R, const float *dev_im1G, const float *dev_im1B,
							 const int width, const int height, const int blockDim, const int radius) {

	float *dev_meanACoeffR = 0; float *dev_meanACoeffG = 0; float *dev_meanACoeffB = 0;
	float *dev_aMxImR = 0; float *dev_aMxImG = 0; float *dev_aMxImB = 0;
	float *dev_tmp1 = 0; float *dev_tmp2 = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaMalloc((void**)&dev_meanACoeffR, size));
	CudaSafeCall(cudaMalloc((void**)&dev_meanACoeffG, size));
	CudaSafeCall(cudaMalloc((void**)&dev_meanACoeffB, size));
	CudaSafeCall(cudaMalloc((void**)&dev_aMxImR, size));
	CudaSafeCall(cudaMalloc((void**)&dev_aMxImG, size));
	CudaSafeCall(cudaMalloc((void**)&dev_aMxImB, size));
	CudaSafeCall(cudaMalloc((void**)&dev_tmp1, size)); CudaSafeCall(cudaMalloc((void**)&dev_tmp2, size));

	boxFilterWithCudaDev(dev_meanACoeffR, dev_aCoeffR, width, height, blockDim, radius);
	boxFilterWithCudaDev(dev_meanACoeffG, dev_aCoeffG, width, height, blockDim, radius);
	boxFilterWithCudaDev(dev_meanACoeffB, dev_aCoeffB, width, height, blockDim, radius);

	operatorWithCudaDev(dev_aMxImR, dev_meanACoeffR, dev_im1R, width, height, blockDim, 2);
	operatorWithCudaDev(dev_aMxImG, dev_meanACoeffG, dev_im1G, width, height, blockDim, 2);
	operatorWithCudaDev(dev_aMxImB, dev_meanACoeffB, dev_im1B, width, height, blockDim, 2);

	operatorWithCudaDev(dev_tmp1, dev_aMxImR, dev_aMxImG, width, height, blockDim, 0);
	operatorWithCudaDev(dev_tmp2, dev_tmp1, dev_aMxImB, width, height, blockDim, 0);
	operatorWithCudaDev(dev_out, dev_bCoeff, dev_tmp2, width, height, blockDim, 0);

	CudaSafeCall(cudaFree(dev_meanACoeffR)); CudaSafeCall(cudaFree(dev_meanACoeffG)); CudaSafeCall(cudaFree(dev_meanACoeffB));
	CudaSafeCall(cudaFree(dev_aMxImR)); CudaSafeCall(cudaFree(dev_aMxImG)); CudaSafeCall(cudaFree(dev_aMxImB));
	CudaSafeCall(cudaFree(dev_tmp1)); CudaSafeCall(cudaFree(dev_tmp2));
}

void disparitySelectionWithCudaDev(float *dev_out, const float *dev_costVolume,
								   const int width, const int height, const int dispMin, const int dispMax) {

	int blockSize = dispMax - dispMin + 1;

	if (!ISPOW2(blockSize)) {
		int x = blockSize;
		blockSize = 2;
		while (x >>= 1) blockSize <<= 1;
	}

	dim3 block(1, 1, blockSize);
	dim3 grid(width, height);

	disparitySelectionKernel << < grid, block, 2 * block.z * sizeof(float) >> >(dev_out, dev_costVolume, width, height, dispMin, dispMax);
	CudaCheckError();
	if (SYNC) CudaSafeCall(cudaDeviceSynchronize());
}

void detectOcclusionWithCudaDev(float *dev_out,
								const float *dev_dispLeft, const float *dev_dispRight,
								const int width, const int height, const int blockDim,
								const float dOcclusion, const int tolDisp) {

	dim3 block(blockDim, blockDim);
	dim3 grid(ceil(width / blockDim), ceil(height / blockDim));

	detectOcclusionKernel << < grid, block >> >(dev_out, dev_dispLeft, dev_dispRight, width, height, dOcclusion, tolDisp);
	CudaCheckError();
	if (SYNC) CudaSafeCall(cudaDeviceSynchronize());
}

// ####################################
// #          HOST FUNCTIONS          #
// ####################################

void rgbToGrayWithCuda(float *host_out, const float *host_imR, const float *host_imG, const float *host_imB,
					   const int width, const int height, const int blockDim) {

	float *dev_out = 0;
	float *dev_imR = 0; float *dev_imG = 0; float *dev_imB = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_imR, size)); CudaSafeCall(cudaMalloc((void**)&dev_imG, size)); CudaSafeCall(cudaMalloc((void**)&dev_imB, size));
	CudaSafeCall(cudaMemcpy(dev_imR, host_imR, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_imG, host_imG, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_imB, host_imB, size, cudaMemcpyHostToDevice));

	TimingGPU timer;
	timer.StartCounter();

	rgbToGrayWithCudaDev(dev_out, dev_imR, dev_imG, dev_imB, width, height, blockDim);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | rgbToGrayKernel : " << time << " ms" << std::endl;
	}

	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void operatorWithCuda(float *host_out, const float *host_in1, const float *host_in2, const int width, const int height, const int blockDim, const int op) {

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

	operatorWithCudaDev(dev_out, dev_in1, dev_in2, width, height, blockDim, op);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | operatorKernel (op=" << op << ") : " << time << " ms" << std::endl;
	}

	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void gradXWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim) {

	float *dev_in = 0;
	float *dev_out = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	TimingGPU timer;
	timer.StartCounter();

	gradXWithCudaDev(dev_out, dev_in, width, height, blockDim);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | gradXKernel : " << time << " ms" << std::endl;
	}

	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void scanWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim, const bool addOri) {

	float *dev_in = 0;
	float *dev_out = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	TimingGPU timer;
	timer.StartCounter();

	scanWithCudaDev(dev_out, dev_in, width, height, blockDim, addOri);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | scanKernel : " << time << " ms" << std::endl;
	}

	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void transposeWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim) {

	float *dev_in = 0;
	float *dev_out = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	TimingGPU timer;
	timer.StartCounter();

	transposeWithCudaDev(dev_out, dev_in, width, height, blockDim);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | transposeKernel : " << time << " ms" << std::endl;
	}

	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void integralWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim) {

	float *dev_in = 0;
	float *dev_out = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	TimingGPU timer;
	timer.StartCounter();

	integralWithCudaDev(dev_out, dev_in, width, height, blockDim);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | integralKernel : " << time << " ms" << std::endl;
	}

	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void boxFilterWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim, int radius) {

	float *dev_in = 0;
	float *dev_out = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_in, size));
	CudaSafeCall(cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice));

	TimingGPU timer;
	timer.StartCounter();

	boxFilterWithCudaDev(dev_out, dev_in, width, height, blockDim, radius);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | boxFilterKernel (r=" << radius << ") : " << time << " ms" << std::endl;
	}

	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void covarianceWithCuda(float *host_out, const float *host_im1, const float *host_mean1, const float *host_im2, const float *host_mean2,
						const int width, const int height, const int blockDim, int radius) {

	float *dev_out = 0;
	float *dev_im1 = 0; float *dev_im2 = 0; float *dev_mean1 = 0; float *dev_mean2 = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_im1, size)); CudaSafeCall(cudaMalloc((void**)&dev_im2, size));
	CudaSafeCall(cudaMalloc((void**)&dev_mean1, size)); CudaSafeCall(cudaMalloc((void**)&dev_mean2, size));
	CudaSafeCall(cudaMemcpy(dev_im1, host_im1, size, cudaMemcpyHostToDevice)); CudaSafeCall(cudaMemcpy(dev_im2, host_im2, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_mean1, host_mean1, size, cudaMemcpyHostToDevice)); CudaSafeCall(cudaMemcpy(dev_mean2, host_mean2, size, cudaMemcpyHostToDevice));

	TimingGPU timer;
	timer.StartCounter();

	covarianceWithCudaDev(dev_out, dev_im1, dev_mean1, dev_im2, dev_mean2, width, height, blockDim, radius);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | covarianceKernel : " << time << " ms" << std::endl;
	}

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

	TimingGPU timer;
	timer.StartCounter();

	disparitySelectionKernel << < grid, block, 2 * block.z * sizeof(float) >> >(dev_out, dev_cost_volume, width, height, dispMin, dispMax);

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | disparitySelectionKernel : " << time << " ms" << std::endl;
	}

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaDeviceReset());
}

void computeDisparityMapWithCuda(float *host_out,
								 const float *host_im1R, const float *host_im1G, const float *host_im1B,
								 const float *host_im2R, const float *host_im2G, const float *host_im2B,
								 const int width, const int height, const int blockDim,
								 const int dispMin, const int dispMax,
								 const float colorTh, const float gradTh, const float alpha,
								 const int radius, const float epsilon) {

	int dispSize = dispMax - dispMin + 1;
	int size = width * height * sizeof(float);

	float *dev_im1R = 0; float *dev_im1G = 0; float *dev_im1B = 0;
	float *dev_im2R = 0; float *dev_im2G = 0; float *dev_im2B = 0;
	float *dev_im1Gray = 0; float *dev_im2Gray = 0;
	float *dev_grad1 = 0; float *dev_grad2 = 0;
	float *dev_mean1R = 0; float *dev_mean1G = 0; float *dev_mean1B = 0;
	float *dev_cov1RR = 0; float *dev_cov1RG = 0; float *dev_cov1RB = 0; float *dev_cov1GG = 0; float *dev_cov1GB = 0; float *dev_cov1BB = 0;
	float *dev_costVolume = 0;
	float *dev_dCost = 0; float *dev_meanCost = 0; float *dev_cov1RCost = 0; float *dev_cov1GCost = 0; float *dev_cov1BCost = 0;
	float *dev_aCoeff = 0; float *dev_aCoeffR = 0; float *dev_aCoeffG = 0; float *dev_aCoeffB = 0;
	float *dev_bCoeff = 0; float *dev_fCost = 0; float *dev_fCostVolume = 0;
	float *dev_out = 0;

	CudaSafeCall(cudaSetDevice(0));
	TimingGPU timer;
	timer.StartCounter();

	// RGB
	CudaSafeCall(cudaMalloc((void**)&dev_im1R, size)); CudaSafeCall(cudaMalloc((void**)&dev_im1G, size)); CudaSafeCall(cudaMalloc((void**)&dev_im1B, size));
	CudaSafeCall(cudaMalloc((void**)&dev_im2R, size)); CudaSafeCall(cudaMalloc((void**)&dev_im2G, size)); CudaSafeCall(cudaMalloc((void**)&dev_im2B, size));

	CudaSafeCall(cudaMemcpy(dev_im1R, host_im1R, size, cudaMemcpyHostToDevice)); CudaSafeCall(cudaMemcpy(dev_im1G, host_im1G, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_im1B, host_im1B, size, cudaMemcpyHostToDevice)); CudaSafeCall(cudaMemcpy(dev_im2R, host_im2R, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_im2G, host_im2G, size, cudaMemcpyHostToDevice)); CudaSafeCall(cudaMemcpy(dev_im2B, host_im2B, size, cudaMemcpyHostToDevice));

	// Gray
	CudaSafeCall(cudaMalloc((void**)&dev_im1Gray, size)); CudaSafeCall(cudaMalloc((void**)&dev_im2Gray, size));

	rgbToGrayWithCudaDev(dev_im1Gray, dev_im1R, dev_im1G, dev_im1B, width, height, blockDim);
	rgbToGrayWithCudaDev(dev_im2Gray, dev_im2R, dev_im2G, dev_im2B, width, height, blockDim);

	// Gradients
	CudaSafeCall(cudaMalloc((void**)&dev_grad1, size)); CudaSafeCall(cudaMalloc((void**)&dev_grad2, size));

	gradXWithCudaDev(dev_grad1, dev_im1Gray, width, height, blockDim);
	gradXWithCudaDev(dev_grad2, dev_im2Gray, width, height, blockDim);

	// Means
	CudaSafeCall(cudaMalloc((void**)&dev_mean1R, size)); CudaSafeCall(cudaMalloc((void**)&dev_mean1G, size)); CudaSafeCall(cudaMalloc((void**)&dev_mean1B, size));

	boxFilterWithCudaDev(dev_mean1R, dev_im1R, width, height, blockDim, radius);
	boxFilterWithCudaDev(dev_mean1G, dev_im1G, width, height, blockDim, radius);
	boxFilterWithCudaDev(dev_mean1B, dev_im1B, width, height, blockDim, radius);

	// Covariances
	CudaSafeCall(cudaMalloc((void**)&dev_cov1RR, size)); CudaSafeCall(cudaMalloc((void**)&dev_cov1RG, size)); CudaSafeCall(cudaMalloc((void**)&dev_cov1RB, size));
	CudaSafeCall(cudaMalloc((void**)&dev_cov1GG, size)); CudaSafeCall(cudaMalloc((void**)&dev_cov1GB, size)); CudaSafeCall(cudaMalloc((void**)&dev_cov1BB, size));

	covarianceWithCudaDev(dev_cov1RR, dev_im1R, dev_mean1R, dev_im1R, dev_mean1R, width, height, blockDim, radius);
	covarianceWithCudaDev(dev_cov1RG, dev_im1R, dev_mean1R, dev_im1G, dev_mean1G, width, height, blockDim, radius);
	covarianceWithCudaDev(dev_cov1RB, dev_im1R, dev_mean1R, dev_im1B, dev_mean1B, width, height, blockDim, radius);
	covarianceWithCudaDev(dev_cov1GG, dev_im1G, dev_mean1G, dev_im1G, dev_mean1G, width, height, blockDim, radius);
	covarianceWithCudaDev(dev_cov1GB, dev_im1G, dev_mean1G, dev_im1B, dev_mean1B, width, height, blockDim, radius);
	covarianceWithCudaDev(dev_cov1BB, dev_im1B, dev_mean1B, dev_im1B, dev_mean1B, width, height, blockDim, radius);

	// Cost Volume
	CudaSafeCall(cudaMalloc((void**)&dev_costVolume, size * dispSize));

	costVolumeWithCudaDev(dev_costVolume,
						  dev_im1R, dev_im1G, dev_im1B,
						  dev_im2R, dev_im2G, dev_im2B,
						  dev_grad1, dev_grad2,
						  width, height, blockDim, dispMin, dispMax, colorTh, gradTh, alpha);

	// Guided Filter
	CudaSafeCall(cudaMalloc((void**)&dev_dCost, size)); CudaSafeCall(cudaMalloc((void**)&dev_meanCost, size));
	CudaSafeCall(cudaMalloc((void**)&dev_cov1RCost, size)); CudaSafeCall(cudaMalloc((void**)&dev_cov1GCost, size)); CudaSafeCall(cudaMalloc((void**)&dev_cov1BCost, size));
	CudaSafeCall(cudaMalloc((void**)&dev_aCoeff, size * 3));
	CudaSafeCall(cudaMalloc((void**)&dev_aCoeffR, size)); CudaSafeCall(cudaMalloc((void**)&dev_aCoeffG, size)); CudaSafeCall(cudaMalloc((void**)&dev_aCoeffB, size));
	CudaSafeCall(cudaMalloc((void**)&dev_bCoeff, size)); CudaSafeCall(cudaMalloc((void**)&dev_fCost, size));
	CudaSafeCall(cudaMalloc((void**)&dev_fCostVolume, size * dispSize));

	for (int i = 0; i < dispSize; i++) {
		copyWithCudaDev(dev_dCost, &dev_costVolume[i * width * height], width, height, blockDim);
		boxFilterWithCudaDev(dev_meanCost, dev_dCost, width, height, blockDim, radius);
		covarianceWithCudaDev(dev_cov1RCost, dev_im1R, dev_mean1R, dev_dCost, dev_meanCost, width, height, blockDim, radius);
		covarianceWithCudaDev(dev_cov1GCost, dev_im1G, dev_mean1G, dev_dCost, dev_meanCost, width, height, blockDim, radius);
		covarianceWithCudaDev(dev_cov1BCost, dev_im1B, dev_mean1B, dev_dCost, dev_meanCost, width, height, blockDim, radius);

		aCoeffSpaceWithCudaDev(dev_aCoeff, dev_cov1RR, dev_cov1RG, dev_cov1RB, dev_cov1GG, dev_cov1GB, dev_cov1BB,
							   dev_cov1RCost, dev_cov1GCost, dev_cov1BCost, width, height, blockDim, epsilon);

		copyWithCudaDev(dev_aCoeffR, &dev_aCoeff[0], width, height, blockDim);
		copyWithCudaDev(dev_aCoeffG, &dev_aCoeff[width * height], width, height, blockDim);
		copyWithCudaDev(dev_aCoeffB, &dev_aCoeff[2 * width * height], width, height, blockDim);

		bCoeffWithCudaDev(dev_bCoeff, dev_meanCost, dev_mean1R, dev_mean1G, dev_mean1B, dev_aCoeffR, dev_aCoeffG, dev_aCoeffB,
						  width, height, blockDim, radius);

		filteredCostWithCudaDev(dev_fCost, dev_bCoeff, dev_aCoeffR, dev_aCoeffG, dev_aCoeffB, dev_im1R, dev_im1G, dev_im1B,
								width, height, blockDim, radius);

		copyWithCudaDev(&dev_fCostVolume[i * width * height], dev_fCost, width, height, blockDim);
	}

	// Disparity Selection
	CudaSafeCall(cudaMalloc((void**)&dev_out, size));

	disparitySelectionWithCudaDev(dev_out, dev_fCostVolume, width, height, dispMin, dispMax);
	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | computeDisparityMapWithCuda : " << time << " ms" << std::endl;
	}

	CudaSafeCall(cudaDeviceReset());
}

void detectOcclusionWithCuda(float *host_out,
							 const float *host_dispLeft, const float *host_dispRight,
							 const int width, const int height, const int blockDim,
							 const float dOcclusion, const int tolDisp) {

	float *dev_out = 0;
	float *dev_dispLeft = 0; float *dev_dispRight = 0;
	int size = width * height * sizeof(float);

	CudaSafeCall(cudaSetDevice(0));
	TimingGPU timer;
	timer.StartCounter();

	CudaSafeCall(cudaMalloc((void**)&dev_out, size));
	CudaSafeCall(cudaMalloc((void**)&dev_dispLeft, size)); CudaSafeCall(cudaMalloc((void**)&dev_dispRight, size));

	CudaSafeCall(cudaMemcpy(dev_dispLeft, host_dispLeft, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_dispRight, host_dispRight, size, cudaMemcpyHostToDevice));

	detectOcclusionWithCudaDev(dev_out, dev_dispLeft, dev_dispRight, width, height, blockDim, dOcclusion, tolDisp);

	CudaSafeCall(cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost));

	float time = timer.GetCounter();
	if (TIMING) {
		std::cout << "GPU | detectOcclusionWithCuda : " << time << " ms" << std::endl;
	}

	CudaSafeCall(cudaDeviceReset());
}
