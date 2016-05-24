#include "costVolume.h"
#include "image.h"
#include "io_png.h"

#include <iostream>

void costVolumeWithCuda(int *c, const int *a, const int *b, unsigned int size);

static Image covarianceGPGPU(Image im1, Image mean1, Image im2, Image mean2, int r) {
	return (im1*im2).boxFilterGPGPU(r) - mean1*mean2;
}

Image filter_cost_volume_GPGPU(Image im1Color, Image im2Color, int dispMin, int dispMax, const ParamGuidedFilter& param) {
	Image im1R = im1Color.r(), im1G = im1Color.g(), im1B = im1Color.b();
	Image im2R = im2Color.r(), im2G = im2Color.g(), im2B = im2Color.b();
	const int width = im1R.width(), height = im1R.height();
	const int r = param.kernel_radius;
	std::cout << "Cost-volume: " << (dispMax - dispMin + 1) << " disparities. ";

	Image disparity(width, height);
	std::fill_n(&disparity(0, 0), width*height, static_cast<float>(dispMin - 1));
	Image cost(width, height);
	std::fill_n(&cost(0, 0), width*height, std::numeric_limits<float>::max());

	Image im1Gray(width, height);
	Image im2Gray(width, height);
	rgb_to_gray(&im1R(0, 0), &im1G(0, 0), &im1B(0, 0), width, height, &im1Gray(0, 0));
	rgb_to_gray(&im2R(0, 0), &im2G(0, 0), &im2B(0, 0), width, height, &im2Gray(0, 0));
	Image gradient1 = im1Gray.gradXGPGPU();
	Image gradient2 = im2Gray.gradXGPGPU();

	// Compute the mean and variance of each patch, eq. (14)
	Image meanIm1R = im1R.boxFilterGPGPU(r);
	Image meanIm1G = im1G.boxFilterGPGPU(r);
	Image meanIm1B = im1B.boxFilterGPGPU(r);

	Image varIm1RR = covarianceGPGPU(im1R, meanIm1R, im1R, meanIm1R, r);
	Image varIm1RG = covarianceGPGPU(im1R, meanIm1R, im1G, meanIm1G, r);
	Image varIm1RB = covarianceGPGPU(im1R, meanIm1R, im1B, meanIm1B, r);
	Image varIm1GG = covarianceGPGPU(im1G, meanIm1G, im1G, meanIm1G, r);
	Image varIm1GB = covarianceGPGPU(im1G, meanIm1G, im1B, meanIm1B, r);
	Image varIm1BB = covarianceGPGPU(im1B, meanIm1B, im1B, meanIm1B, r);

	return disparity;
}

void costVolumeWithCuda() {
}