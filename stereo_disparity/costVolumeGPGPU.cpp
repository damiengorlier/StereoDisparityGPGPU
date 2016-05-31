#include "costVolume.h"
#include "cudaKernels.cuh"
#include "image.h"
#include "io_png.h"

#include <iostream>

// TODO : mettre ça en paramètre du programme
#define BLOCK 16

void toVector(float *tab, std::vector<Image> &vec, int width, int height, int size) {
	vec.reserve(size);
	for (std::vector<int>::size_type i = 0; i != size; i++) {
		vec.push_back(Image(tab + width * height * i, width, height));
	}
}

void toTab(std::vector<Image> const &vec, float *tab, int width, int height) {
	for (std::vector<int>::size_type i = 0; i != vec.size(); i++) {
		float *tabIm = &(const_cast<Image&>(vec[i]))(0, 0);
		for (int j = 0; j < width * height; j++) {
			tab[j + i * width * height] = tabIm[j];
		}
	}
}

Image covarianceGPGPU(Image im1, Image mean1, Image im2, Image mean2, int r) {
	const int width = im1.width(), height = im1.height();
	float *covTab = new float[width * height];
	float *im1Tab = &(const_cast<Image&>(im1))(0, 0); float *im2Tab = &(const_cast<Image&>(im2))(0, 0);
	float *mean1Tab = &(const_cast<Image&>(mean1))(0, 0); float *mean2Tab = &(const_cast<Image&>(mean2))(0, 0);

	covarianceWithCuda(covTab, im1Tab, mean1Tab, im2Tab, mean2Tab, width, height, BLOCK, r);

	Image covariance(covTab, width, height);
	return covariance;
}

static Image covariance(Image im1, Image mean1, Image im2, Image mean2, int r) {
	return (im1*im2).boxFilter(r) - mean1*mean2;
}

static void compute_costVolume_GPGPU(std::vector<Image> const &image1RGBvec, std::vector<Image> const &image2RGBvec,
									 Image gradient1, Image gradient2,
									 const int dispMin, const int dispMax, const ParamGuidedFilter &param,
									 std::vector<Image> &costVolume) {

	const int width = image1RGBvec[0].width(), height = image1RGBvec[0].height();
	const int dispSize = dispMax - dispMin + 1;
	float *costVolumeTab = new float[width * height * dispSize];
	float *tabIm1R = &(const_cast<Image&>(image1RGBvec[0]))(0, 0); float *tabIm1G = &(const_cast<Image&>(image1RGBvec[1]))(0, 0);
	float *tabIm1B = &(const_cast<Image&>(image1RGBvec[2]))(0, 0); float *tabIm2R = &(const_cast<Image&>(image2RGBvec[0]))(0, 0);
	float *tabIm2G = &(const_cast<Image&>(image2RGBvec[1]))(0, 0); float *tabIm2B = &(const_cast<Image&>(image2RGBvec[2]))(0, 0);
	float *tabGrad1 = &(const_cast<Image&>(gradient1))(0, 0); float *tabGrad2= &(const_cast<Image&>(gradient2))(0, 0);

	costVolumeWithCuda(costVolumeTab, tabIm1R, tabIm1G, tabIm1B, tabIm2R, tabIm2G, tabIm2B, tabGrad1, tabGrad2,
					   width, height, BLOCK, dispMin, dispMax, param.color_threshold, param.gradient_threshold, param.alpha);

	toVector(costVolumeTab, costVolume, width, height, dispSize);
}

static void compute_aCoeffSpace_GPGPU(std::vector<Image> const &varianceVec, std::vector<Image> const &covarVec,
									  const float epsilon, std::vector<Image> &aCoeffSpace) {

	const int width = varianceVec[0].width(); const int height = varianceVec[0].height();
	float *aCoeffSpaceTab = new float[width * height * 3];
	float *tabVarRR = &(const_cast<Image&>(varianceVec[0]))(0, 0); float *tabVarRG = &(const_cast<Image&>(varianceVec[1]))(0, 0);
	float *tabVarRB = &(const_cast<Image&>(varianceVec[2]))(0, 0); float *tabVarGG = &(const_cast<Image&>(varianceVec[3]))(0, 0);
	float *tabVarGB = &(const_cast<Image&>(varianceVec[4]))(0, 0); float *tabVarBB = &(const_cast<Image&>(varianceVec[5]))(0, 0);
	float *tabCovarRCost = &(const_cast<Image&>(covarVec[0]))(0, 0);
	float *tabCovarGCost = &(const_cast<Image&>(covarVec[1]))(0, 0);
	float *tabCovarBCost = &(const_cast<Image&>(covarVec[2]))(0, 0);
	
	aCoeffSpaceWithCuda(aCoeffSpaceTab, tabVarRR, tabVarRG, tabVarRB, tabVarGG, tabVarGB, tabVarBB,
						tabCovarRCost, tabCovarGCost, tabCovarBCost, width, height, BLOCK, epsilon);

	toVector(aCoeffSpaceTab, aCoeffSpace, width, height, 3);
}

// Test purpose
static Image disparity_selection_GPGPU(std::vector<Image> const &costVolume,
									   const int width, const int height, const int dispMin, const int dispMax) {
	
	float *disparityTab = new float[width * height];
	float *costVolumeTab = new float[width * height * costVolume.size()];
	toTab(costVolume, costVolumeTab, width, height);

	disparitySelectionWithCuda(disparityTab, costVolumeTab, width, height, dispMin, dispMax);

	Image disparity(disparityTab, width, height);

	return disparity;
}

// Test purpose
std::vector<Image> cost_volume_CPU_GPGPU(Image im1Color, Image im2Color, int dispMin, int dispMax, const ParamGuidedFilter &param) {
	Image im1R = im1Color.r(), im1G = im1Color.g(), im1B = im1Color.b();
	Image im2R = im2Color.r(), im2G = im2Color.g(), im2B = im2Color.b();
	std::vector<Image> image1RGBvec{ im1R, im1G, im1B };
	std::vector<Image> image2RGBvec{ im2R, im2G, im2B };
	const int width = im1R.width(), height = im1R.height();
	const int r = param.kernel_radius;
	const int dispSize = dispMax - dispMin + 1;
	std::cout << "Cost-volume: " << (dispMax - dispMin + 1) << " disparities." << std::endl;

	Image im1Gray(width, height);
	Image im2Gray(width, height);
	rgb_to_gray(&im1R(0, 0), &im1G(0, 0), &im1B(0, 0), width, height, &im1Gray(0, 0));
	rgb_to_gray(&im2R(0, 0), &im2G(0, 0), &im2B(0, 0), width, height, &im2Gray(0, 0));
	Image gradient1 = im1Gray.gradX();
	Image gradient2 = im2Gray.gradX();

	std::vector<Image> costVolume;
	compute_costVolume_GPGPU(image1RGBvec, image2RGBvec, gradient1, gradient2, dispMin, dispMax, param, costVolume);

	return costVolume;
}

// Test purpose
Image disp_cost_volume_CPU_GPGPU(Image im1Color, Image im2Color, int dispMin, int dispMax, const ParamGuidedFilter &param) {
	Image im1R = im1Color.r(), im1G = im1Color.g(), im1B = im1Color.b();
	Image im2R = im2Color.r(), im2G = im2Color.g(), im2B = im2Color.b();
	std::vector<Image> image1RGBvec{ im1R, im1G, im1B };
	std::vector<Image> image2RGBvec{ im2R, im2G, im2B };
	const int width = im1R.width(), height = im1R.height();
	const int r = param.kernel_radius;
	const int dispSize = dispMax - dispMin + 1;
	std::cout << "Cost-volume: " << (dispMax - dispMin + 1) << " disparities." << std::endl;

	Image im1Gray(width, height);
	Image im2Gray(width, height);
	rgb_to_gray(&im1R(0, 0), &im1G(0, 0), &im1B(0, 0), width, height, &im1Gray(0, 0));
	rgb_to_gray(&im2R(0, 0), &im2G(0, 0), &im2B(0, 0), width, height, &im2Gray(0, 0));
	Image gradient1 = im1Gray.gradX();
	Image gradient2 = im2Gray.gradX();

	std::vector<Image> costVolume;
	compute_costVolume_GPGPU(image1RGBvec, image2RGBvec, gradient1, gradient2, dispMin, dispMax, param, costVolume);

	Image disparity = disparity_selection_GPGPU(costVolume, width, height, dispMin, dispMax);

	return disparity;
}

// Test purpose
Image filter_cost_volume_CPU_GPGPU(Image im1Color, Image im2Color, int dispMin, int dispMax, const ParamGuidedFilter &param) {

	// DECOMPOSE IMAGES TO RGB

	Image im1R = im1Color.r(), im1G = im1Color.g(), im1B = im1Color.b();
	Image im2R = im2Color.r(), im2G = im2Color.g(), im2B = im2Color.b();

	std::vector<Image> image1RGBvec{ im1R, im1G, im1B };
	std::vector<Image> image2RGBvec{ im2R, im2G, im2B };

	// INIT CONSTANTS

	const int width = im1R.width(), height = im1R.height();
	const int r = param.kernel_radius;
	const int dispSize = dispMax - dispMin + 1;

	// COMPUTE INTERMEDIATE VARIABLES

	Image im1Gray(width, height);
	Image im2Gray(width, height);
	rgb_to_gray(&im1R(0, 0), &im1G(0, 0), &im1B(0, 0), width, height, &im1Gray(0, 0));
	rgb_to_gray(&im2R(0, 0), &im2G(0, 0), &im2B(0, 0), width, height, &im2Gray(0, 0));

	Image gradient1 = im1Gray.gradX();
	Image gradient2 = im2Gray.gradX();

	Image meanIm1R = im1R.boxFilter(r);
	Image meanIm1G = im1G.boxFilter(r);
	Image meanIm1B = im1B.boxFilter(r);

	Image varIm1RR = covariance(im1R, meanIm1R, im1R, meanIm1R, r);
	Image varIm1RG = covariance(im1R, meanIm1R, im1G, meanIm1G, r);
	Image varIm1RB = covariance(im1R, meanIm1R, im1B, meanIm1B, r);
	Image varIm1GG = covariance(im1G, meanIm1G, im1G, meanIm1G, r);
	Image varIm1GB = covariance(im1G, meanIm1G, im1B, meanIm1B, r);
	Image varIm1BB = covariance(im1B, meanIm1B, im1B, meanIm1B, r);

	std::vector<Image> varianceVec{ varIm1RR, varIm1RG, varIm1RB, varIm1GG, varIm1GB, varIm1BB };

	// COMPUTE COST VOLUME

	std::vector<Image> costVolume;
	compute_costVolume_GPGPU(image1RGBvec, image2RGBvec, gradient1, gradient2, dispMin, dispMax, param, costVolume);

	// GUIDED FILTER

	std::vector<Image> filteredCostVolume;
	filteredCostVolume.reserve(dispSize);

	for (std::vector<int>::size_type i = 0; i != costVolume.size(); i++) {
		std::cout << '*' << std::flush;
		Image dCost = costVolume[i];
		Image meanCost = dCost.boxFilter(r);

		Image covarIm1RCost = covariance(im1R, meanIm1R, dCost, meanCost, r);
		Image covarIm1GCost = covariance(im1G, meanIm1G, dCost, meanCost, r);
		Image covarIm1BCost = covariance(im1B, meanIm1B, dCost, meanCost, r);

		std::vector<Image> covarVec{ covarIm1RCost, covarIm1GCost, covarIm1BCost };

		std::vector<Image> aCoeffSpace;
		compute_aCoeffSpace_GPGPU(varianceVec, covarVec, param.epsilon, aCoeffSpace);

		Image b = (meanCost - aCoeffSpace[0] * meanIm1R - aCoeffSpace[1] * meanIm1G - aCoeffSpace[2] * meanIm1B).boxFilter(r);

		filteredCostVolume.push_back(b + aCoeffSpace[0].boxFilter(r) * im1R + aCoeffSpace[1].boxFilter(r) * im1G + aCoeffSpace[2].boxFilter(r) * im1B);
	}

	// DISPARITY SELECTION

	Image disparity = disparity_selection_GPGPU(filteredCostVolume, width, height, dispMin, dispMax);

	std::cout << std::endl;
	return disparity;
}

Image filter_cost_volume_GPGPU(Image im1Color, Image im2Color, int dispMin, int dispMax, const ParamGuidedFilter &param, int blockDim) {
	Image im1R = im1Color.r(), im1G = im1Color.g(), im1B = im1Color.b();
	Image im2R = im2Color.r(), im2G = im2Color.g(), im2B = im2Color.b();

	const int width = im1R.width(), height = im1R.height();
	const int dispSize = dispMax - dispMin + 1;
	std::cout << "Cost-volume with GPGPU: " << dispSize << " disparities." << std::endl;

	Image disparity(width, height);
	float *dispTab = &(const_cast<Image&>(disparity))(0, 0);
	float *im1RTab = &(const_cast<Image&>(im1R))(0, 0); float *im1GTab = &(const_cast<Image&>(im1G))(0, 0); float *im1BTab = &(const_cast<Image&>(im1B))(0, 0);
	float *im2RTab = &(const_cast<Image&>(im2R))(0, 0); float *im2GTab = &(const_cast<Image&>(im2G))(0, 0); float *im2BTab = &(const_cast<Image&>(im2B))(0, 0);
	computeDisparityMapWithCuda(dispTab, im1RTab, im1GTab, im1BTab, im2RTab, im2GTab, im2BTab, width, height, blockDim,
		dispMin, dispMax, param.color_threshold, param.gradient_threshold, param.alpha, param.kernel_radius, param.epsilon);

	return disparity;
}