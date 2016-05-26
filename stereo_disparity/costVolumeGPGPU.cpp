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

// TODO : A refaire en un seul appel au GPU ?
static Image covarianceGPGPU(Image im1, Image mean1, Image im2, Image mean2, int r) {
	return ((im1.multiplyGPGPU(im2)).boxFilterGPGPU(r)).minusGPGPU(mean1.multiplyGPGPU(mean2));
	//return ((im1 * im2).boxFilterGPGPU(r)) - mean1 * mean2;
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
	float *tabCovarGCost = &(const_cast<Image&>(covarVec[0]))(0, 0);
	float *tabCovarBCost = &(const_cast<Image&>(covarVec[0]))(0, 0);
	
	aCoeffSpaceWithCuda(aCoeffSpaceTab, tabVarRR, tabVarRG, tabVarRB, tabVarGG, tabVarGB, tabVarBB,
						tabCovarRCost, tabCovarGCost, tabCovarBCost, width, height, BLOCK, epsilon);

	toVector(aCoeffSpaceTab, aCoeffSpace, width, height, 3);
}

static Image disparity_selection_GPGPU(std::vector<Image> const &costVolume,
									   const int width, const int height, const int dispMin, const int dispMax) {
	
	float *disparityTab = new float[width * height];
	float *costVolumeTab = new float[width * height * costVolume.size()];
	toTab(costVolume, costVolumeTab, width, height);

	disparitySelectionWithCuda(disparityTab, costVolumeTab, width, height, dispMin, dispMax);

	Image disparity(disparityTab, width, height);

	return disparity;
}

Image compute_cost_volume_CPU_GPGPU(Image im1Color, Image im2Color, int dispMin, int dispMax, const ParamGuidedFilter &param) {
	Image im1R = im1Color.r(), im1G = im1Color.g(), im1B = im1Color.b();
	Image im2R = im2Color.r(), im2G = im2Color.g(), im2B = im2Color.b();
	std::vector<Image> image1RGBvec{ im1R, im1G, im1B };
	std::vector<Image> image2RGBvec{ im1R, im1G, im1B };
	const int width = im1R.width(), height = im1R.height();
	const int r = param.kernel_radius;
	const int dispSize = dispMax - dispMin + 1;
	std::cout << "Cost-volume: " << (dispMax - dispMin + 1) << " disparities. ";

	Image im1Gray(width, height);
	Image im2Gray(width, height);
	rgb_to_gray(&im1R(0, 0), &im1G(0, 0), &im1B(0, 0), width, height, &im1Gray(0, 0));
	rgb_to_gray(&im2R(0, 0), &im2G(0, 0), &im2B(0, 0), width, height, &im2Gray(0, 0));
	Image gradient1 = im1Gray.gradX();
	Image gradient2 = im2Gray.gradX();

	// Compute the mean and variance of each patch, eq. (14)
	Image meanIm1R = im1R.boxFilter(r);
	Image meanIm1G = im1G.boxFilter(r);
	Image meanIm1B = im1B.boxFilter(r);

	Image varIm1RR = covariance(im1R, meanIm1R, im1R, meanIm1R, r);
	Image varIm1RG = covariance(im1R, meanIm1R, im1G, meanIm1G, r);
	Image varIm1RB = covariance(im1R, meanIm1R, im1B, meanIm1B, r);
	Image varIm1GG = covariance(im1G, meanIm1G, im1G, meanIm1G, r);
	Image varIm1GB = covariance(im1G, meanIm1G, im1B, meanIm1B, r);
	Image varIm1BB = covariance(im1B, meanIm1B, im1B, meanIm1B, r);

	std::vector<Image> varianceVec{varIm1RR, varIm1RG, varIm1RB, varIm1GG, varIm1GB, varIm1BB};

	std::vector<Image> costVolume;
	compute_costVolume_GPGPU(image1RGBvec, image2RGBvec, gradient1, gradient2, dispMin, dispMax, param, costVolume);

	Image disparity = disparity_selection_GPGPU(costVolume, width, height, dispMin, dispMax);

	return disparity;
}

Image filter_cost_volume_CPU_GPGPU(Image im1Color, Image im2Color, int dispMin, int dispMax, const ParamGuidedFilter &param) {

	// DECOMPOSE IMAGES TO RGB

	Image im1R = im1Color.r(), im1G = im1Color.g(), im1B = im1Color.b();
	Image im2R = im2Color.r(), im2G = im2Color.g(), im2B = im2Color.b();

	std::vector<Image> image1RGBvec{ im1R, im1G, im1B };
	std::vector<Image> image2RGBvec{ im1R, im1G, im1B };

	// INIT CONSTANTS

	const int width = im1R.width(), height = im1R.height();
	const int r = param.kernel_radius;
	const int dispSize = dispMax - dispMin + 1;
	std::cout << "Cost-volume: " << dispSize << " disparities." << std::endl;

	// COMPUTE INTERMEDIATE VARIABLES

	std::cout << "Compute intermediate images..." <<std::endl;

	Image im1Gray(width, height);
	Image im2Gray(width, height);
	rgb_to_gray(&im1R(0, 0), &im1G(0, 0), &im1B(0, 0), width, height, &im1Gray(0, 0));
	rgb_to_gray(&im2R(0, 0), &im2G(0, 0), &im2B(0, 0), width, height, &im2Gray(0, 0));
	std::cout << "	RGB to Gray Done!" << std::endl;

	Image gradient1 = im1Gray.gradXGPGPU();
	Image gradient2 = im2Gray.gradXGPGPU();
	std::cout << "	Gradient X Done!" << std::endl;

	Image meanIm1R = im1R.boxFilterGPGPU(r);
	Image meanIm1G = im1G.boxFilterGPGPU(r);
	Image meanIm1B = im1B.boxFilterGPGPU(r);
	std::cout << "	Box Filter Done!" << std::endl;

	Image varIm1RR = covarianceGPGPU(im1R, meanIm1R, im1R, meanIm1R, r);
	Image varIm1RG = covarianceGPGPU(im1R, meanIm1R, im1G, meanIm1G, r);
	Image varIm1RB = covarianceGPGPU(im1R, meanIm1R, im1B, meanIm1B, r);
	Image varIm1GG = covarianceGPGPU(im1G, meanIm1G, im1G, meanIm1G, r);
	Image varIm1GB = covarianceGPGPU(im1G, meanIm1G, im1B, meanIm1B, r);
	Image varIm1BB = covarianceGPGPU(im1B, meanIm1B, im1B, meanIm1B, r);
	std::cout << "	Covariances Done!" << std::endl;

	std::vector<Image> varianceVec{varIm1RR, varIm1RG, varIm1RB, varIm1GG, varIm1GB, varIm1BB};

	// COMPUTE COST VOLUME

	std::cout << "Compute Cost Volume..." << std::endl;
	std::vector<Image> costVolume;
	compute_costVolume_GPGPU(image1RGBvec, image2RGBvec, gradient1, gradient2, dispMin, dispMax, param, costVolume);
	std::cout << "	Cost Volume Done!" << std::endl;
	std::cout << "	Cost Volume size = " << costVolume[0].width() << "x" << costVolume[0].height() << "x" << costVolume.size() << std::endl;

	// GUIDED FILTER

	std::cout << "Apply Guided Filter..." << std::endl;
	std::vector<Image> filteredCostVolume;
	filteredCostVolume.reserve(dispSize);

	for (std::vector<int>::size_type i = 0; i != costVolume.size(); i++) {
		std::cout << '*' << std::flush;
		Image dCost = costVolume[i];
		Image meanCost = dCost.boxFilterGPGPU(r);

		Image covarIm1RCost = covarianceGPGPU(im1R, meanIm1R, dCost, meanCost, r);
		Image covarIm1GCost = covarianceGPGPU(im1G, meanIm1G, dCost, meanCost, r);
		Image covarIm1BCost = covarianceGPGPU(im1B, meanIm1B, dCost, meanCost, r);

		std::vector<Image> covarVec{covarIm1RCost, covarIm1GCost, covarIm1BCost};

		std::cout << "		Computing aCoeff for disp " << dispMin + i << "..." << std::endl;
		std::vector<Image> aCoeffSpace;
		compute_aCoeffSpace_GPGPU(varianceVec, covarVec, param.epsilon, aCoeffSpace);
		std::cout << "		Done!" << std::endl;

		Image ARxM1R = aCoeffSpace[0].multiplyGPGPU(meanIm1R);
		Image AGxM1G = aCoeffSpace[1].multiplyGPGPU(meanIm1G);
		Image ABxM1B = aCoeffSpace[2].multiplyGPGPU(meanIm1B);

		std::cout << "		Computing bCoeff for disp " << dispMin + i << "..." << std::endl;
		Image b = (meanCost.minusGPGPU(ARxM1R).minusGPGPU(AGxM1G).minusGPGPU(ABxM1B)).boxFilter(r);
		//Image b = (meanCost - aCoeffSpace[0] * meanIm1R - aCoeffSpace[1] * meanIm1G - aCoeffSpace[2] * meanIm1B).boxFilterGPGPU(r);
		std::cout << "		Done!" << std::endl;

		Image MARx1R = (aCoeffSpace[0].boxFilter(r)).multiplyGPGPU(im1R);
		Image MAGx1G = (aCoeffSpace[1].boxFilter(r)).multiplyGPGPU(im1G);
		Image MABx1B = (aCoeffSpace[2].boxFilter(r)).multiplyGPGPU(im1B);

		std::cout << "		Computing filtered cost map for disp " << dispMin + i << "..." << std::endl;
		filteredCostVolume.push_back(MARx1R.plusGPGPU(MAGx1G).plusGPGPU(MABx1B).plusGPGPU(b));
		//filteredCostVolume.push_back(b + aCoeffSpace[0].boxFilterGPGPU(r) * im1R + aCoeffSpace[1].boxFilterGPGPU(r) * im1G + aCoeffSpace[2].boxFilterGPGPU(r) * im1B);
		std::cout << "		Done!" << std::endl;
	}
	std::cout << "	Guided Filter Done!" << std::endl;
	std::cout << "	Cost Volume size = " << filteredCostVolume[0].width() << "x" << filteredCostVolume[0].height() << "x" << filteredCostVolume.size() << std::endl;

	// DISPARITY SELECTION

	std::cout << "Apply Disparity Selection..." << std::endl;
	Image disparity = disparity_selection_GPGPU(filteredCostVolume, width, height, dispMin, dispMax);
	std::cout << "	Disparity Selection Done!" << std::endl;

	std::cout << std::endl;
	return disparity;
}