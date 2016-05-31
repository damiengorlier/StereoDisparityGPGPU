#include "occlusion.h"
#include "cudaKernels.cuh"
#include "image.h"

#define BLOCK 16

Image detect_occlusion_GPGPU(const Image& disparityLeft, const Image& disparityRight,
							 float dOcclusion, int tolDisp) {
	const int width = disparityLeft.width(), height = disparityRight.height();
	float *dispOccTab = new float[width * height];
	float *dispLeftTab = &(const_cast<Image&>(disparityLeft))(0, 0); float *dispRightTab = &(const_cast<Image&>(disparityRight))(0, 0);

	detectOcclusionWithCuda(dispOccTab, dispLeftTab, dispRightTab, width, height, BLOCK, dOcclusion, tolDisp);

	Image dispOcc(dispOccTab, width, height);
	return dispOcc;
}