#include "image.h"
#include "cudaKernels.cuh"

#include <cassert>

#define BLOCK 16

Image Image::plusGPGPU(const Image& I) const {
	assert(w == I.w && h == I.h);
	Image S(w, h);
	float* out = S.tab;
	
	operatorWithCuda(out, tab, I.tab, w, h, BLOCK, 0);

	return S;
}

Image Image::minusGPGPU(const Image& I) const {
	assert(w == I.w && h == I.h);
	Image S(w, h);
	float* out = S.tab;

	operatorWithCuda(out, tab, I.tab, w, h, BLOCK, 1);

	return S;
}

Image Image::multiplyGPGPU(const Image& I) const {
	assert(w == I.w && h == I.h);
	Image S(w, h);
	float* out = S.tab;

	operatorWithCuda(out, tab, I.tab, w, h, BLOCK, 2);

	return S;
}