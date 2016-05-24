#include "image.h"
#include "cudaKernels.cuh"

#include <cassert>
#include <math.h>

#define BLOCK 16

Image Image::gradXGPGPU() const {
	assert(w >= 2);
	Image D(w, h);
	float *out = D.tab;

	gradXWithCuda(out, tab, w, h, BLOCK);

	return D;
}

Image Image::integralGPGPU(bool addOri) const {
	Image D(w, h);
	float *out = D.tab;
	float *tmp1 = new float[w*h];
	float *tmp2 = new float[w*h];

	scanWithCuda(tmp1, tab, w, h, addOri);
	transposeWithCuda(tmp2, tmp1, w, h, BLOCK);
	scanWithCuda(tmp1, tmp2, h, w, addOri);
	transposeWithCuda(out, tmp1, w, h, BLOCK);

	return D;
}

// Should be private. Only for test purpose
Image Image::scanGPGPU(bool addOri) const {
	Image D(w, h);
	float *out = D.tab;

	scanWithCuda(out, tab, w, h, addOri);

	return D;
}

Image Image::transposeGPGPU() const {
	Image D(h, w);
	float *out = D.tab;

	transposeWithCuda(out, tab, w, h, BLOCK);

	return D;
}

Image Image::boxFilterGPGPU(int radius) const {
	Image I = integralGPGPU(true);
	Image D(w, h);
	float *out = D.tab;

	boxFilterWithCuda(out, I.tab, w, h, BLOCK, radius);

	return D;
}