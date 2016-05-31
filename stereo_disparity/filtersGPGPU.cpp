#include "image.h"
#include "cudaKernels.cuh"

#include <cassert>
#include <math.h>

// TODO : mettre ça en paramètre du programme
#define BLOCK 16

Image Image::rgbToGrayGPGPU() const {
	Image D(w, h);
	float *out = D.tab;

	Image r = Image(tab + 0 * w*h, w, h);
	Image g = Image(tab + 1 * w*h, w, h);
	Image b = Image(tab + 2 * w*h, w, h);

	rgbToGrayWithCuda(out, r.tab, g.tab, b.tab, w, h, BLOCK);

	return D;
}

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

	integralWithCuda(out, tab, w, h, BLOCK);

	return D;
}

// Should be private. Only for test purpose
Image Image::scanGPGPU(bool addOri) const {
	Image D(w, h);
	float *out = D.tab;

	scanWithCuda(out, tab, w, h, BLOCK, addOri);

	return D;
}

Image Image::transposeGPGPU() const {
	Image D(h, w);
	float *out = D.tab;

	transposeWithCuda(out, tab, w, h, BLOCK);

	return D;
}

Image Image::boxFilterGPGPU(int radius) const {
	Image D(w, h);
	float *out = D.tab;

	boxFilterWithCuda(out, tab, w, h, BLOCK, radius);

	return D;
}