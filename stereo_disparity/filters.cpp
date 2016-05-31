/**
 * @file filters.cpp
 * @brief Various image filters
 * @author Pauline Tan <pauline.tan@ens-cachan.fr>
 *         Pascal Monasse <monasse@imagine.enpc.fr>
 * 
 * Copyright (c) 2012-2013, Pauline Tan, Pascal Monasse
 * All rights reserved.
 * 
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * You should have received a copy of the GNU General Pulic License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "image.h"
#include "TimingCPU.h"
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>

#define TIMING 0

/// Fill pixels below value \a vMin using values at two closest pixels on same
/// line above \a vMin. The filling value is the result of \a cmp with the two
/// values as parameters.
void Image::fillX(float vMin, const float& (*cmp)(const float&,const float&)) {
    for(int y=0; y<h; y++) {
        int x0=-1;
        float v0 = vMin;
        while(x0<w) {
            int x1=x0+1;
            while(x1<w && (*this)(x1,y)<vMin) ++x1;
            float v=v0;
            if(x1<w)
                v = cmp(v,v0=(*this)(x1,y));
            std::fill(&(*this)(x0+1,y), &(*this)(x1,y), v);
            x0=x1;
        }
    }
}

/// Fill pixels below value \a vMin with min of values at closest pixels on same
/// line above \a vMin.
void Image::fillMinX(float vMin) {
    fillX(vMin, std::min<float>);
}

/// Fill pixels below value \a vMin with max of values at closest pixels on same
/// line above \a vMin.
void Image::fillMaxX(float vMin) {
    fillX(vMin, std::max<float>);
}

/// Derivative along x-axis
Image Image::gradX() const {
	TimingCPU timer;
	timer.StartCounter();

    assert(w>=2);
    Image D(w,h);
    float* out=D.tab;
    for(int y=0; y<h; y++) {
        const float* in=tab+y*w;
        *out++ = in[1]-in[0];           // Right - current
        for(int x=1; x+1<w; x++, in++)
            *out++ = .5f*(in[2]-in[0]); // Right - left
        *out++ = in[1]-in[0];           // Current - left
    }

	double time = timer.GetCounter();
	if (TIMING) {
		std::cout << "CPU | gradX : " << time << " ms" << std::endl;
	}

    return D;
}

/// For GPGPU test
Image Image::integral() const {
	TimingCPU timer;
	timer.StartCounter();

	float* S = new float[w*h]; // Use double to mitigate precision loss
	for (int i = w*h - 1; i >= 0; i--)
		S[i] = static_cast<double>(tab[i]);

	//cumulative sum table S, eq. (24)
	for (int y = 0; y<h; y++) { //horizontal
		float *in = S + y*w;
		float *out = in + 1;
		for (int x = 1; x<w; x++)
			*out++ += *in++;
	}
	for (int y = 1; y<h; y++) { //vertical
		float *in = S + (y - 1)*w;
		float *out = in + w;
		for (int x = 0; x<w; x++)
			*out++ += *in++;
	}

	Image B(w, h);
	B.tab = S;

	double time = timer.GetCounter();
	if (TIMING) {
		std::cout << "CPU | integral : " << time << " ms" << std::endl;
	}

	return B;
}

/// Averaging filter with box of \a radius.
///
/// Use the integral image for fast computation. The integral image is of type
/// double to mitigate risks of precision loss for large images.
Image Image::boxFilter(int radius) const {
	TimingCPU timer;
	timer.StartCounter();

    double* S = new double[w*h]; // Use double to mitigate precision loss
    for(int i=w*h-1; i>=0; i--)
        S[i] = static_cast<double>(tab[i]);

    //cumulative sum table S, eq. (24)
    for(int y=0; y<h; y++) { //horizontal
        double *in=S+y*w, *out=in+1;
        for(int x=1; x<w; x++)
            *out++ += *in++;
    }
    for(int y=1; y<h; y++) { //vertical
        double *in=S+(y-1)*w, *out=in+w;
        for(int x=0; x<w; x++)
            *out++ += *in++;
    }

    //box filtered image B
    Image B(w,h);
    float *out=B.tab;
    for(int y=0; y<h; y++) {
        int ymin = std::max(-1, y-radius-1);
        int ymax = std::min(h-1, y+radius);
        for(int x=0; x<w; x++, out++) {
            int xmin = std::max(-1, x-radius-1);
            int xmax = std::min(w-1, x+radius);
            // S(xmax,ymax)-S(xmin,ymax)-S(xmax,ymin)+S(xmin,ymin), eq. (25)
            double val = S[ymax*w+xmax];
            if(xmin>=0)
                val -= S[ymax*w+xmin];
            if(ymin>=0)
                val -= S[ymin*w+xmax];
            if(xmin>=0 && ymin>=0)
                val += S[ymin*w+xmin];
            *out = static_cast<float>(val/((xmax-xmin)*(ymax-ymin))); //average
        }
    }

	double time = timer.GetCounter();
	if (TIMING) {
		std::cout << "CPU | boxFilter : " << time << " ms" << std::endl;
	}

    delete [] S;
    return B;
}

/// Median filter, write results in \a M
void Image::median(int radius, Image& M) const {
    int size=2*radius+1;
    size *= size;
    float* v = new float[size];
    for(int y=0; y<h; y++)
        for(int x=0; x<w; x++) {
            int n=0;
            for(int j=-radius; j<=radius; j++)
                if(0<=j+y && j+y<h)
                    for(int i=-radius; i<=radius; i++)
                        if(0<=i+x && i+x<w)
                            v[n++] = (*this)(i+x,j+y);
            std::nth_element(v, v+n/2, v+n);
            M(x,y) = v[n/2];
        }
    delete [] v;
}

/// Median filter for a color image
Image Image::medianColor(int radius) const {
    Image M(w,3*h);
    M.h=h;
    Image I=M.r(); r().median(radius, I);
    I=M.g(); g().median(radius, I);
    I=M.b(); b().median(radius, I);
    return M;
}

/// Square function
inline float sqr(float v1) {
    return v1*v1;
}

/// Square L2 distance between colors at (x1,y1) and at (x2,y2)
float Image::dist2Color(int x1,int y1, int x2,int y2) const {
    return sqr((*this)(x1,y1    )-(*this)(x2,y2    ))+
           sqr((*this)(x1,y1+  h)-(*this)(x2,y2+  h))+
           sqr((*this)(x1,y1+2*h)-(*this)(x2,y2+2*h));
}

/// @brief Compute weighted histogram of image values.
///
/// The area is [x-radius,x+radius]x[y-radius,y+radius] (inter image).
/// Values are shifted by \a vMin.
/// Weights are computed from the \a guidance image with factors \a sSpace for
/// spatial distance and \a sColor for color distance to central pixel.
void Image::weighted_histo(std::vector<float>& tab, int x, int y, int radius,
                           int vMin, const Image& guidance,
                           float sSpace, float sColor) const {
    std::fill(tab.begin(), tab.end(), 0.0f);
    for(int dy=-radius; dy<=radius; dy++)
        if(0<=y+dy && y+dy<h)
            for(int dx=-radius; dx<=radius; dx++)
                if(0<=x+dx && x+dx<w) {
                    float w =
                        exp(-(dx*dx+dy*dy)*sSpace
                            -guidance.dist2Color(x,y,x+dx,y+dy)*sColor);
                    tab[(int)((*this)(x+dx,y+dy))-vMin] += w;
                }
}

/// Index in histogram \a tab reaching median.
static int median_histo(const std::vector<float>& tab) {
    float sum=std::accumulate(tab.begin(), tab.end(), 0.0f)/2;
    int d=-1;
    for(float cumul=0; cumul<sum;)
        cumul += tab[++d];
    return d;
}

/// @brief Weighted median filter of current image.
///
/// Image is assumed to have integer values in [vMin,vMax]. Weight are computed
/// as in bilateral filter in color image \a guidance. Only pixels of image
/// \a where outside [vMin,vMax] are filtered.
Image Image::weightedMedianColor(const Image& guidance,
                                 const Image& where, int vMin, int vMax,
                                 int radius, float sSpace, float sColor) const
{
    sSpace = 1.0f/(sSpace*sSpace);
    sColor = 1.0f/(sColor*sColor);

    const int size=vMax-vMin+1;
    std::vector<float> tab(size);
    Image M(w,h);

#ifdef _OPENMP
#pragma omp parallel for firstprivate(tab)
#endif
    for(int y=0; y<h; y++)
        for(int x=0; x<w; x++) {
            if(where(x,y)>=vMin) {
                M(x,y)=(*this)(x,y);
                continue;
            }
            weighted_histo(tab, x,y, radius, vMin, guidance, sSpace, sColor);
            M(x,y) = static_cast<float>(vMin+median_histo(tab));
        }
    return M;
}
