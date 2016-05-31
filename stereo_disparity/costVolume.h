/**
 * @file costVolume.h
 * @brief Disparity cost volume filtering by guided filter
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

#ifndef COSTVOLUME_H
#define COSTVOLUME_H

#include <vector>

class Image;

/// Parameters specific to the guided filter
struct ParamGuidedFilter {
    float color_threshold;
    float gradient_threshold;
    float alpha;
    int kernel_radius;
    float epsilon;

    /// Constructor with default parameters
    ParamGuidedFilter()
    : color_threshold(7),
      gradient_threshold(2),
      alpha(1-0.1f),
      kernel_radius(9),
      epsilon(0.0001f*255*255) {}
};

// #------------------------#
// #     MAIN FUNCTIONS     #
// #------------------------#

Image filter_cost_volume(Image im1Color, Image im2Color,
	int dispMin, int dispMax,
	const ParamGuidedFilter& param);

Image filter_cost_volume_GPGPU(Image im1Color, Image im2Color,
	int dispMin, int dispMap,
	const ParamGuidedFilter& param);

// #------------------------#
// #     TEST FUNCTIONS     #
// #------------------------#

// CPU

Image covariance(Image im1, Image mean1, Image im2, Image mean2, int r);

std::vector<Image> cost_volume(Image im1Color, Image im2Color,
	int dispMin, int dispMax,
	const ParamGuidedFilter& param);

Image disp_cost_volume(Image im1Color, Image im2Color,
	int dispMin, int dispMax,
	const ParamGuidedFilter& param);

// GPU

Image covarianceGPGPU(Image im1, Image mean1, Image im2, Image mean2, int r);

std::vector<Image> cost_volume_CPU_GPGPU(Image im1Color, Image im2Color,
	int dispMin, int dispMax,
	const ParamGuidedFilter& param);

Image disp_cost_volume_CPU_GPGPU(Image im1Color, Image im2Color,
	int dispMin, int dispMax,
	const ParamGuidedFilter& param);

Image filter_cost_volume_CPU_GPGPU(Image im1Color, Image im2Color,
	int dispMin, int dispMax,
	const ParamGuidedFilter& param);



#endif
