#ifndef CUDAKERNELS_CUH
#define CUDAKERNELS_CUH

// OPERATORS

void operatorWithCuda(float *host_out, const float *host_in1, const float *host_in2, const int width, const int height, const int blockDim, const int op);

// FILTERS

void gradXWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim);
void scanWithCuda(float *host_out, const float *host_in, const int width, const int height, const bool addOri);
void transposeWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim);
void boxFilterWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim, int radius);

// COMPUTE

void costVolumeWithCuda(float *host_out,
						const float *host_im1R, const float *host_im1G, const float *host_im1B,
						const float *host_im2R, const float *host_im2G, const float *host_im2B,
						const float *host_grad1, float *host_grad2,
						const int width, const int height, const int blockDim,
						const int dispMin, const int dispMax, const float colorTh, const float gradTh, const float alpha);

void aCoeffSpaceWithCuda(float *host_out,
						 const float *host_varRR, const float *host_varRG, const float *host_varRB,
						 const float *host_varGG, const float *host_varGB, const float *host_varBB,
						 const float *host_covarRCost, const float *host_covarGCost, const float *host_covarBCost,
						 const int width, const int height, const int blockDim,
						 const float epsilon);

void disparitySelectionWithCuda(float *host_out, const float *host_cost_volume,
								const int width, const int height, const int dispMin, const int dispMax);

#endif