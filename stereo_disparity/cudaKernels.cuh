#ifndef CUDAKERNELS_CUH
#define CUDAKERNELS_CUH

void gradXWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim);
void scanWithCuda(float *host_out, const float *host_in, const int width, const int height, const bool addOri);
void transposeWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim);
void boxFilterWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim, int radius);

#endif