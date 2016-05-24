#ifndef CUDAKERNELS_CUH
#define CUDAKERNELS_CUH

// OPERATORS
void operatorWithCuda(float *host_out, const float *host_in1, const float *host_in2, const int width, const int height, const int blockDim, const int op);

// FILTERS
void gradXWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim);
void scanWithCuda(float *host_out, const float *host_in, const int width, const int height, const bool addOri);
void transposeWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim);
void boxFilterWithCuda(float *host_out, const float *host_in, const int width, const int height, const int blockDim, int radius);

#endif