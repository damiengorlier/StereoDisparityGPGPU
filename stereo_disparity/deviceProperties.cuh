#ifndef DEVICEPROPERTIES_CUH
#define DEVICEPROPERTIES_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

void printDeviceProperties() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		printf("  MultiProcessor Count %d\n", prop.multiProcessorCount);
		printf("  Max threads per MultiProcessor %d\n", prop.maxThreadsPerMultiProcessor);
	}
}

struct deviceProp {
	int multiProcCount;
	int maxThreadsPerMP;
};

deviceProp getFirstDeviceProperties() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	if (nDevices > 1) {
		printf("Warning: More than 1 device");
	}
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	deviceProp devProp;
	devProp.multiProcCount = prop.multiProcessorCount;
	devProp.maxThreadsPerMP = prop.maxThreadsPerMultiProcessor;
}

#endif