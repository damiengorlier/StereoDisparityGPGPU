#ifndef DEVICEPROPERTIES_CUH
#define DEVICEPROPERTIES_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

void printDeviceProperties() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	printf("#---------------------------#\n");
	printf("#     DEVICE PROPERTIES     #\n");
	printf("#---------------------------#\n");
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		printf("  Total global memory (MB): %d\n", prop.totalGlobalMem >> 20);
		printf("  Shared memory per block (KB): %d\n", prop.sharedMemPerBlock >> 10);
		printf("  MultiProcessor Count: %d\n", prop.multiProcessorCount);
		printf("  Max threads per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
		printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("  Max block dimensions: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("  Max grid dimensions: %d x %d x %d\n\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
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

	return devProp;
}

#endif