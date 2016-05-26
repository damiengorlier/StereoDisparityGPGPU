/**************/
/* TIMING GPU */
/**************/

#include "TimingGPU.cuh"
#include "cudaErrorCheck.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

struct PrivateTimingGPU {
	cudaEvent_t     start;
	cudaEvent_t     stop;
};

// default constructor
TimingGPU::TimingGPU() { privateTimingGPU = new PrivateTimingGPU; }

// default destructor
TimingGPU::~TimingGPU() { }

void TimingGPU::StartCounter()
{
	CudaSafeCall(cudaEventCreate(&((*privateTimingGPU).start)));
	CudaSafeCall(cudaEventCreate(&((*privateTimingGPU).stop)));
	CudaSafeCall(cudaEventRecord((*privateTimingGPU).start, 0));
}

void TimingGPU::StartCounterFlags()
{
	int eventflags = cudaEventBlockingSync;

	CudaSafeCall(cudaEventCreateWithFlags(&((*privateTimingGPU).start), eventflags));
	CudaSafeCall(cudaEventCreateWithFlags(&((*privateTimingGPU).stop), eventflags));
	CudaSafeCall(cudaEventRecord((*privateTimingGPU).start, 0));
}

// Gets the counter in ms
float TimingGPU::GetCounter()
{
	float   time;
	CudaSafeCall(cudaEventRecord((*privateTimingGPU).stop, 0));
	CudaSafeCall(cudaEventSynchronize((*privateTimingGPU).stop));
	CudaSafeCall(cudaEventElapsedTime(&time, (*privateTimingGPU).start, (*privateTimingGPU).stop));
	return time;
}