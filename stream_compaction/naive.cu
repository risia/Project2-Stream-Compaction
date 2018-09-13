#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

		__global__ void kernScanDataNaive(int n, int offset, int* out, const int *in) {

			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index > n || n < 1) return;
			
			if (index >= offset) {
				out[index] = in[index] + in[index - offset];
			}
			else {
				out[index] = in[index];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			// allocate memory
			int* dev_out;
			int* dev_in;
			int* swap;

			cudaMalloc((void**)&dev_out, n * sizeof(int));
			cudaMalloc((void**)&dev_in, n * sizeof(int));
			checkCUDAError("naive scan malloc fail!");

			// copy input data to device
			cudaMemset(dev_in, 0, sizeof(int));
			cudaMemcpy(dev_in + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("naive input copy fail!");

            timer().startGpuTimer();

			int d;
			int offset;
			for (d = 1; d <= ilog2ceil(n); d++) {
				offset = pow(2, d - 1);
				kernScanDataNaive<<<fullBlocksPerGrid, blockSize>>>(n, offset, dev_out, dev_in);
				checkCUDAError("naive scan iteration fail!");
				// swap buffers
				swap = dev_in;
				dev_in = dev_out;
				dev_out = swap;

			}

            timer().endGpuTimer();

			// copy output data to host
			cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("naive copy output fail!");

			// cleanup
			cudaFree(dev_in);
			cudaFree(dev_out);
        }
    }
}
