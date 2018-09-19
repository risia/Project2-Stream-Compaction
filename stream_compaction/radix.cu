#include "radix.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define blockSize 128

// macros for bit checks and toggles
// define macro to get nth bit of int
#define bitK(num, k)  ((num >> k) & 1)
// flip bit
#define flipBit(bit) (bit ^ 1)

namespace StreamCompaction {
	namespace Radix {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		__global__ void kernFindMax(int n, int offset1, int offset2, int* buff) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;

			int access = index * offset2 - 1;
			if (access >= n || n < 1 || access < 0) return;

			// modify in place
			if (buff[access] < buff[access - offset1]) {
				buff[access] = buff[access - offset1];
			}

		}

		__global__ void kernBoolMaps(int n, int k, int* input, int* b_arr, int* f_arr) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) return;

			int bit = bitK(input[index], k);
			int fBit = flipBit(bit);

			b_arr[index] = bit; // maps bit k in b_arr
			f_arr[index] = fBit; // copy flipped value here for scan
		}

		__global__ void kernComputeT(int n, int totFalse, int *t_arr, int *f_arr) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) return;

			t_arr[index] = index - f_arr[index] + totFalse;
		}

		__global__ void kernRadixScatter(int n, int *out, int *in, int *b_arr, int *f_arr, int *t_arr) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) return;

			int access = b_arr[index] ? t_arr[index] : f_arr[index];
			out[access] = in[index];
		}

		/*
			Performs Radix Sort on input data using Work-Efficient Scan
		*/

		int max(int n, int *idata) {
			int limit = ilog2ceil(n);
			int size = pow(2, limit);

			dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);

			int d;
			int offset1;
			int offset2;

			int max;
			int *max_arr;
			cudaMalloc((void**)&max_arr, size * sizeof(int));
			cudaMemcpy(max_arr, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemset(max_arr + n, 0, (size - n) * sizeof(int));

			timer().startGpuTimer();

			// find max of data
			for (d = 1; d <= limit; d++) {
				offset1 = pow(2, d - 1);
				offset2 = pow(2, d);
				fullBlocksPerGrid.x = ((size / offset2) + blockSize) / blockSize;
				kernFindMax << <fullBlocksPerGrid, blockSize >> >(size, offset1, offset2, max_arr);
				checkCUDAError("Radix find max fail!");
			}

			timer().endGpuTimer();

			cudaMemcpy(&max, max_arr + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
			
			cudaFree(max_arr);
			return max;


		}

		void sort(int n, int *odata, const int *idata) {

			int limit = ilog2ceil(n);
			int size = pow(2, limit);

			dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);

			int d;
			int offset1;
			int offset2;

			int max;
			int totFalse;
			int temp;
			
			// alloc. memory
			int *b_arr;
			//int *e_arr; // e_arr sorted in f_arr, do not need
			int *f_arr;
			int *t_arr;

			int *dev_in;
			int *dev_out;

			int *swap;

			cudaMalloc((void**)&b_arr, n * sizeof(int));
			//cudaMalloc((void**)&e_arr, n * sizeof(int));
			cudaMalloc((void**)&f_arr, size * sizeof(int)); // sized to power of 2 for scan
			cudaMalloc((void**)&t_arr, n * sizeof(int));

			cudaMalloc((void**)&dev_in, n * sizeof(int));
			cudaMalloc((void**)&dev_out, n * sizeof(int));

			// copy input
			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(f_arr, dev_in, n * sizeof(int), cudaMemcpyDeviceToDevice);
			cudaMemset(f_arr + n, 0, (size - n) * sizeof(int));
			checkCUDAError("Radix mem init fail!");

			timer().startGpuTimer();

			// find max of data
			for (d = 1; d <= limit; d++) {
				offset1 = pow(2, d - 1);
				offset2 = pow(2, d);
				fullBlocksPerGrid.x = ((size / offset2) + blockSize) / blockSize;
				kernFindMax << <fullBlocksPerGrid, blockSize >> >(size, offset1, offset2, f_arr);
				checkCUDAError("Radix find max fail!");
			}

			// copy max value
			cudaMemcpy(&max, f_arr + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
			// zero extra mem
			cudaMemset(f_arr + n, 0, (size - n) * sizeof(int));

			for (int k = 0; k < ilog2ceil(max); k++) {

				// map arrays b & e
				fullBlocksPerGrid.x = ((n + blockSize - 1) / blockSize);
				kernBoolMaps << <fullBlocksPerGrid, blockSize >> > (n, k, dev_in, b_arr, f_arr);
				checkCUDAError("Radix bool maps fail!");

				// get whether last number's bit is false
				cudaMemcpy(&totFalse, f_arr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
				
				// exclusive scan f_arr

				// UpSweep
				for (d = 1; d <= limit; d++) {
					offset1 = pow(2, d - 1);
					offset2 = pow(2, d);
					fullBlocksPerGrid.x = ((size / offset2) + blockSize) / blockSize;
					StreamCompaction::Efficient::kernScanDataUpSweep << <fullBlocksPerGrid, blockSize >> >(size, offset1, offset2, f_arr);
					checkCUDAError("Radix upsweep fail!");
				}

				// DownSweep
				cudaMemset(f_arr + n - 1, 0, (size - n + 1) * sizeof(int));
				for (d = limit; d >= 1; d--) {
					offset1 = pow(2, d - 1);
					offset2 = pow(2, d);
					fullBlocksPerGrid.x = ((size / offset2) + blockSize) / blockSize;
					StreamCompaction::Efficient::kernScanDataDownSweep << <fullBlocksPerGrid, blockSize >> >(size, offset1, offset2, f_arr);
					checkCUDAError("Radix downsweep fail!");
				}

				// total Falses
				cudaMemcpy(&temp, f_arr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
				totFalse += temp;

				// Compute t_arr
				fullBlocksPerGrid.x = ((n + blockSize - 1) / blockSize);
				kernComputeT << <fullBlocksPerGrid, blockSize >> >(n, totFalse, t_arr, f_arr);
				checkCUDAError("Radix t_arr compute fail!");

				// scatter
				kernRadixScatter << <fullBlocksPerGrid, blockSize >> >(n, dev_out, dev_in, b_arr, f_arr, t_arr);
				checkCUDAError("Radix scatter fail!");

				swap = dev_out;
				dev_out = dev_in;
				dev_in = swap;
			}

			timer().endGpuTimer();

			// copy output data to host
			cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("Radix output copy fail!");

			// cleanup
			cudaFree(dev_in);
			cudaFree(dev_out);
			cudaFree(b_arr);
			cudaFree(f_arr);
			cudaFree(t_arr);
		}

	}

}