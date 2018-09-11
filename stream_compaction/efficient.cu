#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernScanDataUpSweep(int n, int offset1, int offset2, int* buff) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;

			int access = index * offset2 - 1;
			if (access >= n || n < 1 || access < 0) return;

			buff[access] += buff[access - offset1];
			

		}
		__global__ void kernScanDataDownSweep(int n, int offset1, int offset2, int* buff) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;

			int access = index * offset2 - 1;
			if (access >= n || n < 1 || access < 0) return;

			int temp = buff[access - offset1];
			buff[access - offset1] = buff[access];
			buff[access] += temp;

		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			int limit = ilog2ceil(n);
			int size = pow(2, limit);

			// allocate memory
			int* dev_buf;
			cudaMalloc((void**)&dev_buf, size * sizeof(int));

			// copy input data to device
			cudaMemset(dev_buf + n, 0, (size - n) * sizeof(int));
			cudaMemcpy(dev_buf, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			timer().startGpuTimer();

			int d;
			int offset1;
			int offset2;
			// UpSweep
			
			for (d = 1; d <= limit; d++) {
				offset1 = pow(2, d - 1);
				offset2 = pow(2, d);
				kernScanDataUpSweep << <fullBlocksPerGrid, blockSize >> >(size, offset1, offset2, dev_buf);
				cudaDeviceSynchronize();
			}

			// DownSweep
			cudaMemset(dev_buf + n - 1, 0, (size - n + 1)* sizeof(int));
			for (d = limit; d >= 1; d--) {
				offset1 = pow(2, d - 1);
				offset2 = pow(2, d);
				kernScanDataDownSweep << <fullBlocksPerGrid, blockSize >> >(size, offset1, offset2, dev_buf);
				cudaDeviceSynchronize();
			}


			timer().endGpuTimer();

			// for debugging
			//printf("Limit: %i, Size: %i, N: %i\n", limit, size, n);

			// copy output data to host
			cudaMemcpy(odata, dev_buf, n * sizeof(int), cudaMemcpyDeviceToHost);

			// cleanup
			cudaFree(dev_buf);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

			int* dev_map; // bool mapping
			int* dev_scan; // scanned data
			int* dev_out; // compacted data to output
			int* dev_in; // input data

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			int limit = ilog2ceil(n);
			int size = pow(2, limit);

			// allocate memory
			cudaMalloc((void**)&dev_in, n * sizeof(int));
			cudaMalloc((void**)&dev_map, n * sizeof(int));
			cudaMalloc((void**)&dev_out, n * sizeof(int));
			cudaMalloc((void**)&dev_scan, size * sizeof(int));

			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			

            timer().startGpuTimer();
            // map
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, dev_map, dev_in);

			cudaMemcpy(dev_scan, dev_map, n * sizeof(int), cudaMemcpyDeviceToDevice); // copy bool data to scan
			cudaMemset(dev_scan + n, 0, (size - n) * sizeof(int)); // zero extra mem

			// scan

			int d;
			int offset1;
			int offset2;
			// UpSweep

			for (d = 1; d <= limit; d++) {
				offset1 = pow(2, d - 1);
				offset2 = pow(2, d);
				kernScanDataUpSweep << <fullBlocksPerGrid, blockSize >> >(size, offset1, offset2, dev_scan);
				cudaDeviceSynchronize();
			}

			// DownSweep
			cudaMemset(dev_scan + n - 1, 0, (size - n + 1) * sizeof(int)); // zero extra
			for (d = limit; d >= 1; d--) {
				offset1 = pow(2, d - 1);
				offset2 = pow(2, d);
				kernScanDataDownSweep << <fullBlocksPerGrid, blockSize >> >(size, offset1, offset2, dev_scan);
				cudaDeviceSynchronize();
			}

			// scatter
			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, dev_out, dev_in, dev_map, dev_scan);

            timer().endGpuTimer();

			// copy output to host
			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			int map_val;
			int r_val;
			cudaMemcpy(&r_val, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&map_val, dev_map + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

			if (map_val != 0) r_val++;

			// cleanup
			cudaFree(dev_in);
			cudaFree(dev_map);
			cudaFree(dev_out);
			cudaFree(dev_scan);

            return r_val;
        }
    }
}
