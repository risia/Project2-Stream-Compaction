#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 64

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

			int limit = ilog2ceil(n);
			int size = pow(2, limit);


			dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);

			// allocate memory
			int* dev_buf;
			cudaMalloc((void**)&dev_buf, size * sizeof(int));
			checkCUDAError("w-e scan malloc fail!");

			// copy input data to device
			cudaMemset(dev_buf + n, 0, (size - n) * sizeof(int));
			cudaMemcpy(dev_buf, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("initializing w-e scan data buff fail!");

			timer().startGpuTimer();

			int d;
			int offset1;
			int offset2;

			int threads;

			// UpSweep
			for (d = 1; d <= limit; d++) {
				offset1 = pow(2, d - 1);
				offset2 = pow(2, d);

				threads = (size / offset2);
				fullBlocksPerGrid.x = (threads / blockSize) + 1;

				kernScanDataUpSweep << <fullBlocksPerGrid, blockSize >> >(size, offset1, offset2, dev_buf);
				checkCUDAError("upsweep fail!");
			}

			// DownSweep
			cudaMemset(dev_buf + n - 1, 0, (size - n + 1)* sizeof(int));
			for (d = limit; d >= 1; d--) {
				offset1 = pow(2, d - 1);
				offset2 = pow(2, d);

				threads = (size / offset2);
				fullBlocksPerGrid.x = (threads / blockSize) + 1;

				kernScanDataDownSweep << <fullBlocksPerGrid, blockSize >> >(size, offset1, offset2, dev_buf);
				checkCUDAError("downsweep fail!");
			}


			timer().endGpuTimer();

			// copy output data to host
			cudaMemcpy(odata, dev_buf, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("copying output data fail!");

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

			int limit = ilog2ceil(n);
			int size = pow(2, limit);

			// allocate memory
			cudaMalloc((void**)&dev_in, n * sizeof(int));
			cudaMalloc((void**)&dev_map, n * sizeof(int));
			cudaMalloc((void**)&dev_out, n * sizeof(int));
			cudaMalloc((void**)&dev_scan, size * sizeof(int));
			checkCUDAError("w-e compact malloc fail!");

			cudaMemset(dev_scan + n, 0, (size - n) * sizeof(int)); // zero extra mem
			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice); // copy input data
			checkCUDAError("initializing w-e compact data buffs fail!");

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // map
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, dev_map, dev_in);
			cudaMemcpy(dev_scan, dev_map, n * sizeof(int), cudaMemcpyDeviceToDevice); // copy bool data to scan
			checkCUDAError("w-e compact bool mapping fail!");

			// scan

			int d;
			int offset1;
			int offset2;

			// UpSweep
			for (d = 1; d <= limit; d++) {
				offset1 = pow(2, d - 1);
				offset2 = pow(2, d);
				fullBlocksPerGrid.x = ((size / offset2) + blockSize) / blockSize;
				kernScanDataUpSweep << <fullBlocksPerGrid, blockSize >> >(size, offset1, offset2, dev_scan);
				checkCUDAError("w-e compact upsweep fail!");
			}

			// DownSweep
			cudaMemset(dev_scan + n - 1, 0, (size - n + 1) * sizeof(int));
			for (d = limit; d >= 1; d--) {
				offset1 = pow(2, d - 1);
				offset2 = pow(2, d);
				fullBlocksPerGrid.x = ((size / offset2) + blockSize) / blockSize;
				kernScanDataDownSweep << <fullBlocksPerGrid, blockSize >> >(size, offset1, offset2, dev_scan);
				checkCUDAError("w-e compact downsweep fail!");
			}

			// scatter
			fullBlocksPerGrid.x = ((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, dev_out, dev_in, dev_map, dev_scan);
			checkCUDAError("w-e compact scatter fail!");

            timer().endGpuTimer();

			// copy output to host
			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("w-e compact output copy fail!");

			// calc # of elements for return
			int map_val;
			int r_val;
			cudaMemcpy(&r_val, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&map_val, dev_map + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("w-e compact calc # elem fail!");

			r_val += map_val;

			// cleanup
			cudaFree(dev_in);
			cudaFree(dev_map);
			cudaFree(dev_out);
			cudaFree(dev_scan);

            return r_val;
        }
    }
}
