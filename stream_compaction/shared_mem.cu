#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "shared_mem.h"

#define blockSize 256

namespace StreamCompaction {
    namespace SharedMem {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		__global__ void kernScanDataShared(int n, int* in, int* out) {
			// init shared mem for block, could improve latency
			__shared__ int sBuf[blockSize + 1];

			int tx = threadIdx.x;
			int index = (blockDim.x * blockIdx.x) + tx;

			// copy used vals to shared mem
			sBuf[tx] = (index >= 0 && index < n) ? in[index] : 0;

			__syncthreads(); // avoid mem issues

			int offset = 1; // step size
			int access; // shared buffer access index
			int i; // iterator

			// Upsweep
			for (i = blockSize >> 1; i > 0; i >>= 1) {
				access = (2 * offset * (tx + 1)) - 1;
				if (access < blockSize) sBuf[access] += sBuf[access - offset];
				offset *= 2;
				__syncthreads(); // avoid mem issues
			}
			
			// copy sBuf[blocksize - 1] to sBuf[blocksize] so keep value safe
			if (tx == 0) { 
				sBuf[blockSize] = sBuf[blockSize - 1];
				sBuf[blockSize - 1] = 0;
			}
			__syncthreads(); // avoid mem issues

			// Downsweep (inclusive)
			// do exclusive downsweep
			int temp;

			for (i = blockSize >> 1; i > 0; i >>= 1) {

				offset >>= 1; // div by 2
				access = (2 * offset * (tx + 1)) - 1;
				if (access < blockSize) {
					temp = sBuf[access - offset]; // store left child
					sBuf[access - offset] = sBuf[access];
					sBuf[access] += temp;
				}
				__syncthreads(); // avoid mem issues

			}
			// Write to dev mem
			if (index < n - 1 ) out[index + 1] += sBuf[tx + 1];
			__syncthreads();
			int add_val = 0;
			for (i = index - tx; i > 0; i -= blockSize) {
				if (index != i) add_val += out[i];
			}
			__syncthreads();

			if (index < n) out[index] += add_val;
		}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int limit = ilog2ceil(n);
			int size = pow(2, limit);

			dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);

			int* dev_out; // data to output
			int* dev_in; // input data

			cudaMalloc((void**)&dev_in, n * sizeof(int));
			cudaMalloc((void**)&dev_out, n * sizeof(int));

			// copy input data to device
			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			cudaMemset(dev_out, 0, n * sizeof(int));
			checkCUDAError("initializing shared mem scan data buff fail!");

			timer().startGpuTimer();

			kernScanDataShared<<<fullBlocksPerGrid, blockSize>>>(n, dev_in, dev_out);
			checkCUDAError("shared mem scan fail!");

			timer().endGpuTimer();

			// copy out data
			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("shared mem scan output copy fail!");

			cudaFree(dev_out);
			cudaFree(dev_in);
			
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
			cudaMalloc((void**)&dev_scan, n * sizeof(int));
			checkCUDAError("w-e compact malloc fail!");

			cudaMemset(dev_scan, 0, n * sizeof(int));
			checkCUDAError("initializing shared mem scan data buff fail!");

			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice); // copy input data
			checkCUDAError("initializing w-e compact data buffs fail!");

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // map
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, dev_map, dev_in);
			checkCUDAError("w-e compact bool mapping fail!");

			// scan the map
			fullBlocksPerGrid.x = ((size + blockSize - 1) / blockSize);
			kernScanDataShared << <fullBlocksPerGrid, blockSize >> >(n, dev_map, dev_scan);
			checkCUDAError("shared mem scan fail!");

			// scatter
			fullBlocksPerGrid.x = ((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, dev_out, dev_in, dev_map, dev_scan);
			checkCUDAError("shared mem compact scatter fail!");

	        timer().endGpuTimer();

			// copy output to host
			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("shared mem compact output copy fail!");

			// calc # of elements for return
			int map_val;
			int r_val;
			cudaMemcpy(&r_val, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&map_val, dev_map + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("shared mem compact calc # elem fail!");

			printf("map[n-1] = %i, scan[n-1] = %i\n", map_val, r_val);

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
