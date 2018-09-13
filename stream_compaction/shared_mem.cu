#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "shared_mem.h"

#define blockSize 256

// for reducing bank conflicts
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

namespace StreamCompaction {
    namespace SharedMem {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		__global__ void kernScanDataShared(int n, int* in, int* out, int* sums) {
			// init shared mem for block, could improve latency
			__shared__ int sBuf[blockSize];

			int tx = threadIdx.x;
			int index = (blockDim.x * blockIdx.x) + tx;

			// copy used vals to shared mem
			sBuf[tx + CONFLICT_FREE_OFFSET(tx)] = (index < n) ? in[index] : 0;

			__syncthreads(); // avoid mem issues

			int offset; // step size
			int access; // shared buffer access index
			int a2;

			// Upsweep
			for (offset = 1; offset < blockSize; offset *=2) {
				access = (2 * offset * (tx + 1)) - 1;
				a2 = access - offset;
				a2 += CONFLICT_FREE_OFFSET(a2);
				access += CONFLICT_FREE_OFFSET(access);
				if (access < blockSize) sBuf[access] += sBuf[a2];
				__syncthreads(); // avoid mem issues
			}

			// prepare array for downsweep
			if (tx == blockSize - 1 + CONFLICT_FREE_OFFSET(blockSize - 1)) {
				sums[blockIdx.x] = sBuf[tx];
				sBuf[tx] = 0;
			}
			__syncthreads();
			if (index >= n - 1) sBuf[tx + CONFLICT_FREE_OFFSET(tx)] = 0;
			__syncthreads(); // avoid mem issues

			// Downsweep (inclusive)
			// do exclusive downsweep
			int temp;

			for (offset = blockSize; offset >= 1; offset /= 2) {
				access = (2 * offset * (tx + 1)) - 1;
				a2 = access - offset;
				a2 += CONFLICT_FREE_OFFSET(a2);
				access += CONFLICT_FREE_OFFSET(access);
				if (access < blockSize) {
					temp = sBuf[a2]; // store left child
					sBuf[a2] = sBuf[access]; // swap
					sBuf[access] += temp; // add
				}
				__syncthreads(); // avoid mem issues
			}
			
			// write to dev memory
			if (index < n) {
				out[index] = sBuf[tx + CONFLICT_FREE_OFFSET(tx)];
			}
		}

		__global__ void kernStitch(int n, int* in, int* sums) {
			int bx = blockIdx.x;
			int index = (blockDim.x * bx) + threadIdx.x;;

			if (bx == 0) return;
			if (index >= n) return;
			for (int i = 0; i < bx; i++) {
				in[index] += sums[i];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int mod = n % blockSize;
			int size = n;

			if (mod != 0) size+= blockSize - mod;

			dim3 fullBlocksPerGrid((size + (blockSize - 1))/ blockSize);

			int* dev_out; // data to output
			int* dev_in; // input data

			int* dev_sums;

			cudaMalloc((void**)&dev_in, n * sizeof(int));
			cudaMalloc((void**)&dev_out, n * sizeof(int));
			cudaMalloc((void**)&dev_sums, fullBlocksPerGrid.x * sizeof(int));

			// copy input data to device
			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			cudaMemset(dev_out, 0, n * sizeof(int));
			checkCUDAError("initializing shared mem scan data buff fail!");

			timer().startGpuTimer();

			kernScanDataShared<<<fullBlocksPerGrid, blockSize>>>(n, dev_in, dev_out, dev_sums);
			checkCUDAError("shared mem scan fail!");

			kernStitch << <fullBlocksPerGrid, blockSize >> >(n, dev_out, dev_sums);
			checkCUDAError("shared mem scan stitch fail!");


			timer().endGpuTimer();

			// copy out data
			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("shared mem scan output copy fail!");

			cudaFree(dev_out);
			cudaFree(dev_in);
			cudaFree(dev_sums);
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

			int* dev_sums;

			int mod = n % blockSize;
			int size = n;
			if (mod != 0) size += blockSize - mod;

			dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);

			// allocate memory
			cudaMalloc((void**)&dev_in, n * sizeof(int));
			cudaMalloc((void**)&dev_map, n * sizeof(int));
			cudaMalloc((void**)&dev_out, n * sizeof(int));
			cudaMalloc((void**)&dev_scan, n * sizeof(int));

			cudaMalloc((void**)&dev_sums, fullBlocksPerGrid.x * sizeof(int));
			checkCUDAError("shared mem compact malloc fail!");

			cudaMemset(dev_scan, 0, n * sizeof(int));
			checkCUDAError("initializing shared mem scan data buff fail!");

			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice); // copy input data
			checkCUDAError("initializing w-e compact data buffs fail!");

			

            timer().startGpuTimer();
            // map
			fullBlocksPerGrid.x = ((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, dev_map, dev_in);
			checkCUDAError("w-e compact bool mapping fail!");

			// scan the map
			fullBlocksPerGrid.x = ((size + blockSize - 1) / blockSize);
			kernScanDataShared << <fullBlocksPerGrid, blockSize >> >(n, dev_map, dev_scan, dev_sums);
			checkCUDAError("shared mem scan fail!");

			kernStitch << <fullBlocksPerGrid, blockSize >> >(n, dev_scan, dev_sums);
			checkCUDAError("shared mem scan stitch fail!");

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

			r_val += map_val;

			// cleanup
			cudaFree(dev_in);
			cudaFree(dev_map);
			cudaFree(dev_out);
			cudaFree(dev_scan);
			cudaFree(dev_sums);

            return r_val;
        }
    }
}
