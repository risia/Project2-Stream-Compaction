#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "shared_mem.h"

#define blockSize 128

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
		__global__ void kernScanBlockSum(int n, int* sum_buf) {

			int index = (blockDim.x * blockIdx.x) + threadIdx.x;

			int offset;
			int access;
			int a2;

			int temp;

			// Upsweep
			for (offset = 1; offset < blockSize; offset *= 2) {
				access = (2 * offset * (index + 1)) - 1;
				a2 = access - offset;
				if (access < blockSize) sum_buf[access] += sum_buf[a2];
				__syncthreads(); // avoid mem issues
			}
			if (index >= n - 1) sum_buf[index] = 0;
			__syncthreads(); // avoid mem issues

			//downsweep
			for (offset = blockSize; offset >= 1; offset /= 2) {
				access = (2 * offset * (index + 1)) - 1;
				a2 = access - offset;
				if (access < blockSize) {
					temp = sum_buf[a2]; // store left child
					sum_buf[a2] = sum_buf[access]; // swap
					sum_buf[access] += temp; // add
				}
				__syncthreads(); // avoid mem issues
			}

		}

		__global__ void kernScanDataShared(int n, int* in, int* out, int* sums) {
			// init shared mem for block, could improve latency
			__shared__ int sBuf[blockSize];

			int tx = threadIdx.x;
			int index = (blockDim.x * blockIdx.x) + tx;

			int off_tx = tx + CONFLICT_FREE_OFFSET(tx);

			// copy used vals to shared mem
			sBuf[off_tx] = (index < n) ? in[index] : 0;

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
				sums[blockIdx.x] = sBuf[off_tx];
				sBuf[off_tx] = 0;
			}
			__syncthreads();
			if (index >= n - 1) sBuf[off_tx] = 0;
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
				out[index] = sBuf[off_tx];
			}
		}

		__global__ void kernStitch(int n, int* in, int* sums) {
			int bx = blockIdx.x;
			int index = (blockDim.x * bx) + threadIdx.x;;

			if (bx == 0) return;
			if (index >= n) return;
			in[index] += sums[bx];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int mod = n % blockSize;
			int size = n;

			if (mod != 0) size+= blockSize - mod;

			int num_blocks = size / blockSize;

			dim3 fullBlocksPerGrid(num_blocks);

			int* dev_out; // data to output
			int* dev_in; // input data

			int* dev_sums;

			int x;

			cudaMalloc((void**)&dev_in, n * sizeof(int));
			cudaMalloc((void**)&dev_out, n * sizeof(int));
			cudaMalloc((void**)&dev_sums, num_blocks * sizeof(int));

			// copy input data to device
			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			cudaMemset(dev_out, 0, n * sizeof(int));
			checkCUDAError("initializing shared mem scan data buff fail!");

			timer().startGpuTimer();

			// scan blocks of data
			kernScanDataShared<<<fullBlocksPerGrid, blockSize>>>(n, dev_in, dev_out, dev_sums);
			checkCUDAError("shared mem scan fail!");

			

			fullBlocksPerGrid.x = (num_blocks + blockSize - 1) / blockSize;
			// scan sums from blocks
			kernScanBlockSum << <fullBlocksPerGrid, blockSize >> >(num_blocks, dev_sums);
			checkCUDAError("shared mem block scan fail!");

			fullBlocksPerGrid.x = num_blocks;
			kernStitch << <fullBlocksPerGrid, blockSize >> >(n, dev_out, dev_sums);
			checkCUDAError("shared mem scan stitch fail!");


			timer().endGpuTimer();

			// copy out data
			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("shared mem scan output copy fail!");

			for (int i = 0; i < num_blocks; i++) {
				cudaMemcpy(&x, dev_sums + i, sizeof(int), cudaMemcpyDeviceToHost);
				printf("Sum %i: %i\n", i, x);
			}

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

			int num_blocks = size / blockSize;

			dim3 fullBlocksPerGrid(num_blocks);

			// allocate memory
			cudaMalloc((void**)&dev_in, n * sizeof(int));
			cudaMalloc((void**)&dev_map, n * sizeof(int));
			cudaMalloc((void**)&dev_out, n * sizeof(int));
			cudaMalloc((void**)&dev_scan, n * sizeof(int));

			cudaMalloc((void**)&dev_sums, num_blocks * sizeof(int));
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
			fullBlocksPerGrid.x = num_blocks;
			kernScanDataShared << <fullBlocksPerGrid, blockSize >> >(n, dev_map, dev_scan, dev_sums);
			checkCUDAError("shared mem scan fail!");

			int r_val;
			cudaMemcpy(&r_val, dev_sums + num_blocks - 1, sizeof(int), cudaMemcpyDeviceToHost);

			fullBlocksPerGrid.x = (num_blocks + blockSize - 1) / blockSize;
			// scan sums from blocks
			kernScanBlockSum << <fullBlocksPerGrid, blockSize >> >(num_blocks, dev_sums);
			checkCUDAError("shared mem block scan fail!");

			fullBlocksPerGrid.x = num_blocks;
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
			int r_val2;
			cudaMemcpy(&r_val2, dev_sums + num_blocks -1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("shared mem compact calc # elem fail!");

			// cleanup
			cudaFree(dev_in);
			cudaFree(dev_map);
			cudaFree(dev_out);
			cudaFree(dev_scan);
			cudaFree(dev_sums);

            return r_val + r_val2;
        }
    }
}
