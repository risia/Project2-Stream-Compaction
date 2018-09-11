#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

			if (n < 1) return; // no data
			odata[0] = 0; // Exclusive scan
			for (int i = 1; i < n; i++) {
				odata[i] = idata[i-1] + odata[i-1];
			}

	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

			if (n < 1) return -1; // no data
			int nElem = 0; // remaining elements after compact

			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[nElem] = idata[i];
					nElem++;
				}
			}

	        timer().endCpuTimer();
            return nElem;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {

			int *map_data = (int*)(malloc(n * sizeof(int)));
			int *scan_data = (int*)(malloc(n * sizeof(int)));

	        timer().startCpuTimer();

			//map
			int i;
			for (i = 0; i < n; i++) {
				if (idata[i] != 0) {
					map_data[i] = 1;
				}
				else map_data[i] = 0;
			}

			// scan
			scan_data[0] = 0; // Exclusive scan
			for (int i = 1; i < n; i++) {
				scan_data[i] = map_data[i - 1] + scan_data[i - 1];
			}

			int r_val;

			// scatter
			for (i = 0; i < n; i++) {
				if (map_data[i] == 1) {
					r_val = scan_data[i];
					odata[r_val] = idata[i];
				}
			}
			r_val++;

	        timer().endCpuTimer();

			free(map_data);
			free(scan_data);

            return r_val;
        }
    }
}
