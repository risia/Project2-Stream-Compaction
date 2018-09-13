#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

		__global__ void kernScanDataUpSweep(int n, int offset1, int offset2, int* buff);
		__global__ void kernScanDataDownSweep(int n, int offset1, int offset2, int* buff);

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
