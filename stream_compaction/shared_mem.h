#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace SharedMem {
        StreamCompaction::Common::PerformanceTimer& timer();

		__global__ void kernScanDataShared(int n, int* in, int* out);

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
