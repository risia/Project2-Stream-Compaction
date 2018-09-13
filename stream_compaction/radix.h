#pragma once

#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
	namespace Radix {
		StreamCompaction::Common::PerformanceTimer& timer();

		void sort(int n, int *odata, const int *idata);

		int max(int n, int* idata);
	}
}
