/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include <stream_compaction/shared_mem.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 15; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

	/* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
	onesArray(SIZE, c);
	printDesc("1s array for finding bugs");
	StreamCompaction::Naive::scan(SIZE, c, a);
	printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

	zeroArray(SIZE, c);
	printDesc("Find max, power-of-two");
	int max = StreamCompaction::Radix::max(SIZE, a);
	printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	printf("max = %i\n", max);

	zeroArray(SIZE, c);
	printDesc("Find max, non-power-of-two");
	max = StreamCompaction::Radix::max(NPOT, a);
	printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	printf("max = %i\n", max);

	zeroArray(SIZE, c);
	//int radix_tst[8] = { 4, 7, 2, 6, 3, 5, 1, 0 };
	printDesc("Radix sort, power-of-two");
	StreamCompaction::Radix::sort(SIZE, c, a);
	printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	printArray(SIZE, c, true);

	zeroArray(SIZE, c);
	printDesc("Radix sort, non-power-of-two");
	StreamCompaction::Radix::sort(NPOT, c, a);
	printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	printArray(NPOT, c, true);

	zeroArray(SIZE, c);
	int radix_tst[8] = { 4, 7, 2, 6, 3, 5, 1, 0 };
	printDesc("Radix example sort");
	printf("Test input array:\n");
	printArray(8, radix_tst, true);
	StreamCompaction::Radix::sort(8, c, radix_tst);
	printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	printf("Sorted Output:\n");
	printArray(8, c, true);

	zeroArray(SIZE, c);
	printDesc("Shared Memory Efficient Scan, power-of-two");
	StreamCompaction::SharedMem::scan(SIZE, c, a);
	printElapsedTime(StreamCompaction::SharedMem::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	printArray(SIZE, c, true);
	printCmpResult(SIZE, b, c);

	zeroArray(SIZE, c);
	printDesc("Shared Memory Efficient Scan, non-power-of-two");
	StreamCompaction::SharedMem::scan(NPOT, c, a);
	printElapsedTime(StreamCompaction::SharedMem::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	printArray(NPOT, c, true);
	printCmpResult(NPOT, b, c);

	//zeroArray(SIZE, c);
	//printDesc("Shared Memory Efficient Scan, power-of-two");
	//StreamCompaction::SharedMem::scan(8, c, radix_tst);
	//printElapsedTime(StreamCompaction::SharedMem::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	//printArray(8, c, true);

	//zeroArray(SIZE, c);
	//printDesc("Shared Memory Efficient Scan, non-power-of-two");
	//StreamCompaction::SharedMem::scan(7, c, radix_tst);
	//printElapsedTime(StreamCompaction::SharedMem::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	//printArray(7, c, true);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

	zeroArray(SIZE, c);
	printDesc("Shared Memory work-efficient compact, power-of-two");
	count = StreamCompaction::SharedMem::compact(SIZE, c, a);
	printElapsedTime(StreamCompaction::SharedMem::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	//printArray(count, c, true);
	printCmpLenResult(count, expectedCount, b, c);

	zeroArray(SIZE, c);
	printDesc("Shared Memory work-efficient compact, non-power-of-two");
	count = StreamCompaction::SharedMem::compact(NPOT, c, a);
	printElapsedTime(StreamCompaction::SharedMem::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	//printArray(count, c, true);
	printCmpLenResult(count, expectedNPOT, b, c);


	// loop 100 tests to get avgs
	// make time variables
	float time_N_S_POT = 0.0f; // naive pow 2 scan
	float time_N_S_NPOT = 0.0f; // naive not pow 2 scan
	float time_WE_S_POT = 0.0f; //
	float time_WE_S_NPOT = 0.0f; //
	float time_WE_C_POT = 0.0f; //
	float time_WE_C_NPOT = 0.0f; //
	float time_SM_S_POT = 0.0f; //
	float time_SM_S_NPOT = 0.0f; //
	float time_T_S_POT = 0.0f; //
	float time_T_S_NPOT = 0.0f; //
	float time_R_S_POT = 0.0f; //
	float time_R_S_NPOT = 0.0f; //
	float time_CPU_S_POT = 0.0f;
	float time_CPU_S_NPOT = 0.0f;
	float time_CPU_C_S = 0.0f;
	float time_CPU_C_NS = 0.0f;
	float time_CPU_C_S_NPOT = 0.0f;
	float time_CPU_C_NS_NPOT = 0.0f;

	for (int i = 0; i < 100; i++) {
		// gen array
		genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
		a[SIZE - 1] = 0;

		// cpu scan POT
		zeroArray(SIZE, b);
		StreamCompaction::CPU::scan(SIZE, b, a);
		time_CPU_S_POT += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

		// cpu scan POT
		zeroArray(SIZE, b);
		StreamCompaction::CPU::scan(NPOT, b, a);
		time_CPU_S_NPOT += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

		// cpu compact w/o scan
		zeroArray(SIZE, b);
		StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
		time_CPU_C_NS += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

		// cpu compact w/o scan
		zeroArray(SIZE, b);
		StreamCompaction::CPU::compactWithoutScan(NPOT, b, a);
		time_CPU_C_NS_NPOT += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

		// cpu compact w/ scan
		zeroArray(SIZE, b);
		StreamCompaction::CPU::compactWithScan(SIZE, b, a);
		time_CPU_C_S += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

		// cpu compact w/ scan
		zeroArray(SIZE, b);
		StreamCompaction::CPU::compactWithScan(NPOT, b, a);
		time_CPU_C_S_NPOT += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

		// Naive scan POT
		zeroArray(SIZE, b);
		StreamCompaction::Naive::scan(SIZE, b, a);
		time_N_S_POT += StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();

		// Naive scan N_POT
		zeroArray(SIZE, b);
		StreamCompaction::Naive::scan(NPOT, b, a);
		time_N_S_NPOT += StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();

		// WE scan POT
		zeroArray(SIZE, b);
		StreamCompaction::Efficient::scan(SIZE, b, a);
		time_WE_S_POT += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();

		// WE scan N_POT
		zeroArray(SIZE, b);
		StreamCompaction::Efficient::scan(NPOT, b, a);
		time_WE_S_NPOT += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();

		// WE compact POT
		zeroArray(SIZE, b);
		StreamCompaction::Efficient::compact(SIZE, b, a);
		time_WE_C_POT += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();

		// WE compact N_POT
		zeroArray(SIZE, b);
		StreamCompaction::Efficient::compact(NPOT, b, a);
		time_WE_C_NPOT += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();

		// SM scan POT
		zeroArray(SIZE, b);
		StreamCompaction::SharedMem::scan(SIZE, b, a);
		time_SM_S_POT += StreamCompaction::SharedMem::timer().getGpuElapsedTimeForPreviousOperation();

		// SM scan N_POT
		zeroArray(SIZE, b);
		StreamCompaction::SharedMem::scan(NPOT, b, a);
		time_SM_S_NPOT += StreamCompaction::SharedMem::timer().getGpuElapsedTimeForPreviousOperation();

		// Thrust scan POT
		zeroArray(SIZE, b);
		StreamCompaction::Thrust::scan(SIZE, b, a);
		time_T_S_POT += StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();

		// Thrust scan N_POT
		zeroArray(SIZE, b);
		StreamCompaction::Thrust::scan(NPOT, b, a);
		time_T_S_NPOT += StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();

		// Radix sort POT
		zeroArray(SIZE, b);
		StreamCompaction::Radix::sort(SIZE, b, a);
		time_R_S_POT += StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation();

		// Radic sort N_POT
		zeroArray(SIZE, b);
		StreamCompaction::Radix::sort(NPOT, b, a);
		time_R_S_NPOT += StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation();

	}

	// print avg times
	printf("CPU Scan POT: %f\n", time_CPU_S_POT / 100.0f);
	printf("CPU Scan NPOT: %f\n", time_CPU_S_NPOT / 100.0f);
	printf("CPU Compact POT: %f\n", time_CPU_C_NS / 100.0f);
	printf("CPU Scan Compact NPOT: %f\n", time_CPU_C_S_NPOT / 100.0f);
	printf("CPU Compact NPOT: %f\n", time_CPU_C_NS_NPOT / 100.0f);
	printf("CPU Scan Compact POT: %f\n", time_CPU_C_S / 100.0f);
	printf("Naive POT: %f\n", time_N_S_POT / 100.0f);
	printf("Naive NPOT: %f\n", time_N_S_NPOT / 100.0f);
	printf("WE Scan POT: %f\n", time_WE_S_POT / 100.0f);
	printf("WE Scan NPOT: %f\n", time_WE_S_NPOT / 100.0f);
	printf("WE Comp POT: %f\n", time_WE_C_POT / 100.0f);
	printf("WE Comp NPOT: %f\n", time_WE_C_NPOT / 100.0f);
	printf("SM Scan POT: %f\n", time_SM_S_POT / 100.0f);
	printf("SM Scan NPOT: %f\n", time_SM_S_NPOT / 100.0f);
	printf("Thrust Scan POT: %f\n", time_T_S_POT / 100.0f);
	printf("Thrust Scan NPOT: %f\n", time_T_S_NPOT / 100.0f);
	printf("Radix POT: %f\n", time_R_S_POT / 100.0f);
	printf("Radix NPOT: %f\n", time_R_S_NPOT / 100.0f);



    system("pause"); // stop Win32 console from closing on exit
	delete[] a;
	delete[] b;
	delete[] c;
}
