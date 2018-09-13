CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Angelina Risi
  * [LinkedIn](www.linkedin.com/in/angelina-risi)
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 8GB, GTX 960M 4096MB (Personal Laptop)
  
  
## Extra Credit

### Radix Sort Implementation

Radix sort is a method of sorting data in an array from min to max using the values' binary data. This is done by sorting by the least-significant bit (LSB) first, iterating through bit sorts until the most-significant bit (MSB).  
Before we can sort, we actually need to find the dataset's maximum value. By taking the ceiling of log<sub>2</sub>(max), we can get the max number of bits representing the data, which bit is the MSB. This reduces the number of redundant iterations of sorting from the number of bits in the data type to only as many relevant ones there are in the data range. This is done on the GPU using a parallel reduction algorithm comparing pairs of values. The code is reproduced below with extra commentary.  
  
```cpp
// each thread compares a pair of integers from the input buffer 
// and selects the greater of the two
__global__ void kernFindMax(int n, int offset1, int offset2, int* buff) {
   int index = (blockDim.x * blockIdx.x) + threadIdx.x;
   
   // compute which index to compare
   int access = index * offset2 - 1;
   if (access >= n || n < 1 || access < 0) return;

   // modify in place
   if (buff[access] < buff[access - offset1]) {
   buff[access] = buff[access - offset1];
   }
}
```
```cpp
// The loop iterates deeper into the reduction until the final max value is sorted to the end
// This essentially sweeps the max value up to the root of a balanced binary tree
for (d = 1; d <= limit; d++) {
   offset1 = pow(2, d - 1);
   offset2 = pow(2, d);
   fullBlocksPerGrid.x = ((size / offset2) + blockSize) / blockSize;
   kernFindMax << <fullBlocksPerGrid, blockSize >> >(size, offset1, offset2, max_arr);
   checkCUDAError("Radix find max fail!"); // error checking
}
```  
  
To perform the sort itself efficiently, we generate a a pair of boolean buffers indicating whether the currently tested bit at that index is 0 or 1. One buffer is the true buffer, called b_arr, and the other the false buffer, called f_arr. If the bit value is 1, b_arr[index] is set to 1 and f_arr to 0, and vice versa. We save the last value of f_arr for later to compute the number of "falses" for indexing.  
  
```cpp
__global__ void kernBoolMaps(int n, int k, int* input, int* b_arr, int* f_arr) {
   int index = (blockDim.x * blockIdx.x) + threadIdx.x;
   if (index >= n) return;
 
   // retrieve the kth bit from the input val
   int bit = bitK(input[index], k);
   // flip the bit
   int fBit = flipBit(bit);

   b_arr[index] = bit; // maps bit k into b_arr
   f_arr[index] = fBit; // copy flipped value here for scan
}
```  
  
The f_arr is scanned using the work-efficient exclusive scan to generate the "false" indices, the locations to store the data values if b_arr[index] == 0 in the output array. The "true" indices, t_arr, are generated as "index - f_arr[index] + totFalse". The total false values is the last value in the scanned f_arr plus the value we stored earlier from f_arr before scanning. By using a GPU-implemented scatter function, we save the input values sorted into the output buffer. To remove the need for more intermediate buffers for each sort step, the input and output arrays are ping-ponged (switch their pointers) each sort step.  
  
```cpp
__global__ void kernRadixScatter(int n, int *out, int *in, int *b_arr, int *f_arr, int *t_arr) {
   int index = (blockDim.x * blockIdx.x) + threadIdx.x;
   if (index >= n) return;
   
   // We compute the index to access by checking the boolean in b_arr
   // If true, we use the index in t_arr (true indexing array)
   // Else, we choose the index in f_arr (false indexing array)
   // The index "access" is where in the output array the input goes to.
   int access = b_arr[index] ? t_arr[index] : f_arr[index];
   out[access] = in[index];
}
```

Once the input array has been sorted for each bit, the output is correctly sorted in order of ascending value. This implementation is intended to work on integer values, and currently operates on global device memory, bottlenecking performance. An example of a small array radix sort is depicted:
![Radix Sort Example](/img/radix_example.PNG)
  
  
### Shared Memory Work-Efficient Scan & Compact
  
An alternative implementation of the work-efficient scan using shared memory to reduce latency is included. Each block stores an array shared among its threads to store the intermediate values before outputting. By reducing global memory accesses and instead using faster shared memory, we can potentially increase thoroughput.   
Both the upsweep and downsweep are done in the same kernel as they need to both used the shared memory cache. This means we cannot dynamically change the block and threadcount as we traverse the tree as done in the global memory solution, and we must be careful to synchronize threads between write and read operations to prevent race conditions. Each block essentially performs a scan on a portion of the input data.  
To allow the merging of the blocks' solutions, while we calculate an exclusive scan through the downsweep, we save the root value of the tree in the index blockSize of the shared memory array. 
The blocks must add the root value of all previous blocks to their total to calculate the correct prefix sum values of the array. A second kernel call to do this to stitch together the blocks into the full exclusive scan is used to ensure all blocks have written their data to the device output buffers before attempting to fetch it.  
  
```cpp
__global__ void kernStitch(int n, int* in, int* sums) {
   int bx = blockIdx.x;
   int index = (blockDim.x * bx) + threadIdx.x;;

   if (bx == 0) return;
   if (index >= n) return;
   for (int i = 0; i < bx; i++) {
      in[index] += sums[i];
   }
}
```  
#### Bank Conflict Avoidance  
  
This algorithm is further improved by using offsets on the shared memory access iterators to reduce bank conflicts, events where multiple threads attempt to access a region of shared memory at the same time and thus must wait for the bus to become free. This is done by applying macros to calculate the offset on the index based on the assumed number of memory banks. These are taken from the example code in GPU Gems 3 Ch. 39 linked in the instructions.
