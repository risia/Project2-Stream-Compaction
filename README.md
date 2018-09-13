CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Angelina Risi
  * [LinkedIn](www.linkedin.com/in/angelina-risi)
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 8GB, GTX 960M 4096MB (Personal Laptop)

### (TODO: Your README)

**Radix Sort Implementation**

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

Once the input array has been sorted for each bit, the output is correctly sorted in order of ascending value. This implementation is intended to work on integer values. An example of a small array radix sort is depicted:
![Radix Sort Example](/img/radix_example.PNG)
