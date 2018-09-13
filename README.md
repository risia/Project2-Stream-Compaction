CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Angelina Risi
  * [LinkedIn](www.linkedin.com/in/angelina-risi)
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 8GB, GTX 960M 4096MB (Personal Laptop)

### (TODO: Your README)

**Radix Sort Implementation**

Radix sort is a method of sorting data in an array from min to max using the values' binary data. This is done by sorting by the least-significant bit (LSB) first, iterating through bit sorts until the most-significant bit (MSB).  
Before we can sort, we actually need to find the dataset's maximum value. By taking the ceiling of log<sub>2</sub>(max), we can get the max number of bits representing the data. This reduces the number of redundant iterations of sorting from the number of bits in the data type to only as many relevant ones there are. This is done on the GPU using a parallel reduction algorithm comparing pairs of values.  
To perform the sort itself efficiently, we generate a a pair of boolean buffers indicating whether the currently tested bit at that index is 0 or 1. One buffer is the true buffer, called b_arr, and the other the false buffer, called f_arr. If the bit value is 1, b_arr[index] is set to 1 and f_arr to 0, and vice versa. We save the last value of f_arr for later to compute the number of "falses" for indexing.
The f_arr is scanned using the work-efficient exclusive scan to generate the "false" indices, the locations to store the data values if b_arr[index] == 0 in the output array. The "true" indices, t_arr, are generated as "index - f_arr[index] + totFalse". The total false values is the last value in the scanned f_arr plus the value we stored earlier from f_arr before scanning. By using a GPU-implemented scatter function, we save the input values sorted into the output buffer. To remove the need for more intermediate buffers for each sort step, the input and output arrays are ping-ponged (switch their pointers) each sort step.


Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

