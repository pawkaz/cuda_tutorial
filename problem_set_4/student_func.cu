//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

  __global__
void histogram_10(const unsigned int* const dvalues, unsigned int* const dout, const size_t  numElements)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (numElements * 10 <= id) return;
  int offset = id / numElements;
  unsigned int bin;
  unsigned int base = pow(10, offset) + .5;

  if (offset < 9)
  {
    bin = (dvalues[id % numElements] % (base * 10)) / base;
  }
  else
  {
    bin = dvalues[id % numElements] / base;
  }

  // if (id == 18)
  // {
  //   printf("Check value %d, base %d, bin %d\n", dvalues[id % numElements], base, bin);
  // }

  atomicAdd(dout + bin + 10 * offset, 1);
}

__global__
void scan_add_serial(const unsigned int* const values, unsigned int * const dout, const size_t numElements)
{
  int offset = threadIdx.x * numElements;
  dout[offset] = 0;
  for(int i=1;i<numElements;++i)
  {
    dout[offset + i] = dout[offset + i-1] +  values[offset + i-1];
  }
}

__global__
void predicate_bin(const unsigned int* const dvalues, unsigned int * const dout, const size_t offset, const size_t numElements)
{
  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (numElements * 10 <= id) return;
  int bin_gt = id / numElements;
  
  unsigned int bin;
  unsigned int base = pow(10, offset) + .5;

  if (offset < 9)
  {
    bin = (dvalues[id % numElements] % (base * 10)) / base;
  }
  else
  {
    bin = dvalues[id % numElements] / base;
  }

  dout[id] = bin_gt == bin ? 1 : 0;

}

__global__
void calc_new_idxs(const unsigned int* const dvalues, const unsigned int* const d_global, const unsigned int* const d_local, unsigned int* const d_out, const unsigned int offset, const size_t numElements)
{
  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (numElements <= id) return;
  
  unsigned int bin;
  unsigned int base = pow(10, offset) + .5;

  if (offset < 9)
  {
    bin = (dvalues[id] % (base * 10)) / base;
  }
  else
  {
    bin = dvalues[id] / base;
  }
  int new_idx = d_global[offset * 10 + bin] + d_local[bin * numElements + id];
  d_out[new_idx] = id;

}

__global__
void move_idxs(const unsigned int* const values, unsigned int * const dout, unsigned int * const idxs, const size_t numElements)
{
  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (numElements <= id) return;

  dout[id] = values[idxs[id]];

}

 __global__
void scan_add(const unsigned int* const values, unsigned int * const dout, const size_t numElements)
{
  extern __shared__ int temp[];
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  temp[2* tid] = 2*id < numElements ? values[2*id] : 0;
  temp[2* tid+1] = 2 * id + 1 ? values[2*id + 1]: 0;

  for (int s = 1; blockDim.x / s > 0; s<<=1)
  {
    int curr_idx = (tid + 1) * (s << 1) - 1;

    if (tid >= (blockDim.x / s))
    { 
      continue;
    }

    // printf("Tid %d, curr_idx %d, step size %d\n", tid, curr_idx, s);
    
    temp[curr_idx] += temp[curr_idx - s];
    __syncthreads();
  }


  for (int s = blockDim.x; s > 0; s>>=1)
  {
    int curr_idx = (tid + 1) * (s << 1) - 1;

    if (tid >= (blockDim.x / s))
    { 
      continue;
    }

    if (tid != 0)
    {
      auto right_temp = temp[curr_idx];
      temp[curr_idx] += temp[curr_idx - s];
      temp[curr_idx - s] = right_temp;
    }
    else
    {
      temp[curr_idx] = temp[curr_idx - s];
      temp[curr_idx - s] = 0; 
    }
    
    __syncthreads();
    
  }
  if ((2*id) < numElements)
  {
    dout[2*id] = temp[2 * tid];
  }

  if ((2*id + 1)< numElements)
  {
    dout[2*id + 1] = temp[2 * tid + 1];
  }


}

template<class T>
void print_data(T *p, int size, int numElems, int stride=10)
{
  T h_test_data[numElems];  
  checkCudaErrors(cudaMemcpy(h_test_data, p, size, cudaMemcpyDeviceToHost));
  // std::cout << "Num elements: " << numElems << std::endl;
  for (size_t i=0;i<numElems;++i){
    std::cout << h_test_data[i] << ' ';
    if (i % stride == stride - 1)
    {
      std::cout << std::endl;
    }
  }
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE

  unsigned int * d_hist;
  checkCudaErrors(cudaMalloc(&d_hist,  sizeof(int) * 100));
  int K = 1024;
  int blocks = (numElems * 10) / K + 1;

  // Test
  // unsigned int * d_vals_test;
  // unsigned int * d_out_vals_test;
  // checkCudaErrors(cudaMalloc(&d_vals_test,  sizeof(int) * 8));
  // checkCudaErrors(cudaMalloc(&d_out_vals_test,  sizeof(int) * 8));
  // unsigned int h_vals_test[8]{1, 22, 16, 100, 200, 300, 1000, 8000};
  // checkCudaErrors(cudaMemcpy(d_vals_test, h_vals_test, 8 * sizeof(unsigned int), cudaMemcpyHostToDevice));

  // histogram_10<<<1, 80>>>(d_vals_test, d_hist, 8); 
  histogram_10<<<blocks, K>>>(d_inputVals, d_hist, numElems); 
  // std::cout << "Histogram" << std::endl;
  // print_data(d_hist, 100 * sizeof(int), 100, 10);

  unsigned int * d_hist_cdf;
  checkCudaErrors(cudaMalloc(&d_hist_cdf,  sizeof(unsigned int) * 100));
  scan_add_serial<<<1, 10>>>(d_hist, d_hist_cdf, 10); 
  // std::cout << "CDF Histogram" << std::endl;
  // print_data(d_hist_cdf, 100 * sizeof(int), 100, 10);

  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));

  // checkCudaErrors(cudaMemcpy(d_out_vals_test, d_vals_test, sizeof(int) * 8, cudaMemcpyDeviceToDevice));

  unsigned int * d_predicate_mask;
  unsigned int * d_relative_offset;
  unsigned int * d_new_idxs;
  unsigned int * d_temp;

  checkCudaErrors(cudaMalloc(&d_predicate_mask,  sizeof(unsigned int) * numElems * 10));
  checkCudaErrors(cudaMalloc(&d_relative_offset,  sizeof(unsigned int) * numElems * 10));
  checkCudaErrors(cudaMalloc(&d_new_idxs,  sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&d_temp,  sizeof(unsigned int) * numElems));

  // checkCudaErrors(cudaMalloc(&d_predicate_mask,  sizeof(int) * 8 * 10));
  // checkCudaErrors(cudaMalloc(&d_relative_offset,  sizeof(int) * 8 * 10));
  // checkCudaErrors(cudaMalloc(&d_new_idxs,  sizeof(int) * 8));
  // checkCudaErrors(cudaMalloc(&d_temp,  sizeof(int) * 8));

  for (int i = 0; i<10; ++i)
  {
    // printf("Iteration %d: ", i);
    // print_data(d_out_vals_test, 8 * sizeof(int), 8, 8);
    predicate_bin<<<blocks, K>>>(d_outputVals, d_predicate_mask, i, numElems);
    
    scan_add_serial<<<1, 10>>>(d_predicate_mask, d_relative_offset, numElems);
    calc_new_idxs<<<blocks, K>>>(d_outputVals, d_hist_cdf, d_relative_offset, d_new_idxs, i, numElems);
    move_idxs<<<blocks, K>>>(d_outputVals, d_temp, d_new_idxs, numElems);
    cudaMemcpy(d_outputVals, d_temp, numElems * sizeof(int),cudaMemcpyDeviceToDevice );
    move_idxs<<<blocks, K>>>(d_outputPos, d_temp, d_new_idxs, numElems);
    cudaMemcpy(d_outputPos, d_temp, numElems * sizeof(int),cudaMemcpyDeviceToDevice );

    // print_data(d_predicate_mask, 80 * sizeof(int), 80, 8);
    // print_data(d_relative_offset, 80 * sizeof(int), 80, 8);
    // print_data(d_new_idxs, 8 * sizeof(int), 8, 8);
    
  }

 
  // checkCudaErrors(cudaFree(d_hist));
  // checkCudaErrors(cudaFree(d_hist_cdf));
  // checkCudaErrors(cudaFree(d_predicate_mask));
  // checkCudaErrors(cudaFree(d_relative_offset));
  // checkCudaErrors(cudaFree(d_new_idxs));
  // checkCudaErrors(cudaFree(d_temp));
  


}
