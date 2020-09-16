/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <stdio.h>

__global__
void reduce_min(const float* const values, float * d_out)
{
   int id = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
   int tid = threadIdx.x;
   extern __shared__ float sdata[];
   
   sdata[tid] = min(values[id], values[id + 1]);
   __syncthreads();

   for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
   {
      if (tid < s)
      {
         auto & left = sdata[tid];
         auto & right = sdata[tid + s];
         left = min(left, right);

         if (s > 1 and s & 1 == 1 and tid == s - 1)
         {
            auto & left_before = sdata[tid - 1];
            left_before = max(left, left_before);
         }
      }


       __syncthreads(); 
   }

   if (tid == 0)
   {
      d_out[blockIdx.x] = sdata[0];
   }
   

}

__global__
void reduce_max(const float* const values, float * d_out)
{
   int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
   // if (gridDim.x == 1)
   // {
   //    printf("idx %d, tidx %d, v1=%f, b2=%f, max=%f\n",idx, threadIdx.x, values[idx], values[idx + 1], max(values[idx], values[idx + 1]));
   // }
   int tid = threadIdx.x;
   extern __shared__ float sdata[];
   
   sdata[tid] = max(values[idx], values[idx + 1]);
   __syncthreads();

   for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
   {
      if (tid < s)
      {
         //  if (gridDim.x == 1)
         // {
         //    printf("idx %d, tidx %d, s=%d, v1=%f, b2=%f, max=%f\n",idx, threadIdx.x,s, sdata[tid], sdata[tid + s], max(sdata[tid], sdata[tid + s]));
         // }

         auto & left = sdata[tid];
         auto & right = sdata[tid + s];
         left = max(left, right);

         if (s > 1 and s & 1 == 1 and tid == s - 1)
         {
            auto & left_before = sdata[tid - 1];
            left_before = max(left, left_before);
         }
      } 
       
       __syncthreads();
   }

   if (tid == 0)
   {
      d_out[blockIdx.x] = sdata[0];
   }
   
}


__global__
void histogram(const float* const values, int * dout, const float max_value, const float min_value, const int bins)
{
   
   int id = threadIdx.x + blockIdx.x * blockDim.x;
   int bin = (values[id] - min_value) / (max_value - min_value) * (float) bins; 
   atomicAdd(dout + bin, 1);
   
}


__global__
void scan_add(const int* const values, unsigned int * const dout, const int size)
{
   // dout[0] = 0;

   // for (unsigned int i = 1; i < size;++i)
   // {
   //    dout[i] = values[i-1] + dout[i-1];
   // }

   external __shared__ int temp[];
   int id = threadIdx.x + blockIdx.x * blockDim.x;
   int tid = threadIdx.x;

   temp[tid] = values[2 * id] + values[2 * id + 1];
   int offset = 1;
   
   __sync_threads();

   for(unsigned int s = blockDim.x/2; i>0; i>>=1)
   {
      if (tid < s)
      {
         
      }
   }

   


}




void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
   
   
   const size_t length = numRows * numCols;
   const int maxThreads = 1024;
   const int blocks = length / maxThreads / 2;
   
   float * d_temp;
   checkCudaErrors(cudaMalloc(&d_temp,  sizeof(float) * blocks));

   reduce_min<<<blocks, maxThreads, maxThreads * sizeof(float)>>>(d_logLuminance, d_temp);   
   reduce_min<<<1, blocks / 2, blocks / 2 * sizeof(float)>>>(d_temp, d_temp);
   
   checkCudaErrors(cudaMemcpy(&min_logLum, d_temp, sizeof(float), cudaMemcpyDeviceToHost));

   reduce_max<<<blocks, maxThreads, maxThreads * sizeof(float)>>>(d_logLuminance, d_temp);   
   reduce_max<<<1, blocks / 2, blocks / 2 * sizeof(float)>>>(d_temp, d_temp);

   checkCudaErrors(cudaMemcpy(&max_logLum, d_temp, sizeof(float), cudaMemcpyDeviceToHost));
   
   checkCudaErrors(cudaFree(d_temp));
   
   std::cout << "Min value: " << min_logLum << " Max value: " << max_logLum << std::endl;
   // max_logLum = 2.18911;
   int * d_bins;
   checkCudaErrors(cudaMalloc(&d_bins,  sizeof(float) * numBins));
   checkCudaErrors(cudaMemset(d_bins, 0, numBins*sizeof(float)));

   histogram<<<blocks * 2, maxThreads>>>(d_logLuminance, d_bins, max_logLum, min_logLum, numBins);

   int h_bins[numBins];

   checkCudaErrors(cudaMemcpy(&h_bins, d_bins, numBins * sizeof(int), cudaMemcpyDeviceToHost));

   // for (auto & bin: h_bins){
   //    std::cout << bin << ' ';
   // }
   
   // std::cout << std::endl;

   scan_add<<<1,1>>>(d_bins, d_cdf, numBins);

   checkCudaErrors(cudaFree(d_bins));

   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
