#pragma once

#define _USE_MATH_DEFINES
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "kernel.h"
#include "utils.h"
#include "zero_init.h"

template <typename T, int N>
DEVICE T bruteForceSummationGPU(const LeafData<N>* data, int num_elements,
                                const FLOAT* queries, int num_queries,
                                KernelFunc<T, N> kernel_func, int query_idx) {
  extern __shared__ unsigned char block_data_buffer[];

  // Cast the untyped buffer into the appropriate type for writing - this needs
  // to be done in order to avoid a compilation error, since nvcc doesn't
  // support shared variables of the same name but different types.
  // I'm not sure if shared memory alignment can cause issues here but it hasn't
  // so far in testing - keep an eye out though.
  LeafData<N>* block_data = reinterpret_cast<LeafData<N>*>(block_data_buffer);

  T total = ZeroInitializer<T>::zero();
  T c = ZeroInitializer<T>::zero();

  VectorNd<N> q;
  if (query_idx < num_queries) {
    q(0) = queries[query_idx];
    q(1) = queries[num_queries + query_idx];
    if (N == 3) {
      q(2) = queries[2 * num_queries + query_idx];
    }
  }
  
  // Loop over elements by block, with shared memory loads to speed up accesses
  // This uses query-level parallelism
  for (int i = 0; i < num_elements; i += blockDim.x) {
    int j = i + threadIdx.x;
    if (j < num_elements) {
      block_data[threadIdx.x] = data[j];
    }
    __syncthreads();
    if (query_idx < num_queries) {
      for (int b = 0; b < blockDim.x && i + b < num_elements; b++) {
        // Compensated summation - we need this to avoid numerical errors in
        // single-precision
        T kernel_val = kernel_func(block_data[b], q);
        T y = kernel_val - c;
        T t = total + y;
        c = (t - total) - y;
        total = t;
      }
    }
    __syncthreads();
  }

  return total;
}
