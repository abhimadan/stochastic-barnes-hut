#pragma once

#define _USE_MATH_DEFINES
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <curand_kernel.h>
#include <cub/cub.cuh>

#include "kernel.h"
#include "leaf_data.h"
#include "sampler.h"
#include "utils.h"
#include "zero_init.h"

// "Query sampling" version
template <typename T, typename TREETYPE>
DEVICE T multiLevelPrefixControlVariateSampleGPUQuery(
    const LeafData<TREETYPE::N>* data, size_t num_elements,
    const TREETYPE* root, const TREETYPE* node_buffer,
    const LeafData<TREETYPE::N>* contrib_data, VectorNd<TREETYPE::N> q,
    KernelFunc<T, TREETYPE::N> kernel_func, int samples_per_subdomain,
    int seed) {
  T barnes_hut_estimate = ZeroInitializer<T>::zero();
  T mlcv_estimate = ZeroInitializer<T>::zero();

  // Not returned to caller in device code (but could write to a buffer if we
  // really want this, so keep track of it)
  int total_contribs = 0;

  curandState s;
  curandState s2;
  curand_init(seed, 0, 0, &s);
  curand_init(blockIdx.x, 0, 0, &s2); // paths only, synced between all threads

  // I think this comparison operator is fine since we know endIdx will always
  // bound every index passed in, but keep an eye out
  auto cmp = [] DEVICE (const TREETYPE* node, size_t idx) {
    return node->endIdx() <= idx;
  };

  for (auto sub_it = root->childIter(); sub_it; ++sub_it) {
    const TREETYPE* subdomain = root->child(*sub_it, node_buffer);
    if (subdomain->isLeaf()) {
      size_t start = subdomain->startIdx();
      size_t end = subdomain->endIdx();
      for (size_t i = start; i < end; i++) {
        barnes_hut_estimate += kernel_func(data[i], q);
        total_contribs++;
      }
    } else {
      // Otherwise, add kernel contribution
      barnes_hut_estimate += kernel_func(subdomain->leaf_data(), q);
      total_contribs++;
    }

    size_t num_elements_contrib = subdomain->endIdx() - subdomain->startIdx();
    T subdomain_total = ZeroInitializer<T>::zero();
    for (int ss = 0; ss < samples_per_subdomain; ss++) {
      // pick a data sample, shared between all threads
      size_t idx = min((size_t)(num_elements_contrib * curand_uniform(&s2) +
                                subdomain->startIdx()),
                       subdomain->endIdx() - 1);
      FLOAT sample_prob_inv = num_elements_contrib;
      VectorNd<TREETYPE::N> sample_pos = data[idx].position;

      const TREETYPE* prev_node = subdomain;
      const TREETYPE* cur_node = subdomain;

      // New step: identify the point lower down in the tree where we stop, and
      // the corresponding probability for picking it
      FLOAT event_prob = FP(1.0);
      int cur_depth = 0;
      T weighted_diff_all = ZeroInitializer<T>::zero();
      FLOAT cur_beta = fmax(FP(1.0), cur_node->farFieldRatio(q));
      while (!cur_node->isLeaf()) {
        FLOAT far_field_ratio = cur_node->farFieldRatio(q);
        // Our method
        FLOAT traverse_prob = fmin(cur_beta/far_field_ratio, FP(1.0));
        // Fixed prob
        //FLOAT traverse_prob = 0.5;
        // No RR
        //FLOAT traverse_prob = 1.0;
        FLOAT p = curand_uniform(&s);
        /* FLOAT p = 1; // no randomization */
        if (p > traverse_prob) {
          event_prob *= FP(1.0) - traverse_prob;
          break;
        }
        event_prob *= traverse_prob;

        prev_node = cur_node;
        cur_depth++;
        cur_node = cur_node->containedInChild(sample_pos, node_buffer);

        cur_beta = fmax(cur_beta, far_field_ratio);

        T prev_result = kernel_func(prev_node->leaf_data(), q);
        T cur_result = ZeroInitializer<T>::zero();
        int first_child_idx = prev_node->firstChildIdx();
        for (int child_offset = 0; child_offset < prev_node->num_children();
             child_offset++) {
          int child_idx = first_child_idx + child_offset;
          const TREETYPE* child = &node_buffer[child_idx];
          if (child->isLeaf()) {
            size_t start = child->startIdx();
            size_t end = child->endIdx();
            for (size_t i = start; i < end; i++) {
              cur_result += kernel_func(data[i], q);
              total_contribs++;
            }
          } else {
            // Otherwise, add kernel contribution
            cur_result += kernel_func(contrib_data[child_idx], q);
            total_contribs++;
          }
        }
        T diff = cur_result - prev_result;
        T weighted_diff =
            sample_prob_inv /
            (event_prob * prev_node->numContainedPoints()) *
            diff;

        weighted_diff_all += weighted_diff;
      }
      subdomain_total += weighted_diff_all;
    }
    mlcv_estimate += subdomain_total / samples_per_subdomain;
  }

  return barnes_hut_estimate + mlcv_estimate;
}
