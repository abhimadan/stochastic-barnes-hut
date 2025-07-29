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

#include "kernel.h"
#include "leaf_data.h"
#include "sampler.h"
#include "zero_init.h"


template <typename T, typename TREETYPE>
std::pair<T, int> multiLevelPrefixControlVariateSample_stackless(
    const LeafData<TREETYPE::N>* data, size_t num_elements,
    const TREETYPE* root, const TREETYPE* node_buffer, VectorNd<TREETYPE::N> q,
    KernelFunc<T, TREETYPE::N> kernel_func, std::mt19937& rng,
    int samples_per_subdomain) {
  T barnes_hut_estimate = ZeroInitializer<T>::zero();
  T mlcv_estimate = ZeroInitializer<T>::zero();

  // Count BH contribs as we iterate over the root's descendants and build the
  // control variate.
  int total_contribs = 0;

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
      // pick a data sample
      size_t idx =
          uniformSample(num_elements_contrib, rng) + subdomain->startIdx();
      FLOAT sample_prob_inv = num_elements_contrib;
      const LeafData<TREETYPE::N>& sample = data[idx];

      // New step: identify the point lower down in the tree where we stop, and
      // the corresponding probability for picking it
      FLOAT event_prob = 1;
      std::uniform_real_distribution<FLOAT> distrib(0, 1);
      const TREETYPE* prev_node = subdomain;
      const TREETYPE* cur_node = subdomain;
      T weighted_diff_all = ZeroInitializer<T>::zero();
      FLOAT cur_beta = std::max((FLOAT)1.0, cur_node->farFieldRatio(q));
      while (!cur_node->isLeaf()) {
        FLOAT far_field_ratio = cur_node->farFieldRatio(q);
        // Our method
        FLOAT traverse_prob = std::min(cur_beta/far_field_ratio, (FLOAT)1.0);
        // Fixed probability
        /* FLOAT traverse_prob = 0.5; */
        // No RR
        /* FLOAT traverse_prob = 1.0; */
        FLOAT p = distrib(rng);
        if (p > traverse_prob) {
          event_prob *= 1 - traverse_prob;
          break;
        }
        event_prob *= traverse_prob;
        prev_node = cur_node;
        cur_node = cur_node->containedInChild(sample.position, node_buffer);
        cur_beta = std::max(cur_beta, far_field_ratio);

        T prev_result = kernel_func(prev_node->leaf_data(), q);
        T cur_result = ZeroInitializer<T>::zero();
        for (auto it = prev_node->childIter(); it; ++it) {
          const TREETYPE* child = prev_node->child(*it, node_buffer);
          if (child->isLeaf()) {
            size_t start = child->startIdx();
            size_t end = child->endIdx();
            for (size_t i = start; i < end; i++) {
              cur_result += kernel_func(data[i], q);
              total_contribs++;
            }
          } else {
            // Otherwise, add kernel contribution
            cur_result += kernel_func(child->leaf_data(), q);
            total_contribs++;
          }
        }
        T diff = cur_result - prev_result;
        T weighted_diff = sample_prob_inv /
                          (event_prob * prev_node->numContainedPoints()) * diff;

        weighted_diff_all += weighted_diff;
      }
      subdomain_total += weighted_diff_all;
    }
    mlcv_estimate += subdomain_total / samples_per_subdomain;
  }

  return std::make_pair(barnes_hut_estimate + mlcv_estimate, total_contribs);
}
