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

#include "barnes_hut.h"
#include "kernel.h"
#include "leaf_data.h"
#include "zero_init.h"

template <typename T, typename TREETYPE>
DEVICE T evaluateBarnesHutGPUVote(const LeafData<TREETYPE::N>* data,
                                  const TREETYPE* root,
                                  const TREETYPE* node_buffer,
                                  VectorNd<TREETYPE::N> q, FLOAT beta,
                                  KernelFunc<T, TREETYPE::N> kernel_func) {
  T total(ZeroInitializer<T>::zero());
  const TREETYPE* node = root;
  TraversalState state = TraversalState::FromParent;
  unsigned mask = 0xFFFFFFFF;  // I think this only looks at non-exited threads
                               // so it should be fine...
  while (true) {
    if (state != TraversalState::FromChild) {
      // perform the check at the node and decide whether to go down, go to a
      // sibling, or go back up
      if (__all_sync(mask, node->useFarField(q, beta))) {
        // Add contribution
        if (node->isLeaf()) {
          size_t start = node->startIdx();
          size_t end = node->endIdx();
          for (size_t i = start; i < end; i++) {
            total += kernel_func(data[i], q);
          }
        } else {
          // Otherwise, add kernel contribution
          total += kernel_func(node->leaf_data(), q);
        }
      } else {
        // go to first child and continue
        const TREETYPE* child = node->firstChild(node_buffer);
        if (child != nullptr) {
          node = child;
          state = TraversalState::FromParent;
          continue;
        }
      }
    }

    if (node == root) {
      // Done
      return total;
    }
    // find position of node in parent
    const TREETYPE* parent = &node_buffer[node->parentIdx()];
    const TREETYPE* next_child = parent->nextChild(node, node_buffer);
    if (next_child == nullptr) {
      node = parent;
      state = TraversalState::FromChild;
    } else {
      node = next_child;
      state = TraversalState::FromSibling;
    }
  }
}

template <typename T, typename TREETYPE>
DEVICE T evaluateBarnesHutGPURecursive(const LeafData<TREETYPE::N>* data,
                                       const TREETYPE* node,
                                       const TREETYPE* node_buffer,
                                       VectorNd<TREETYPE::N> q, FLOAT beta,
                                       KernelFunc<T, TREETYPE::N> kernel_func) {
  T total(ZeroInitializer<T>::zero());
  if (node->useFarField(q, beta)) {
    // Add contribution
    if (node->isLeaf()) {
      size_t start = node->startIdx();
      size_t end = node->endIdx();
      for (size_t i = start; i < end; i++) {
        total += kernel_func(data[i], q);
      }
    } else {
      // Otherwise, add kernel contribution
      total += kernel_func(node->leaf_data(), q);
    }
  } else {
    // loop over all children and add to total
    for (auto it = node->childIter(); it; ++it) {
      total +=
          evaluateBarnesHutGPURecursive(data, node->child(*it, node_buffer),
                                        node_buffer, q, beta, kernel_func);
    }
  }
  return total;
}
