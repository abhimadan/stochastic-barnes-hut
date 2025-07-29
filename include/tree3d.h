#pragma once

#define _USE_MATH_DEFINES
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <memory>
#include <vector>

#include "aabb.h"
#include "bitmask.h"
#include "leaf_data.h"
#include "utils.h"

template <uint64_t LEAF_SIZE, uint64_t DIM_SPLIT_LOG>
class Tree3D;

template <uint64_t LEAF_SIZE, uint64_t DIM_SPLIT_LOG>
class Tree3DNode {
 public:
  static constexpr int N = 3;

  // NOTE: default destructor (not virtual) for now - it has no
  // superclass/subclass so this is fine but keep in mind
  BOTH Tree3DNode() = default;

  BOTH bool isLeaf() const {
    return is_leaf_;
  }

  BOTH bool isLeafPredicate() const {
    return depth_ >= max_depth_ || end_ - start_ <= LEAF_SIZE;
  }

  BOTH bool isRoot() const {
    return buffer_idx_ == 0;
  }

  BOTH typename BitMask3D<DIM_SPLIT_LOG>::Iter childIter() const {
    return bitmask_.Begin();
  }

  BOTH const Tree3DNode<LEAF_SIZE, DIM_SPLIT_LOG>* child(
      uint64_t idx, const Tree3DNode* buffer) const {
    return &buffer[child_buffer_idxs_[idx]];
  }

  BOTH bool containsChild(uint64_t idx) const {
    return bitmask_.containsBit(idx);
  }

  BOTH uint64_t num_children() const {
    return num_children_;
  }

  // TODO: can coordsToIndex and indexToCoords be non-member functions? keep
  // them as members for now in case we want to automatically skip empty
  // entries, but fine for the time being
  BOTH uint64_t coordsToIndex(uint64_t x, uint64_t y, uint64_t z) const {
    uint64_t idx = (((z << DimSplitLog) + y) << DimSplitLog) + x;
    return idx;
  }

  uint64_t coordsToIndex(std::tuple<uint64_t, uint64_t, uint64_t> xyz) const {
    return coordsToIndex(std::get<0>(xyz), std::get<1>(xyz), std::get<2>(xyz));
  }

  BOTH uint64_t coordsToIndex(Eigen::Vector3i xyz) const {
    return coordsToIndex(xyz.x(), xyz.y(), xyz.z());
  }

  // Return a tuple because that way the type is the same regardless of
  // dimension.
  std::tuple<uint64_t, uint64_t, uint64_t> indexToCoords(uint64_t idx) const {
    constexpr uint64_t coord_mask = DimSplit - 1;
    uint64_t x = idx & coord_mask;
    uint64_t y = (idx >> DimSplitLog) & coord_mask;
    uint64_t z = (idx >> 2 * DimSplitLog) & coord_mask;
    return std::make_tuple(x, y, z);
  }

  BOTH const LeafData<3>& leaf_data() const {
    return contrib_data_;
  }

  BOTH VectorNd<3> center() const {
    return contrib_data_.position;
  }

  BOTH VectorNd<3> weighted_normal() const {
    return contrib_data_.normal;
  }

  BOTH uint64_t startIdx() const {
    return start_;
  }

  BOTH uint64_t endIdx() const {
    return end_;
  }

  BOTH uint64_t parentIdx() const {
    return parent_idx_;
  }

  BOTH uint64_t firstChildIdx() const {
    // Assumes that isLeaf() is false
    return child_buffer_idxs_[bitmask_.firstBit()];
  }

  BOTH uint64_t numContainedPoints() const {
    return end_ - start_;
  }

  BOTH AABB<3> bounds() const {
    return bounds_;
  }

  BOTH int depth() const {
    return depth_;
  }

  BOTH FLOAT total_mass() const {
    return contrib_data_.mass;
  }

  BOTH FLOAT alpha() const {
    return contrib_data_.alpha;
  }

  BOTH FLOAT farFieldRatio(VectorNd<3> q) const {
    if (isLeaf()) {
      return INFINITY;
    }
    FLOAT dist = (q - center()).norm();
    return dist / diameter_;
  }

  BOTH FLOAT farFieldRatioSquared(VectorNd<3> q) const {
    if (isLeaf()) {
      return INFINITY;
    }
    FLOAT dist = (q - center()).squaredNorm();
    return dist / (diameter_ * diameter_);
  }


  BOTH bool useFarField(VectorNd<3> q, FLOAT beta) const {
    return farFieldRatio(q) >= beta;
  }

  BOTH const Tree3DNode* containedInChild(int data_idx,
                                          const Tree3DNode* node_buffer) const {
    if (data_idx < start_ || data_idx >= end_) {
      return nullptr;
    }
    for (auto it = bitmask_.Begin(); it; ++it) {
      const auto* child = this->child(*it, node_buffer);
      if (child->start_ <= data_idx && data_idx < child->end_) {
        if (child->num_children_ == 1) {
          return child->containedInChild(data_idx, node_buffer);
        }
        return child;
      }
    }
    return nullptr;  // shouldn't get here, this case should be caught earlier
  }

  BOTH const Tree3DNode* containedInChild(VectorNd<3> p,
                                          const Tree3DNode* node_buffer) const {
    if (!bounds_.contains(p)) {
      return nullptr;
    }
    Eigen::Vector3i quantized = bounds_.quantize(p, DimSplit);
    const Tree3DNode* child = this->child(coordsToIndex(quantized), node_buffer);
    if (child->num_children_ == 1) {
      return child->containedInChild(p, node_buffer);
    }
    return child;
  }

  BOTH const Tree3DNode* firstChild(const Tree3DNode* node_buffer) const {
    uint64_t first_coord = bitmask_.firstBit();
    if (first_coord >= BranchFactor) {
      return nullptr;
    }
    return child(first_coord, node_buffer);
  }

  BOTH const Tree3DNode* nextChild(const Tree3DNode* child,
                                   const Tree3DNode* node_buffer) const {
    // quantize child's midpoint, find next available bit, return that node
    VectorNd<3> midpoint = child->bounds_.center();
    if (!bounds_.contains(midpoint)) {
      return nullptr;
    }
    Eigen::Vector3i quantized_midpoint = bounds_.quantize(midpoint, DimSplit);
    uint64_t next_coord = bitmask_.nextBit(coordsToIndex(quantized_midpoint));
    if (next_coord >= BranchFactor) {
      return nullptr;
    }
    return this->child(next_coord, node_buffer);
  }

  static constexpr uint64_t DimSplitLog = DIM_SPLIT_LOG;
  static constexpr uint64_t DimSplit = 1 << DimSplitLog;
  static constexpr uint64_t BranchFactor = DimSplit * DimSplit * DimSplit;

 private:
  uint64_t start_;
  uint64_t end_;
  uint64_t buffer_idx_;
  uint64_t parent_idx_;
  AABB<3> bounds_;
  FLOAT diameter_;
  LeafData<3> contrib_data_;
  BitMask3D<DIM_SPLIT_LOG> bitmask_;
  uint64_t child_buffer_idxs_[BranchFactor];
  int num_children_;
  int depth_;
  int max_depth_;
  bool is_leaf_;

  friend class Tree3D<LEAF_SIZE, DIM_SPLIT_LOG>;

  struct BuildInfo {
    uint64_t start;
    uint64_t end;
    AABB<3> bounds;
    int depth;
    uint64_t buffer_idx;
    uint64_t parent_idx;
  };

  void build(std::vector<LeafData<3>>& leaf_data,
             Tree3D<LEAF_SIZE, DIM_SPLIT_LOG>& tree,
             std::deque<BuildInfo>& build_queue, const BuildInfo& build_info,
             int max_depth) {
    assert(build_info.start < build_info.end);

    start_ = build_info.start;
    end_ = build_info.end;
    buffer_idx_ = build_info.buffer_idx;
    parent_idx_ = build_info.parent_idx;
    bounds_ = build_info.bounds;
    diameter_ = build_info.bounds.diagonal().norm();
    num_children_ = 0;
    depth_ = build_info.depth;
    max_depth_ = max_depth;
    is_leaf_ = isLeafPredicate();

    if (isLeaf()) {
      for (uint64_t i = build_info.start; i < build_info.end; i++) {
        contrib_data_.alpha = leaf_data[i].alpha;
        contrib_data_.mass += leaf_data[i].mass;
        contrib_data_.position += leaf_data[i].mass * leaf_data[i].position;
        contrib_data_.normal += leaf_data[i].mass * leaf_data[i].normal;
      }
      contrib_data_.position /= contrib_data_.mass;
      contrib_data_.normal /= contrib_data_.mass;
      return;
    }

    // Internal node - partition and recurse.
    uint64_t start = build_info.start;
    VectorNd<3> inc = build_info.bounds.diagonal() / DimSplit;
    for (uint64_t z = 0; z < DimSplit; z++) {
      FLOAT pz = build_info.bounds.min_corner[2] + z * inc[2];
      FLOAT pz_next = pz + inc[2];
      for (uint64_t y = 0; y < DimSplit; y++) {
        FLOAT py = build_info.bounds.min_corner[1] + y * inc[1];
        FLOAT py_next = py + inc[1];
        for (uint64_t x = 0; x < DimSplit; x++) {
          FLOAT px = build_info.bounds.min_corner[0] + x * inc[0];
          FLOAT px_next = px + inc[0];
          AABB<3> sub_bounds(VectorNd<3>(px, py, pz),
                             VectorNd<3>(px_next, py_next, pz_next));
          const LeafData<3>* split_ptr = std::partition(
              &leaf_data[start], &leaf_data[build_info.end],
              [&build_info, x, y, z](const LeafData<3>& d) {
                Eigen::Vector3i quantized =
                    build_info.bounds.quantize(d.position, DimSplit);
                return quantized.x() == x && quantized.y() == y &&
                       quantized.z() == z;
              });
          uint64_t split_idx = (split_ptr - &leaf_data[start]) + start;

          if (split_idx > start) {
            uint64_t child_buffer_idx = tree.reserveNewNode();

            build_queue.push_back({start, split_idx, sub_bounds,
                                   build_info.depth + 1, child_buffer_idx,
                                   buffer_idx_});

            uint64_t child_idx = coordsToIndex(x, y, z);
            bitmask_.enableBit(child_idx);
            child_buffer_idxs_[child_idx] = child_buffer_idx;
            num_children_++;
          }

          start = split_idx;
        }
      }
    }
  }

  void aggregateChildInfo(const std::vector<Tree3DNode>& nodes) {
    if (isLeaf()) {
      return;
    }

    for (auto it = bitmask_.Begin(); it; ++it) {
      const Tree3DNode& child = nodes[child_buffer_idxs_[*it]];

      contrib_data_.alpha = child.contrib_data_.alpha;
      contrib_data_.mass += child.contrib_data_.mass;
      contrib_data_.position +=
          child.contrib_data_.mass * child.contrib_data_.position;
      contrib_data_.normal +=
          child.contrib_data_.mass * child.contrib_data_.normal;
    }
    contrib_data_.position /= contrib_data_.mass;
    contrib_data_.normal /= contrib_data_.mass;
  }
};

template<uint64_t LEAF_SIZE, uint64_t DIM_SPLIT_LOG>
class Tree3D {
  // Contains a buffer, iteratively builds the nodes (partially) breadth-first,
  // and then goes bottom-up to propagate aggregated leaf data. As long as the
  // nodes are topologically sorted, we can always do this.
 public:
  using Node = Tree3DNode<LEAF_SIZE, DIM_SPLIT_LOG>;

  Tree3D(std::vector<LeafData<3>>& leaf_data, AABB<3> bounds,
       int max_depth = std::numeric_limits<int>::max())
      : global_max_depth_(0), num_reserved_nodes_(0) {
    std::deque<typename Node::BuildInfo> build_queue;
    uint64_t root_buffer_idx = reserveNewNode();
    allocateNewNodes();
    build_queue.push_back({0, leaf_data.size(), bounds, 0, root_buffer_idx, 0});
    while (!build_queue.empty()) {
      const typename Node::BuildInfo cur_node_info = build_queue.front();
      build_queue.pop_front();

      Node& cur_node = node_buffer_[cur_node_info.buffer_idx];
      cur_node.build(leaf_data, *this, build_queue, cur_node_info, max_depth);

      global_max_depth_ = std::max(global_max_depth_, cur_node.depth_);

      // Allocate children *after* we finish writing to the current node, so the
      // changes are copied to the new node buffer location if a reallocation
      // occurs.
      allocateNewNodes();
    }

    // NOTE: the iteration index has to be signed so we don't get underflow
    contrib_data_buffer_.resize(node_buffer_.size());
    for (int i = node_buffer_.size() - 1; i >= 0; i--) {
      node_buffer_[i].aggregateChildInfo(node_buffer_);
      contrib_data_buffer_[i] = node_buffer_[i].contrib_data_;
    }

    root_ = node_buffer_.data();

    // Useful for debugging in some cases
    /* std::cout << "Global max depth is: " << global_max_depth_ << std::endl; */
  }

  const Node* root() const { return root_; }

  const Node* node_buffer() const {
    return node_buffer_.data();
  }

  const std::vector<Node>& node_vector() const { return node_buffer_; }

  const LeafData<3>* contrib_data_buffer() const {
    return contrib_data_buffer_.data();
  }

  const std::vector<LeafData<3>>& contrib_data_vector() const {
    return contrib_data_buffer_;
  }

  int global_max_depth() const { return global_max_depth_; }

 private:
  friend class Tree3DNode<LEAF_SIZE, DIM_SPLIT_LOG>;

  std::vector<Node> node_buffer_;
  std::vector<LeafData<3>> contrib_data_buffer_;
  const Node* root_;
  int global_max_depth_;
  int num_reserved_nodes_;

  uint64_t reserveNewNode() {
    uint64_t idx = node_buffer_.size() + num_reserved_nodes_;
    num_reserved_nodes_++;
    return idx;
  }

  void allocateNewNodes() {
    for (int i = 0; i < num_reserved_nodes_; i++) {
      node_buffer_.push_back(Node());
    }
    num_reserved_nodes_ = 0;
  }
};

using OctreeNode = Tree3DNode<1, 3>;
using Octree = Tree3D<1, 3>;
