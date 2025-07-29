#pragma once

#include <iostream>

#include "utils.h"

// Doesn't handle all cases but should be enough for mac/linux and cuda
BOTH inline uint64_t FindLowestOn(uint64_t mask) {
#if defined(__CUDA_ARCH__)
  return __ffsll(mask) - 1;
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_ctzll(mask);
#else
  std::cout << "???\n";
  return 0;
#endif
}

// VDB-style bitmask for fast and compact child iterators without a vector.
// This is mostly based on nanovdb's Mask implementation: 
template<uint64_t DIM_SPLIT_LOG, int N>
struct BitMask {
  static constexpr uint64_t MaskSize = 1 << (N * DIM_SPLIT_LOG);
  static constexpr uint64_t ArraySize = std::max((uint64_t)1, MaskSize >> 6);
  uint64_t mask_[ArraySize];

  BOTH BitMask() {
    for (int i = 0; i < ArraySize; i++) {
      mask_[i] = 0;
    }
  }

  BOTH BitMask(const BitMask& other) {
    for (int i = 0; i < ArraySize; i++) {
      mask_[i] = other.mask_[i];
    }
  }

  BOTH uint64_t firstBit() const {
    uint64_t n = 0;
    while (n < ArraySize && !mask_[n]) {
      n++;
    }
    return (n >= ArraySize) ? MaskSize : (n << 6) + FindLowestOn(mask_[n]);
  }

  BOTH uint64_t nextBit(uint64_t last_pos) const {
    uint64_t next_pos = last_pos + 1;
    if (next_pos >= MaskSize) {
      return MaskSize; // out of bounds now
    }
    uint64_t n = next_pos >> 6;
    uint64_t bit = next_pos & 63;
    uint64_t b = mask_[n];
    if (b & (uint64_t(1) << bit)) {
      return next_pos;
    }
    b &= ~uint64_t(0) << bit; // mask out already-seen bits
    while (!b && (n + 1 < ArraySize)) {
      n++;
      b = mask_[n];
    }
    return !b ? MaskSize : (n << 6) + FindLowestOn(b);
  }

  BOTH void enableBit(uint64_t idx) {
    uint64_t n = idx >> 6;
    uint64_t bit = idx & 63;
    mask_[n] |= uint64_t(1) << bit;
  }

  BOTH bool containsBit(uint64_t idx) const {
    uint64_t n = idx >> 6;
    uint64_t bit = idx & 63;
    return mask_[n] & (uint64_t(1) << bit);
  }

  struct Iter {
    BOTH Iter(const BitMask* parent)
        : parent_(parent), pos_(parent->firstBit()) {
        }

    BOTH operator bool() const { return pos_ < BitMask::MaskSize; }
    BOTH Iter& operator++() {
      pos_ = parent_->nextBit(pos_);
      return *this;
    }
    BOTH uint64_t operator*() const { return pos_; }

    const BitMask* parent_;
    uint64_t pos_;
  };

  BOTH Iter Begin() const { return Iter(this); }
};

template<uint64_t DIM_SPLIT_LOG>
using BitMask2D = BitMask<DIM_SPLIT_LOG, 2>;

template<uint64_t DIM_SPLIT_LOG>
using BitMask3D = BitMask<DIM_SPLIT_LOG, 3>;
