#pragma once

#include <random>

size_t uniformSample(size_t num_elements, std::mt19937& rng) {
  if (num_elements == 0) {
    // With unsigned ints, the distribution will go up to the max 64-bit integer
    // if we pass in 0
    return 0;
  }
  return rng() % num_elements;
}

