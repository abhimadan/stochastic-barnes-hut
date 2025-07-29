#pragma once

#ifdef __CUDACC__
#define BOTH __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define BOTH
#define DEVICE
#define HOST
#endif

// #define USE_DOUBLE

#ifdef USE_DOUBLE

#define FLOAT double
#define FP(x) x
#define fmax fmax
#define fmin fmin
#define exp exp
#define sqrt sqrt
#define curand_uniform curand_uniform_double

#ifdef CUDA_CC
#define PI CUDART_PI
#else
#define PI M_PI
#endif

#else

#define FLOAT float
#define FP(x) x##f
#define fmax fmaxf
#define fmin fminf
#define exp expf
#define sqrt sqrtf
#define curand_uniform curand_uniform

#ifdef CUDA_CC
#define PI CUDART_PI_F
#else
#define PI M_PI
#endif

#endif

#define _USE_MATH_DEFINES
#include <Eigen/Core>

template<int N>
using VectorNd = Eigen::Matrix<FLOAT, N, 1>;

template<int N>
using VectorNi = Eigen::Matrix<int, N, 1>;

using DynamicVector = Eigen::Matrix<FLOAT, Eigen::Dynamic, 1>;
using DynamicMatrix = Eigen::Matrix<FLOAT, Eigen::Dynamic, Eigen::Dynamic>;
