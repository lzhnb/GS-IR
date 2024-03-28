#pragma once
#include <math.h>
#include <vector_functions.h>
#include <vector_types.h>

constexpr uint32_t THREADS = 1024;

#ifdef __NVCC__
#define HOST_DEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOST_DEVICE
#define DEVICE
#endif

template <typename T>
HOST_DEVICE T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

// Changed from x.type().is_cuda() due to deprecation
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x)                                                     \
  CHECK_CPU(x);                                                                \
  CHECK_CONTIGUOUS(x)

#if defined(__CUDACC__)
// #define _EXP(x) expf(x) // SLOW EXP
#define _EXP(x) __expf(x) // FAST EXP
#define _SIGMOID(x) (1 / (1 + _EXP(-(x))))

#else

#define _EXP(x) expf(x)
#define _SIGMOID(x) (1 / (1 + expf(-(x))))
#endif
#define _SQR(x) ((x) * (x))
