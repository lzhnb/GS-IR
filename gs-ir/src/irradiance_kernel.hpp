// Copyright 2023 Zhihao Liang
#pragma once
#include <torch/extension.h>
#include <tuple>

#include "utils.h"
#include "vec_math.h"

using torch::Tensor;

namespace GSIR {
Tensor trilinear_interpolate_coefficients_forward(const Tensor coefficients, // [res, res, res, d2, C]
                                                  const Tensor aabb,         // [6]
                                                  const Tensor points,       // [num_rays, 3]
                                                  const Tensor normals,      // [num_rays, 3]
                                                  const uint32_t sh_degree);
Tensor trilinear_interpolate_coefficients_backward(const Tensor coefficients_grad, // [num_rays, d2, C]
                                                   const Tensor aabb,              // [6]
                                                   const Tensor points,            // [num_rays, 3]
                                                   const Tensor normals,           // [num_rays, 3]
                                                   const uint32_t res, const uint32_t sh_degree);
} // namespace GSIR
