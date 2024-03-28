// Copyright 2023 Zhihao Liang
#pragma once
#include <torch/extension.h>
#include <tuple>

#include "utils.h"
#include "vec_math.h"

using torch::Tensor;

namespace GSIR {
std::tuple<Tensor, Tensor> sparse_interpolate_coefficients(const Tensor occlusion_coefficients,
                                                           const Tensor occlusion_ids,
                                                           const Tensor aabb, const Tensor points,
                                                           const Tensor normals,
                                                           const uint32_t sh_degree);
Tensor SH_reconstruction(const Tensor coefficients, const Tensor lobes, const Tensor roughness,
                         const uint32_t num_samples, const uint32_t sh_degree);
void dialate_occlusion_ids(Tensor occlusion_ids);
} // namespace GSIR
