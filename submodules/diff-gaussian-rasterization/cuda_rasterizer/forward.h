/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "vec_math.h"

namespace FORWARD
{
	// lite render for baking
	void lite_render(
		const dim3 grid, dim3 block,
		int W, int H,
		const uint2* ranges,
		const uint32_t* point_list,
		const float* features,
		const float2* points_xy_image,
		const float4* conic_opacity,
		const float* depth,
		const float* bg_color,
		uint32_t* n_contrib,
		float* final_T,
		float* out_color,
		float* out_opacity,
		float* out_depth,
		bool argmax_depth);

	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		bool* clamped,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		uint32_t* tiles_touched,
		const dim3 grid,
		const bool prefiltered,
		const bool cubemap);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const int W, int H,
		const float* means3D,
		const float* cam_pos,
		const uint2* ranges,
		const uint32_t* point_list,
		const float* features,
		const float* normal,
		const float* albedo,
		const float* roughness,
		const float* metallic,
		const float2* points_xy_image,
		const float4* conic_opacity,
		const float* depth,
		const float* bg_color,
		uint32_t* n_contrib,
		float* final_T,
		float* out_color,
		float* out_opacity,
		float* out_depth,
		float* out_normal,
		float* out_albedo,
		float* out_roughness,
		float* out_metallic,
		const bool argmax_depth,
		const bool inference);
	
	void depthToNormal(
		const dim3 grid,
		const dim3 block,
		const int W, const int H,
		const float focal_x,
		const float focal_y,
		const float* viewmatrix,
		const float* depthMap,
		float* normalMap);
}


#endif