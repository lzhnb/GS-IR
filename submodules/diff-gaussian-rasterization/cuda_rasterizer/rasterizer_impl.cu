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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	const int P,
	const int* radii,
	const dim3 grid,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], grid, rect_min, rect_max);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		// Refer to the Appendix C (higher 32 bit for tile ID and lower 32 bit for depth)
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;  // NOTE: offset[idx] - offset[idx - 1] = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y)
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(const int L, const uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	const uint32_t currtile = point_list_keys[idx] >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		const uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum<<<(P + 255) / 256, 256>>> (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);  		// float*
	obtain(chunk, geom.clamped, P * 3, 128);	// bool*
	obtain(chunk, geom.internal_radii, P, 128);	// int*
	obtain(chunk, geom.means2D, P, 128);		// float2*
	obtain(chunk, geom.cov3D, P * 6, 128);		// float*
	obtain(chunk, geom.conic_opacity, P, 128);	// float4*
	obtain(chunk, geom.rgb, P * 3, 128);		// float*
	obtain(chunk, geom.tiles_touched, P, 128);	// uint32_t*
	obtain(chunk, geom.point_offsets, P, 128);	// uint32_t*
	// NOTE: do not perform InclusiveSum due to `nullptr`, just allocate size and store in `scan_size`
	// https://nvlabs.github.io/cub/structcub_1_1_device_scan.html#a9416ac1ea26f9fde669d83ddc883795a
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);	// char*
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);	// float*
	obtain(chunk, img.n_contrib, N, 128);	// uint32_t*
	obtain(chunk, img.ranges, N, 128);		// uint2*
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);					// uint32_t*
	obtain(chunk, binning.point_list_unsorted, P, 128);			// uint32_t*
	obtain(chunk, binning.point_list_keys, P, 128);				// uint64_t*
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);	// uint64_t*
	// NOTE: do not perform SortPairs due to `nullptr`, just allocate size and store in `sorting_size`
	// https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#a65e82152de448c6373ed9563aaf8af7e
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);	// char*
	return binning;
}

void CudaRasterizer::Rasterizer::depthToNormal(
	const int width, const int height,
	const float focal_x,
	const float focal_y,
	const float* viewmatrix,
	const float* depthMap,
	float* normalMap
) {
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	FORWARD::depthToNormal(
		tile_grid, block,
		width, height, focal_x, focal_y,
		viewmatrix,
		depthMap,
		normalMap
	);
}

// Lite forward for baking
int CudaRasterizer::Rasterizer::lite_forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	const bool argmax_depth,
	float* out_color,
	float* out_opacity,
	float* out_depth,
	int* radii)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.clamped,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		geomState.tiles_touched,
		tile_grid,
		prefiltered,
		false);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys<<<(P + 255) / 256, 256 >>>(
		P,
		radii,
		tile_grid,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit);

	cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges<<<(num_rendered + 255) / 256, 256 >>>(
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	FORWARD::lite_render(
		tile_grid, block,
		width, height,
		imgState.ranges,
		binningState.point_list,
		feature_ptr,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.depths,
		background,
		imgState.n_contrib,
		imgState.accum_alpha,
		out_color,
		out_opacity,
		out_depth,
		argmax_depth);

	return num_rendered;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,  		// [3, H, W]
	const int width, int height,
	const float* means3D,			// [P, 3]
	const float* shs,				// [P, d2, 3]
	const float* colors_precomp,	// [P, 3]
	const float* opacities,			// [P, 1]
	const float* normal,			// [P, 3]
	const float* albedo,			// [P, 3]
	const float* roughness,			// [P, 1]
	const float* metallic,			// [P, 1]
	const float* scales,			// [P, 3]
	const float scale_modifier,
	const float* rotations,			// [P, 4]
	const float* cov3D_precomp,		// [P, 6]
	const float* viewmatrix,		// [4, 4]
	const float* projmatrix,		// [4. 4]
	const float* cam_pos,			// [3]
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	const bool argmax_depth,
	const bool inference,
	float* out_color,		// [3, H, W]
	float* out_opacity,		// [1, H, W]
	float* out_depth,		// [1, H, W]
	float* out_normal,		// [3, H, W]
	float* out_albedo,		// [3, H, W]
	float* out_roughness,	// [1, H, W]
	float* out_metallic,	// [1, H, W]
	int* radii,				// [P]
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);						// Calculate the chunk size needed for geometry buffer
	char* chunkptr = geometryBuffer(chunk_size);						// Allocate the chunk according to the obtained chunk size
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);  	// Initialize the allocated chunk

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// Define the thread grid (16 x 16) for CUDA kernel
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	// The allocation is similar to GeometryState
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.clamped,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		geomState.tiles_touched,
		tile_grid,
		prefiltered,
		false
	), debug);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	//       (tiles_touched) -> (point_offsets)
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// Allocate binning for pairsort
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys<<<(P + 255) / 256, 256 >>>(
		P,
		radii,
		tile_grid,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted)
	CHECK_CUDA(, debug);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug);
	// NOTE: An optional bit subrange [begin_bit, end_bit) ([`0`, `32 + bi`)) of differentiating key bits can be specified.
	// This can reduce overall sorting overhead and yield a corresponding performance improvement.

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0) {
		identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	}
	CHECK_CUDA(, debug);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* normal_ptr = normal;
	const float* albedo_ptr = albedo;
	const float* roughness_ptr = roughness;
	const float* metallic_ptr = metallic;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		width, height,
		means3D,
		cam_pos,
		imgState.ranges,
		binningState.point_list,
		feature_ptr,
		normal_ptr,
		albedo_ptr,
		roughness_ptr,
		metallic_ptr,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.depths,
		background,
		imgState.n_contrib,
		imgState.accum_alpha,
		out_color,
		out_opacity,
		out_depth,
		out_normal,
		out_albedo,
		out_roughness,
		out_metallic,
		argmax_depth,
		inference), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* normal,
	const float* albedo,
	const float* roughness,
	const float* metallic,
	const float* scales,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const int* radii,
	const float scale_modifier,
	const float tan_fovx, float tan_fovy,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_dpix_opacity,
	const float* dL_dpix_normal,
	const float* dL_dpix_albedo,
	const float* dL_dpix_roughness,
	const float* dL_dpix_metallic,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dnormal,
	float* dL_dalbedo,
	float* dL_droughness,
	float* dL_dmetallic,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		width, height,
		means3D,
		cam_pos,
		imgState.ranges,
		binningState.point_list,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		normal,
		albedo,
		roughness,
		metallic,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_dpix_opacity,
		dL_dpix_normal,
		dL_dpix_albedo,
		dL_dpix_roughness,
		dL_dpix_metallic,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dnormal,
		dL_dalbedo,
		dL_droughness,
		dL_dmetallic), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		(glm::vec3*)cam_pos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}
