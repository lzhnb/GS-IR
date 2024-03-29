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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LiteRasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& sh,
	const torch::Tensor& campos,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float scale_modifier,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const int degree,
	const bool prefiltered,
	const bool argmax_depth
) {
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}
	
	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_opacity = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
	
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
	
	int rendered = 0;
	if(P != 0) {
		int M = 0;
		if(sh.size(0) != 0) {
			M = sh.size(1);
		}

		rendered = CudaRasterizer::Rasterizer::lite_forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data<float>(),
			opacity.contiguous().data<float>(), 
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(), 
			viewmatrix.contiguous().data<float>(), 
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			argmax_depth,
			out_color.contiguous().data<float>(),
			out_opacity.contiguous().data<float>(),
			out_depth.contiguous().data<float>(),
			radii.contiguous().data<int>());
  	}
  	return std::make_tuple(
		rendered,
		out_color,
		out_opacity,
		radii,
		out_depth
	);
}


std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,  	// [3, H, W]
	const torch::Tensor& means3D,  		// [P, 3]
    const torch::Tensor& colors,  		// [P, 3]
    const torch::Tensor& opacity,  		// [P, 1]
    const torch::Tensor& normal,  		// [P, 3]
    const torch::Tensor& albedo,  		// [P, 3]
    const torch::Tensor& roughness,  	// [P, 1]
    const torch::Tensor& metallic,  	// [P, 1]
	const torch::Tensor& scales,  		// [P, 3]
	const torch::Tensor& rotations,  	// [P, 4]
	const torch::Tensor& cov3D_precomp,	// [P, 6]
	const torch::Tensor& sh,  			// [P, d2, 3]
	const torch::Tensor& campos,  		// [3]
	const torch::Tensor& viewmatrix,  	// [4, 4]
	const torch::Tensor& projmatrix,  	// [4, 4]
	const float scale_modifier,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const int degree,
	const bool prefiltered,
	const bool argmax_depth,
	const bool inference,
	const bool debug
) {
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}
	
	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor out_opacity = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor out_normal = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_albedo = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_roughness = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor out_metallic = torch::full({1, H, W}, 0.0, float_opts);
	
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
	
	int rendered = 0;
	if(P != 0) {
		int M = 0;
		if(sh.size(0) != 0) {
			M = sh.size(1);
		}

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data<float>(),
			opacity.contiguous().data<float>(), 
			normal.contiguous().data<float>(),
			albedo.contiguous().data<float>(),
			roughness.contiguous().data<float>(),
			metallic.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(), 
			viewmatrix.contiguous().data<float>(), 
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			argmax_depth,
			inference,
			out_color.contiguous().data<float>(),
			out_opacity.contiguous().data<float>(),
			out_depth.contiguous().data<float>(),
			out_normal.contiguous().data<float>(),
			out_albedo.contiguous().data<float>(),
			out_roughness.contiguous().data<float>(),
			out_metallic.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			debug);
  	}
  	return std::make_tuple(
		rendered,
		out_color,
		radii,
		geomBuffer,
		binningBuffer,
		imgBuffer,
		out_opacity,
		out_depth,
		out_normal,
		out_albedo,
		out_roughness,
		out_metallic
	);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& normal,
	const torch::Tensor& albedo,
	const torch::Tensor& roughness,
	const torch::Tensor& metallic,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& sh,
	const torch::Tensor& campos,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float scale_modifier,
	const float tan_fovx,
	const float tan_fovy,
	const int degree,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_opacity,
    const torch::Tensor& dL_dout_normal,
    const torch::Tensor& dL_dout_albedo,
    const torch::Tensor& dL_dout_roughness,
    const torch::Tensor& dL_dout_metallic,
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int R,
	const bool debug
) {
	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);
	
	int M = 0;
	if(sh.size(0) != 0)
	{	
		M = sh.size(1);
	}

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
	torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dalbedo = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_droughness = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dmetallic = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
	
	if(P != 0) {  
		CudaRasterizer::Rasterizer::backward(P, degree, M, R,
			background.contiguous().data<float>(),
			W, H, 
			means3D.contiguous().data<float>(),
			sh.contiguous().data<float>(),
			colors.contiguous().data<float>(),
			normal.contiguous().data<float>(),
			albedo.contiguous().data<float>(),
			roughness.contiguous().data<float>(),
			metallic.contiguous().data<float>(),
			scales.data_ptr<float>(),
			rotations.data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			scale_modifier,
			tan_fovx,
			tan_fovy,
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			dL_dout_color.contiguous().data<float>(),
    		dL_dout_opacity.contiguous().data<float>(),
			dL_dout_normal.contiguous().data<float>(),
			dL_dout_albedo.contiguous().data<float>(),
			dL_dout_roughness.contiguous().data<float>(),
			dL_dout_metallic.contiguous().data<float>(),
			dL_dmeans2D.contiguous().data<float>(),
			dL_dconic.contiguous().data<float>(),  
			dL_dopacity.contiguous().data<float>(),
			dL_dnormal.contiguous().data<float>(),
			dL_dalbedo.contiguous().data<float>(),
			dL_droughness.contiguous().data<float>(),
			dL_dmetallic.contiguous().data<float>(),
			dL_dcolors.contiguous().data<float>(),
			dL_dmeans3D.contiguous().data<float>(),
			dL_dcov3D.contiguous().data<float>(),
			dL_dsh.contiguous().data<float>(),
			dL_dscales.contiguous().data<float>(),
			dL_drotations.contiguous().data<float>(),
			debug);
	}

	return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dnormal, dL_dalbedo,
		dL_droughness, dL_dmetallic, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix
) { 
	const int P = means3D.size(0);
	
	torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
	if(P != 0)
	{
		CudaRasterizer::Rasterizer::markVisible(P,
			means3D.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			present.contiguous().data<bool>());
	}
	
	return present;
}

torch::Tensor depthToNormal(
	const int width, const int height,
	const float focal_x,
	const float focal_y,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& depthMap
) {	
	torch::Tensor normalMap = torch::full({3, height, width}, 0.0, depthMap.options());

	CudaRasterizer::Rasterizer::depthToNormal(
		width, height, focal_x, focal_y,
		viewmatrix.contiguous().data<float>(),
		depthMap.contiguous().data<float>(),
		normalMap.contiguous().data<float>()
	);
  
	return normalMap;
}