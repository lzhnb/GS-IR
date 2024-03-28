// Copyright 2023 Zhihao Liang
#include <curand.h>
#include <curand_kernel.h>

#include "irradiance_kernel.hpp"
#include "pbr_utils.cuh"

constexpr float EPS = 1e-4;

// kernels
__global__ void trilinear_interpolate_coefficients_forward_kernel(
    const uint32_t num_rays, const uint32_t sh_degree, const uint32_t res, const uint32_t channel,
    const float *__restrict__ coeffs_ptr,  // [res, res, res, d2, C]
    const float *__restrict__ aabb_ptr,    // [6]
    const float *__restrict__ points_ptr,  // [num_rays, 3]
    const float *__restrict__ normals_ptr, // [num_rays, 3]
    // output
    float *__restrict__ output_coeffs // [num_rays, d2, C]
) {
    const int32_t ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= num_rays) {
        return;
    }

    // get the aabb
    const float3 aabb_min = make_float3(aabb_ptr[0], aabb_ptr[1], aabb_ptr[2]);
    const float3 aabb_max = make_float3(aabb_ptr[3], aabb_ptr[4], aabb_ptr[5]);

    // get the point and normal
    const float3 point = make_float3(points_ptr[ray_id * 3 + 0], points_ptr[ray_id * 3 + 1],
                                     points_ptr[ray_id * 3 + 2]);
    const float3 normal = make_float3(normals_ptr[ray_id * 3 + 0], normals_ptr[ray_id * 3 + 1],
                                      normals_ptr[ray_id * 3 + 2]);

    // calculate the quantized coordinate
    const float3 grid = (aabb_max - aabb_min) / (float(res) - 1.0f);
    float3 n_xyz = (point - aabb_min) / grid; // normalized coordinate
    n_xyz = clamp(n_xyz, 0.0, float(res) - 1.0f);
    float3 f_xyz = floor(n_xyz); // floor normalized coordinate
    const int3 quat = make_int3(f_xyz);

    // get trilinear weights
    const float3 o_xyz = n_xyz - f_xyz; // offset
    const float weight000 = (1.0f - o_xyz.x) * (1.0f - o_xyz.y) * (1.0f - o_xyz.z);
    const float weight001 = (1.0f - o_xyz.x) * (1.0f - o_xyz.y) * o_xyz.z;
    const float weight010 = (1.0f - o_xyz.x) * o_xyz.y * (1.0f - o_xyz.z);
    const float weight011 = (1.0f - o_xyz.x) * o_xyz.y * o_xyz.z;
    const float weight100 = o_xyz.x * (1.0f - o_xyz.y) * (1.0f - o_xyz.z);
    const float weight101 = o_xyz.x * (1.0f - o_xyz.y) * o_xyz.z;
    const float weight110 = o_xyz.x * o_xyz.y * (1.0f - o_xyz.z);
    const float weight111 = o_xyz.x * o_xyz.y * o_xyz.z;

    // get coefficient idx
    const uint32_t d2 = sh_degree * sh_degree, r2 = res * res, r1 = res;
    const int32_t coeff000_idx = quat.x * r2 + quat.y * r1 + quat.z;
    const int32_t coeff001_idx = quat.x * r2 + quat.y * r1 + min(quat.z + 1, res - 1);
    const int32_t coeff010_idx = quat.x * r2 + min(quat.y + 1, res - 1) * r1 + quat.z;
    const int32_t coeff011_idx =
        quat.x * r2 + min(quat.y + 1, res - 1) * r1 + min(quat.z + 1, res - 1);
    const int32_t coeff100_idx = min(quat.x + 1, res - 1) * r2 + quat.y * r1 + quat.z;
    const int32_t coeff101_idx =
        min(quat.x + 1, res - 1) * r2 + quat.y * r1 + min(quat.z + 1, res - 1);
    const int32_t coeff110_idx =
        min(quat.x + 1, res - 1) * r2 + min(quat.y + 1, res - 1) * r1 + quat.z;
    const int32_t coeff111_idx =
        min(quat.x + 1, res - 1) * r2 + min(quat.y + 1, res - 1) * r1 + min(quat.z + 1, res - 1);

    // interpolation
    const float *coeffs000 = coeffs_ptr + coeff000_idx * d2;
    const float *coeffs001 = coeffs_ptr + coeff001_idx * d2;
    const float *coeffs010 = coeffs_ptr + coeff010_idx * d2;
    const float *coeffs011 = coeffs_ptr + coeff011_idx * d2;
    const float *coeffs100 = coeffs_ptr + coeff100_idx * d2;
    const float *coeffs101 = coeffs_ptr + coeff101_idx * d2;
    const float *coeffs110 = coeffs_ptr + coeff110_idx * d2;
    const float *coeffs111 = coeffs_ptr + coeff111_idx * d2;
    float *output_coeff = output_coeffs + ray_id * d2;
    for (uint32_t i = 0; i < d2; ++i) {
        for (uint32_t j = 0; j < channel; ++j) {
            output_coeff[i * channel + j] = weight000 * coeffs000[i * channel + j] + weight001 * coeffs001[i * channel + j] +
                            weight010 * coeffs010[i * channel + j] + weight011 * coeffs011[i * channel + j] +
                            weight100 * coeffs100[i * channel + j] + weight101 * coeffs101[i * channel + j] +
                            weight110 * coeffs110[i * channel + j] + weight111 * coeffs111[i * channel + j];
        }
    }
}

__global__ void trilinear_interpolate_coefficients_backward_kernel(
    const uint32_t num_rays, const uint32_t sh_degree, const uint32_t res, const uint32_t channel,
    const float *__restrict__ coeffs_grad_ptr, // [num_rays, d2, C]
    const float *__restrict__ aabb_ptr,        // [6]
    const float *__restrict__ points_ptr,      // [num_rays, 3]
    const float *__restrict__ normals_ptr,     // [num_rays, 3]
    // output
    float *__restrict__ output_coeffs_grad // [res, res, res, d2, C]
) {
    const int32_t ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= num_rays) {
        return;
    }

    // get the aabb
    const float3 aabb_min = make_float3(aabb_ptr[0], aabb_ptr[1], aabb_ptr[2]);
    const float3 aabb_max = make_float3(aabb_ptr[3], aabb_ptr[4], aabb_ptr[5]);

    // get the point and normal
    const float3 point = make_float3(points_ptr[ray_id * 3 + 0], points_ptr[ray_id * 3 + 1],
                                     points_ptr[ray_id * 3 + 2]);
    const float3 normal = make_float3(normals_ptr[ray_id * 3 + 0], normals_ptr[ray_id * 3 + 1],
                                      normals_ptr[ray_id * 3 + 2]);

    // calculate the quantized coordinate
    const float3 grid = (aabb_max - aabb_min) / (float(res) - 1.0f);
    float3 n_xyz = (point - aabb_min) / grid; // normalized coordinate
    n_xyz = clamp(n_xyz, 0.0, float(res) - 1.0f);
    float3 f_xyz = floor(n_xyz); // floor normalized coordinate
    const int3 quat = make_int3(f_xyz);

    // get trilinear weights
    const float3 o_xyz = n_xyz - f_xyz; // offset
    const float weight000 = (1.0f - o_xyz.x) * (1.0f - o_xyz.y) * (1.0f - o_xyz.z);
    const float weight001 = (1.0f - o_xyz.x) * (1.0f - o_xyz.y) * o_xyz.z;
    const float weight010 = (1.0f - o_xyz.x) * o_xyz.y * (1.0f - o_xyz.z);
    const float weight011 = (1.0f - o_xyz.x) * o_xyz.y * o_xyz.z;
    const float weight100 = o_xyz.x * (1.0f - o_xyz.y) * (1.0f - o_xyz.z);
    const float weight101 = o_xyz.x * (1.0f - o_xyz.y) * o_xyz.z;
    const float weight110 = o_xyz.x * o_xyz.y * (1.0f - o_xyz.z);
    const float weight111 = o_xyz.x * o_xyz.y * o_xyz.z;

    // get coefficient idx
    const uint32_t d2 = sh_degree * sh_degree, r2 = res * res, r1 = res;
    const int32_t coeff000_idx = quat.x * r2 + quat.y * r1 + quat.z;
    const int32_t coeff001_idx = quat.x * r2 + quat.y * r1 + min(quat.z + 1, res - 1);
    const int32_t coeff010_idx = quat.x * r2 + min(quat.y + 1, res - 1) * r1 + quat.z;
    const int32_t coeff011_idx =
        quat.x * r2 + min(quat.y + 1, res - 1) * r1 + min(quat.z + 1, res - 1);
    const int32_t coeff100_idx = min(quat.x + 1, res - 1) * r2 + quat.y * r1 + quat.z;
    const int32_t coeff101_idx =
        min(quat.x + 1, res - 1) * r2 + quat.y * r1 + min(quat.z + 1, res - 1);
    const int32_t coeff110_idx =
        min(quat.x + 1, res - 1) * r2 + min(quat.y + 1, res - 1) * r1 + quat.z;
    const int32_t coeff111_idx =
        min(quat.x + 1, res - 1) * r2 + min(quat.y + 1, res - 1) * r1 + min(quat.z + 1, res - 1);

    // interpolation
    float *coeffs_grad_000 = output_coeffs_grad + coeff000_idx * d2;
    float *coeffs_grad_001 = output_coeffs_grad + coeff001_idx * d2;
    float *coeffs_grad_010 = output_coeffs_grad + coeff010_idx * d2;
    float *coeffs_grad_011 = output_coeffs_grad + coeff011_idx * d2;
    float *coeffs_grad_100 = output_coeffs_grad + coeff100_idx * d2;
    float *coeffs_grad_101 = output_coeffs_grad + coeff101_idx * d2;
    float *coeffs_grad_110 = output_coeffs_grad + coeff110_idx * d2;
    float *coeffs_grad_111 = output_coeffs_grad + coeff111_idx * d2;
    const float *output_coeff_grad = coeffs_grad_ptr + ray_id * d2;
    for (uint32_t i = 0; i < d2; ++i) {
        for (uint32_t j = 0; j < channel; ++j) {
            atomicAdd(&coeffs_grad_000[i * channel + j], output_coeff_grad[i * channel + j] * weight000);
            atomicAdd(&coeffs_grad_001[i * channel + j], output_coeff_grad[i * channel + j] * weight001);
            atomicAdd(&coeffs_grad_010[i * channel + j], output_coeff_grad[i * channel + j] * weight010);
            atomicAdd(&coeffs_grad_011[i * channel + j], output_coeff_grad[i * channel + j] * weight011);
            atomicAdd(&coeffs_grad_100[i * channel + j], output_coeff_grad[i * channel + j] * weight100);
            atomicAdd(&coeffs_grad_101[i * channel + j], output_coeff_grad[i * channel + j] * weight101);
            atomicAdd(&coeffs_grad_110[i * channel + j], output_coeff_grad[i * channel + j] * weight110);
            atomicAdd(&coeffs_grad_111[i * channel + j], output_coeff_grad[i * channel + j] * weight111);
        }
    }
}

// wrap functions
Tensor GSIR::trilinear_interpolate_coefficients_forward(
    const Tensor coefficients, // [res, res, res, d2, C]
    const Tensor aabb,         // [6]
    const Tensor points,       // [num_rays, 3]
    const Tensor normals,      // [num_rays, 3]
    const uint32_t sh_degree) {
    CHECK_INPUT(coefficients);
    CHECK_INPUT(points);
    CHECK_INPUT(normals);

    const torch::Device device = normals.device();
    const uint32_t num_rays = normals.size(0);
    const uint32_t res = coefficients.size(0);
    const uint32_t C = coefficients.size(4);

    const uint32_t blocks = div_round_up(num_rays, THREADS);

    Tensor output_coefficients =
        torch::zeros({num_rays, sh_degree * sh_degree, C},
                     torch::TensorOptions().dtype(torch::kFloat).device(device));

    trilinear_interpolate_coefficients_forward_kernel<<<blocks, THREADS>>>(
        num_rays, sh_degree, res, C,
        coefficients.data_ptr<float>(), // [res, res, res, d2, C]
        aabb.data_ptr<float>(),         // [6]
        points.data_ptr<float>(),       // [num_rays, 3]
        normals.data_ptr<float>(),      // [num_rays, 3]
        // output
        output_coefficients.data_ptr<float>() // [num_rays, d2, C]
    );

    return output_coefficients;
}

Tensor GSIR::trilinear_interpolate_coefficients_backward(
    const Tensor coefficients_grad, // [num_rays, d2, C]
    const Tensor aabb,              // [6]
    const Tensor points,            // [num_rays, 3]
    const Tensor normals,           // [num_rays, 3]
    const uint32_t res, const uint32_t sh_degree) {
    CHECK_INPUT(coefficients_grad);
    CHECK_INPUT(points);
    CHECK_INPUT(normals);

    const torch::Device device = normals.device();
    const uint32_t num_rays = normals.size(0);
    const uint32_t C = coefficients_grad.size(2);

    const uint32_t blocks = div_round_up(num_rays, THREADS);

    Tensor output_coefficients_gard =
        torch::zeros({res, res, res, sh_degree * sh_degree, C},
                     torch::TensorOptions().dtype(torch::kFloat).device(device));

    trilinear_interpolate_coefficients_backward_kernel<<<blocks, THREADS>>>(
        num_rays, sh_degree, res, C,
        coefficients_grad.data_ptr<float>(), // [num_rays, d2, C]
        aabb.data_ptr<float>(),              // [6]
        points.data_ptr<float>(),            // [num_rays, 3]
        normals.data_ptr<float>(),           // [num_rays, 3]
        // output
        output_coefficients_gard.data_ptr<float>() // [res, res, res, d2, C]
    );

    return output_coefficients_gard;
}
