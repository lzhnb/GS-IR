// Copyright 2023 Zhihao Liang
#include <curand.h>
#include <curand_kernel.h>

#include "pbr_utils.cuh"
#include "occlusion_kernel.hpp"

constexpr float EPS = 1e-4;

__device__ void fillup(int32_t *__restrict__ input_ids, const int32_t occlu_res, const int32_t x_idx,
                       const int32_t y_idx, const int32_t z_idx, const int32_t x_shift,
                       const int32_t y_shift, const int32_t z_shift) {
    const int32_t ids = input_ids[clamp(x_idx + x_shift, 0, occlu_res - 1) * occlu_res * occlu_res +
                                  clamp(y_idx + y_shift, 0, occlu_res - 1) * occlu_res +
                                  clamp(z_idx + z_shift, 0, occlu_res - 1)];
    if (ids >= 0) {
        input_ids[x_idx * occlu_res * occlu_res + y_idx * occlu_res + z_idx] = ids;
    }
}

// kernels
__global__ void sparse_interpolate_coefficients_kernel(
    const uint32_t num_rays, const uint32_t sh_degree, const uint32_t occlu_res,
    const float *__restrict__ coeffs_ptr,  // [num_grid, d2]
    const int32_t *__restrict__ input_ids, // [occlu_res, occlu_res, occlu_res]
    const float *__restrict__ aabb_ptr,    // [6]
    const float *__restrict__ points_ptr,  // [num_rays, 3]
    const float *__restrict__ normals_ptr, // [num_rays, 3]
    // output
    float *__restrict__ output_coeffs, // [num_rays, d2, 1]
    int32_t *__restrict__ output_ids // [num_rays, 8]
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
    const float3 grid = (aabb_max - aabb_min) / (float(occlu_res) - 1.0f);
    float3 n_xyz = (point - aabb_min) / grid; // normalized coordinate
    n_xyz = clamp(n_xyz, 0.0, float(occlu_res) - 1.0f);
    float3 f_xyz = floor(n_xyz);                    // floor normalized coordinate
    const int3 quat = make_int3(f_xyz);

    // get eight directions
    const float3 dir000 = make_float3(f_xyz.x, f_xyz.y, f_xyz.z) * grid + aabb_min - point;
    const float3 dir001 = make_float3(f_xyz.x, f_xyz.y, f_xyz.z + 1) * grid + aabb_min - point;
    const float3 dir010 = make_float3(f_xyz.x, f_xyz.y + 1, f_xyz.z) * grid + aabb_min - point;
    const float3 dir011 = make_float3(f_xyz.x, f_xyz.y + 1, f_xyz.z + 1) * grid + aabb_min - point;
    const float3 dir100 = make_float3(f_xyz.x + 1, f_xyz.y, f_xyz.z) * grid + aabb_min - point;
    const float3 dir101 = make_float3(f_xyz.x + 1, f_xyz.y, f_xyz.z + 1) * grid + aabb_min - point;
    const float3 dir110 = make_float3(f_xyz.x + 1, f_xyz.y + 1, f_xyz.z) * grid + aabb_min - point;
    const float3 dir111 =
        make_float3(f_xyz.x + 1, f_xyz.y + 1, f_xyz.z + 1) * grid + aabb_min - point;

    // cosine mask for normal aware
    const float mask000 = dot(dir000, normal) > 0.0f ? 1.0f : 0.0f;
    const float mask001 = dot(dir001, normal) > 0.0f ? 1.0f : 0.0f;
    const float mask010 = dot(dir010, normal) > 0.0f ? 1.0f : 0.0f;
    const float mask011 = dot(dir011, normal) > 0.0f ? 1.0f : 0.0f;
    const float mask100 = dot(dir100, normal) > 0.0f ? 1.0f : 0.0f;
    const float mask101 = dot(dir101, normal) > 0.0f ? 1.0f : 0.0f;
    const float mask110 = dot(dir110, normal) > 0.0f ? 1.0f : 0.0f;
    const float mask111 = dot(dir111, normal) > 0.0f ? 1.0f : 0.0f;

    // get trilinear weights
    const float3 o_xyz = n_xyz - f_xyz; // offset
    float weight000 = (1.0f - o_xyz.x) * (1.0f - o_xyz.y) * (1.0f - o_xyz.z) * mask000;
    float weight001 = (1.0f - o_xyz.x) * (1.0f - o_xyz.y) * o_xyz.z * mask001;
    float weight010 = (1.0f - o_xyz.x) * o_xyz.y * (1.0f - o_xyz.z) * mask010;
    float weight011 = (1.0f - o_xyz.x) * o_xyz.y * o_xyz.z * mask011;
    float weight100 = o_xyz.x * (1.0f - o_xyz.y) * (1.0f - o_xyz.z) * mask100;
    float weight101 = o_xyz.x * (1.0f - o_xyz.y) * o_xyz.z * mask101;
    float weight110 = o_xyz.x * o_xyz.y * (1.0f - o_xyz.z) * mask110;
    float weight111 = o_xyz.x * o_xyz.y * o_xyz.z * mask111;

    // normalize weights
    float weight_sum = weight000 + weight001 + weight010 + weight011 + weight100 + weight101 +
                       weight110 + weight111;
    weight_sum = (weight_sum == 0.0f) ? EPS : weight_sum;
    weight000 = weight000 / weight_sum;
    weight001 = weight001 / weight_sum;
    weight010 = weight010 / weight_sum;
    weight011 = weight011 / weight_sum;
    weight100 = weight100 / weight_sum;
    weight101 = weight101 / weight_sum;
    weight110 = weight110 / weight_sum;
    weight111 = weight111 / weight_sum;

    // get coefficient idx
    const uint32_t d2 = sh_degree * sh_degree, r2 = occlu_res * occlu_res, r1 = occlu_res;
    const int32_t coeff000_idx = input_ids[quat.x * r2 + quat.y * r1 + quat.z];
    const int32_t coeff001_idx =
        input_ids[quat.x * r2 + quat.y * r1 + min(quat.z + 1, occlu_res - 1)];
    const int32_t coeff010_idx =
        input_ids[quat.x * r2 + min(quat.y + 1, occlu_res - 1) * r1 + quat.z];
    const int32_t coeff011_idx =
        input_ids[quat.x * r2 + min(quat.y + 1, occlu_res - 1) * r1 + min(quat.z + 1, occlu_res - 1)];
    const int32_t coeff100_idx =
        input_ids[min(quat.x + 1, occlu_res - 1) * r2 + quat.y * r1 + quat.z];
    const int32_t coeff101_idx =
        input_ids[min(quat.x + 1, occlu_res - 1) * r2 + quat.y * r1 + min(quat.z + 1, occlu_res - 1)];
    const int32_t coeff110_idx =
        input_ids[min(quat.x + 1, occlu_res - 1) * r2 + min(quat.y + 1, occlu_res - 1) * r1 + quat.z];
    const int32_t coeff111_idx =
        input_ids[min(quat.x + 1, occlu_res - 1) * r2 + min(quat.y + 1, occlu_res - 1) * r1 +
                  min(quat.z + 1, occlu_res - 1)];
    
    output_ids[ray_id * 8 + 0] = coeff000_idx;
    output_ids[ray_id * 8 + 1] = coeff001_idx;
    output_ids[ray_id * 8 + 2] = coeff010_idx;
    output_ids[ray_id * 8 + 3] = coeff011_idx;
    output_ids[ray_id * 8 + 4] = coeff100_idx;
    output_ids[ray_id * 8 + 5] = coeff101_idx;
    output_ids[ray_id * 8 + 6] = coeff110_idx;
    output_ids[ray_id * 8 + 7] = coeff111_idx;

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
        output_coeff[i] = weight000 * coeffs000[i] + weight001 * coeffs001[i] +
                          weight010 * coeffs010[i] + weight011 * coeffs011[i] +
                          weight100 * coeffs100[i] + weight101 * coeffs101[i] +
                          weight110 * coeffs110[i] + weight111 * coeffs111[i];
    }
}

__global__ void SH_reconstruction_kernel(const uint32_t num_rays, const uint32_t sh_degree,
                                         const uint32_t channels, const uint32_t num_samples,
                                         const bool jitter,
                                         const float *__restrict__ coeffs_ptr, // [num_rays, C, d2]
                                         const float *__restrict__ lobes_ptr,  // [num_rays, 3]
                                         const float *__restrict__ roughness_ptr, // [num_rays, 1]
                                         // output
                                         float *__restrict__ output_recon // [num_rays, C]
) {
    const int32_t ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= num_rays) {
        return;
    }

    // get the normal
    const float roughness = roughness_ptr[ray_id];
    const float3 L = make_float3(lobes_ptr[ray_id * 3 + 0], lobes_ptr[ray_id * 3 + 1],
                                 lobes_ptr[ray_id * 3 + 2]);
    if (L.x == 0.0 && L.y == 0 && L.z == 0)
        return;

    // sampling and calculate SH
    // from tangent-space H vector to world-space sample vector
    float3 up = fabsf(L.z) < 0.999 ? make_float3(0.0, 0.0, 1.0) : make_float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, L));
    float3 bitangent = cross(L, tangent);

    float eps = 0.0f;
    if (jitter) {
        curandState_t state;
        curand_init(1234, ray_id, 0, &state);
        eps = curand_uniform(&state);
    }

    // float *curr_sample_dir = output_recon + ray_id * num_samples * 3;
    const uint32_t d2 = sh_degree * sh_degree;
    float *curr_output = output_recon + ray_id * channels;          // [C]
    const float *curr_coeffs = coeffs_ptr + ray_id * d2 * channels; // [C, d2]
    for (uint32_t i = 0u; i < num_samples; ++i) {
        float2 Xi = Hammersley(i, num_samples);
        float3 sample_dir = importanceSampleGGX(Xi, L, roughness, eps);

        // const float phi = 2.0f * M_PIf * (Xi.x + eps);
        // // const float cosTheta = 1.0 - Xi.y; // uniform
        // const float cosTheta = sqrt(1.0 - Xi.y); // cos
        // const float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        // // from spherical coordinates to cartesian coordinates - halfway vector
        // float3 H = make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
        // H = normalize(H);

        // float3 sample_dir = normalize(tangent * H.x + bitangent * H.y + L * H.z);

        // const uint32_t offset = i * 3;
        // curr_sample_dir[offset + 0] = sample_dir.x;
        // curr_sample_dir[offset + 1] = sample_dir.y;
        // curr_sample_dir[offset + 2] = sample_dir.z;
        const float x = sample_dir.x, y = sample_dir.y, z = sample_dir.z;
        // conduct SH reconstruction
        for (uint32_t c = 0u; c < channels; ++c) {
            float accum = 0.0f;
            const float *coeffs = curr_coeffs + c * d2; // [d2]
            accum += 0.28209479177387814f * coeffs[0];
            if (sh_degree <= 1) {
                continue;
            }
            accum += -0.4886025119029199f * y * coeffs[1];
            accum += 0.4886025119029199f * z * coeffs[2];
            accum += -0.4886025119029199f * x * coeffs[3];
            if (sh_degree <= 2) {
                continue;
            }
            const float xy = x * y, yz = y * z, xz = x * z, xx = x * x, yy = y * y, zz = z * z;
            accum += 1.0925484305920792f * xy * coeffs[4];
            accum += -1.0925484305920792f * yz * coeffs[5];
            accum += 0.31539156525252005 * (2.0 * zz - xx - yy) * coeffs[6];
            accum += -1.0925484305920792f * xz * coeffs[7];
            accum += 0.5462742152960396f * (xx - yy) * coeffs[8];
            if (sh_degree <= 3) {
                continue;
            }
            accum += -0.5900435899266435f * y * (3.0f * xx - yy) * coeffs[9];
            accum += 2.890611442640554f * xy * z * coeffs[10];
            accum += -0.4570457994644658f * y * (4.0f * zz - xx - yy) * coeffs[11];
            accum += 0.3731763325901154f * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * coeffs[12];
            accum += -0.4570457994644658f * x * (4.0f * zz - xx - yy) * coeffs[13];
            accum += 1.4453057213202769f * z * (xx - yy) * coeffs[14];
            accum += -0.5900435899266435f * x * (xx - 3.0f * yy) * coeffs[15];

            curr_output[c] += clamp(accum, 0.0f, 1.0f);
        }
    }
    for (uint32_t c = 0u; c < channels; ++c) {
        curr_output[c] /= float(num_samples);
    }

}

__global__ void
dialate_occlusion_ids_kernel(const int32_t occlu_res,
                              int32_t *__restrict__ input_ids // [occlu_res, occlu_res, occlu_res]
) {
    const int32_t grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= occlu_res * occlu_res * occlu_res) {
        return;
    }

    if (input_ids[grid_id] >= 0)
        return;

    const int32_t r2 = occlu_res * occlu_res, r1 = occlu_res;
    const int32_t x_idx = grid_id / r2;
    const int32_t y_idx = (grid_id / r1) % r1;
    const int32_t z_idx = grid_id % r1;

    // find the neighbors
    // // 8 vertices
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, +1, +1, +1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, +1, +1, -1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, +1, -1, +1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, +1, -1, -1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, -1, +1, +1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, -1, +1, -1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, -1, -1, +1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, -1, -1, -1);
    // // 12 edges
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, 0, +1, -1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, 0, +1, +1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, +1, -1, 0);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, +1, 0, -1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, +1, 0, +1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, +1, +1, 0);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, -1, -1, 0);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, -1, 0, -1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, -1, 0, +1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, -1, +1, 0);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, 0, -1, -1);
    // fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, 0, -1, +1);
    // 6 nearby
    fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, 0, 0, +1);
    fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, 0, +1, 0);
    fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, +1, 0, 0);
    fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, -1, 0, 0);
    fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, 0, -1, 0);
    fillup(input_ids, occlu_res, x_idx, y_idx, z_idx, 0, 0, -1);
}

// wrap functions
std::tuple<Tensor, Tensor>
GSIR::sparse_interpolate_coefficients(const Tensor occlusion_coefficients, // [num_grid, d2]
                                      const Tensor occlusion_ids, // [occlu_res, occlu_res, occlu_res]
                                      const Tensor aabb,           // [6]
                                      const Tensor points,         // [num_rays, 3]
                                      const Tensor normals,        // [num_rays, 3]
                                      const uint32_t sh_degree) {
    CHECK_INPUT(occlusion_coefficients);
    CHECK_INPUT(occlusion_ids);
    CHECK_INPUT(points);
    CHECK_INPUT(normals);

    const torch::Device device = normals.device();
    const uint32_t num_rays = normals.size(0);
    const uint32_t occlu_res = occlusion_ids.size(0);

    const uint32_t blocks = div_round_up(num_rays, THREADS);

    Tensor output_coefficients =
        torch::zeros({num_rays, sh_degree * sh_degree, 1},
                     torch::TensorOptions().dtype(torch::kFloat).device(device));

    Tensor output_ids =
        torch::zeros({num_rays, 8},
                     torch::TensorOptions().dtype(torch::kInt).device(device));

    sparse_interpolate_coefficients_kernel<<<blocks, THREADS>>>(
        num_rays, sh_degree, occlu_res,
        occlusion_coefficients.data_ptr<float>(), // [num_grid, d2]
        occlusion_ids.data_ptr<int32_t>(),        // [occlu_res, occlu_res, occlu_res]
        aabb.data_ptr<float>(),                    // [6]
        points.data_ptr<float>(),                  // [num_rays, 3]
        normals.data_ptr<float>(),                 // [num_rays, 3]
        // output
        output_coefficients.data_ptr<float>(), // [num_rays, d2, 1]
        output_ids.data_ptr<int32_t>()  // [num_rays, 8]
    );

    return std::make_tuple(output_coefficients, output_ids);
}

Tensor GSIR::SH_reconstruction(const Tensor coefficients, // [num_rays, d2, C]
                               const Tensor lobes,        // [num_rays, 3]
                               const Tensor roughness,    // [num_rays, 1]
                               const uint32_t num_samples, const uint32_t sh_degree) {
    CHECK_INPUT(coefficients);
    CHECK_INPUT(lobes);
    CHECK_INPUT(roughness);

    const torch::Device device = lobes.device();
    const uint32_t num_rays = lobes.size(0);
    const uint32_t C = coefficients.size(1);

    const uint32_t blocks = div_round_up(num_rays, THREADS);

    Tensor reconstruction =
        torch::zeros({num_rays, C}, torch::TensorOptions().dtype(torch::kFloat).device(device));

    SH_reconstruction_kernel<<<blocks, THREADS>>>(
        num_rays, sh_degree, C, num_samples, true,
        coefficients.data_ptr<float>(), // [num_rays, C, d2]
        lobes.data_ptr<float>(),        // [num_rays, 3]
        roughness.data_ptr<float>(),    // [num_rays, 1]
        // output
        reconstruction.data_ptr<float>() // [num_rays, C]
    );

    return reconstruction;
}

void GSIR::dialate_occlusion_ids(Tensor occlusion_ids) {
    CHECK_INPUT(occlusion_ids);

    const uint32_t occlu_res = occlusion_ids.size(0);
    const uint32_t blocks = div_round_up(occlu_res * occlu_res * occlu_res, THREADS);

    dialate_occlusion_ids_kernel<<<blocks, THREADS>>>(
        int32_t(occlu_res),
        occlusion_ids.data_ptr<int32_t>() // [occlu_res, occlu_res, occlu_res]
    );
}
