// Copyright 2023 Zhihao Liang
#include <torch/extension.h>

#include "irradiance_kernel.hpp"
#include "occlusion_kernel.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // occlusion
    m.def("sparse_interpolate_coefficients", &GSIR::sparse_interpolate_coefficients);
    m.def("SH_reconstruction", &GSIR::SH_reconstruction);
    m.def("dialate_occlusion_ids", &GSIR::dialate_occlusion_ids);
    // irradiance
    m.def("trilinear_interpolate_coefficients_forward", &GSIR::trilinear_interpolate_coefficients_forward);
    m.def("trilinear_interpolate_coefficients_backward", &GSIR::trilinear_interpolate_coefficients_backward);
}