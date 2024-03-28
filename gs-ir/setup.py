#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

os.path.dirname(os.path.abspath(__file__))

setup(
    name="gs_ir",
    packages=["gs_ir"],
    ext_modules=[
        CUDAExtension(
            name="gs_ir._C",
            sources=[
                "src/bindings.cpp",
                "src/irradiance_kernel.cu",
                "src/occlusion_kernel.cu",
            ],
            extra_compile_args={"nvcc": ["-I"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
