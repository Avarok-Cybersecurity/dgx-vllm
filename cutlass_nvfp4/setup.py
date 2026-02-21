#!/usr/bin/env python3
"""
Setup script for CUDA FP4 PyTorch Extension

Builds our CUDA FP4 library as a PyTorch extension!

Installation:
    pip install -e .

Usage in Python:
    import torch
    import cuda_fp4_ext

    # Quantize
    fp4_data, scales, global_scale = cuda_fp4_ext.quantize(fp32_tensor)

    # GEMM
    output = cuda_fp4_ext.gemm(fp4_A, scales_A, scale_A,
                                fp4_B, scales_B, scale_B,
                                M, N, K)
"""

from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

# Extension module
ext_modules = [
    cpp_extension.CUDAExtension(
        name='cuda_fp4_ext',
        sources=[
            'cuda_fp4_pytorch_ext.cpp',
            'cuda_fp4_kernels.cu',
        ],
        include_dirs=[
            '.',
            '/workspace/cutlass/include',
        ],
        extra_compile_args={
            'cxx': [
                '-O3',
                '-std=c++17',
            ],
            'nvcc': [
                '-O3',
                '-std=c++17',
                '-arch=sm_121',
                '-gencode=arch=compute_121,code=sm_121',
                '-use_fast_math',
                '--expt-relaxed-constexpr',
                '-lineinfo',
            ]
        },
        libraries=[],
        define_macros=[
            ('TORCH_EXTENSION_NAME', 'cuda_fp4_ext'),
            ('CUDA_FP4_VERSION_MAJOR', '1'),
            ('CUDA_FP4_VERSION_MINOR', '0'),
            ('CUDA_FP4_VERSION_PATCH', '0'),
        ]
    )
]

setup(
    name='cuda_fp4_ext',
    version='1.0.0',
    author='Claude Code + Community',
    author_email='community@example.com',
    description='CUDA FP4 Extension for PyTorch - GB10 Blackwell Optimized',
    long_description=open('README_CUDA_FP4_EXTENSION.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
    install_requires=[
        'torch>=2.0.0',
        'numpy',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='cuda fp4 quantization pytorch machine-learning deep-learning',
    project_urls={
        'Documentation': 'https://github.com/yourrepo/cuda-fp4',
        'Source': 'https://github.com/yourrepo/cuda-fp4',
        'Tracker': 'https://github.com/yourrepo/cuda-fp4/issues',
    },
)
