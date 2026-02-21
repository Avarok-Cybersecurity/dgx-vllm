import os
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda')

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="sparse_fp4_v7",
    ext_modules=[CUDAExtension(
        "sparse_fp4_v7",
        ["sparse_fp4_v7.cu"],
        extra_compile_args={
            "nvcc": ["-O3", "--use_fast_math", "-arch=sm_120"]
        }
    )],
    cmdclass={"build_ext": BuildExtension}
)
