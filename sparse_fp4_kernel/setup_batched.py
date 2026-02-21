from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sparse_fp4_batched',
    ext_modules=[
        CUDAExtension(
            name='sparse_fp4_batched',
            sources=['sparse_fp4_batched.cu'],
            extra_compile_args={
                'nvcc': ['-O3', '--use_fast_math', '-arch=sm_120'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
