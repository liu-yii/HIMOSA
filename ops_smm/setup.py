import os
import glob
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
from setuptools import setup, find_packages

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")  # Assuming source code is in src directory

    # Get C++ and CUDA source files
    source_cpp = glob.glob(os.path.join(extensions_dir, "*.cpp"))  # Can be ignored if no C++ files
    source_cuda = glob.glob(os.path.join(extensions_dir, "*.cu"))  # Find CUDA source files

    # Combine all source files
    sources = source_cpp + source_cuda
    extension = CppExtension  # Default to C++ extension
    extra_compile_args = {"cxx": []}
    define_macros = []

    # Switch to CUDA extension if CUDA is available
    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-gencode", "arch=compute_70,code=sm_70",
            "-gencode", "arch=compute_75,code=sm_75",
            "-gencode", "arch=compute_80,code=sm_80",  # For CUDA 11.7 and above
            "-gencode", "arch=compute_86,code=sm_86",
            "-gencode", "arch=compute_89,code=sm_89", # Comment this if CUDA version is too low
            "-lineinfo",  # Output detailed debug information
        ]
    else:
        raise NotImplementedError('CUDA is not available or not found.')

    # Ensure full paths for all source files
    sources = [os.path.join(extensions_dir, s) for s in sources]

    # Include directories for header files
    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            name="smm_cuda",  # Module name
            sources=sources,  # Source files
            include_dirs=include_dirs,  # Header directories
            define_macros=define_macros,  # Macro definitions
            extra_compile_args=extra_compile_args,  # Compilation options
        )
    ]
    return ext_modules


setup(
    name="smm_cuda",
    version="1.0",
    author="WeiLong",
    description="Sparse Matrix Multiplication (CUDA)",
    packages=find_packages(),
    ext_modules=get_extensions(),  # Get extension modules
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    install_requires=requirements,
)