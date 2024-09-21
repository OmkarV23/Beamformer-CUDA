from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import os
import pybind11

class custom_build_ext(build_ext):
    def build_extensions(self):
        # Compile CUDA source files into object files
        nvcc = 'nvcc'
        for ext in self.extensions:
            cuda_sources = [s for s in ext.sources if os.path.splitext(s)[1] == '.cu']
            other_sources = [s for s in ext.sources if os.path.splitext(s)[1] != '.cu']
            ext.sources = other_sources  # Remove CUDA sources from sources

            extra_objects = ext.extra_objects or []
            for source in cuda_sources:
                obj_file = os.path.splitext(source)[0] + '.o'
                include_dirs = ext.include_dirs or []
                include_args = [item for sublist in [['-I', inc] for inc in include_dirs] for item in sublist]

                # Set CUDA compiler flags
                nvcc_flags = [
                    '-c', source,
                    '-o', obj_file,
                    '-Xcompiler', '-fPIC',
                    '-std=c++17',
                    '-arch=sm_75'  # Replace with your GPU's compute capability
                ] + include_args

                print('Compiling CUDA source:', ' '.join([nvcc] + nvcc_flags))
                subprocess.check_call([nvcc] + nvcc_flags)
                extra_objects.append(obj_file)

            ext.extra_objects = extra_objects

        # Now call the original build_extensions method
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        'FastBeamformer',
        sources=[
            'src/ext.cpp',
            'src/_readsignal.cpp',
            'src/_signalprocessing.cpp',
            'src/beamformer.cu',
        ],
        include_dirs=[
            pybind11.get_include(),
            '/usr/include/boost',
            '/usr/local/cuda/include',  # CUDA headers
        ],
        library_dirs=[
            '/usr/lib',
            '/usr/local/lib',
            '/usr/local/cuda/lib64',    # CUDA libraries
        ],
        libraries=[
            'boost_filesystem',
            'boost_system',
            'fftw3',
            'cudart',  # CUDA runtime library
        ],
        extra_compile_args=[
            '-std=c++17',
        ],
        language='c++',
    ),
]

setup(
    name='FastBeamformer',
    version='0.1',
    ext_modules=ext_modules,
    cmdclass={'build_ext': custom_build_ext},
)

