from setuptools import setup
import unittest

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def test_all():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

ext_modules=[
    CUDAExtension('load_textures_cuda', [
        'neural_renderer/cuda/load_textures_cuda.cpp',
        'neural_renderer/cuda/load_textures_cuda_kernel.cu',
        ]),
    CUDAExtension('rasterize_cuda', [
        'neural_renderer/cuda/rasterize_cuda.cpp',
        'neural_renderer/cuda/rasterize_cuda_kernel.cu',
        ]),
    CUDAExtension('create_texture_image_cuda', [
        'neural_renderer/cuda/create_texture_image_cuda.cpp',
        'neural_renderer/cuda/create_texture_image_cuda_kernel.cu',
        ])
    ]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setup(
    description='PyTorch implementation of "A 3D mesh renderer for neural networks"',
    author='Nikolaos Kolotouros',
    author_email='nkolot@seas.upenn.edu',
    license='MIT License',
    version='0.1.0x1.1.3',
    name='neural_renderer',
    test_suite='setup.test_all',
    packages=['neural_renderer'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
