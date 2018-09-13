# CXX=g++-4.9 CC=gcc-4.9 python setup_jit.py install # or develop
import unittest
from setuptools import setup, find_packages
from torch.utils.cpp_extension import load
import os

cur_dir = os.path.abspath(os.path.dirname(__file__))
build_dir = os.path.join(cur_dir, 'neural_renderer/cuda')


def test_all():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite


os.system('rm -rf {}/*.so'.format(build_dir))

load_textures = load(
    'load_textures', ['neural_renderer/cuda/load_textures_cuda.cpp', 
    'neural_renderer/cuda/load_textures_cuda_kernel.cu'], build_directory=build_dir, verbose=True)
rasterize = load(
    'rasterize', ['neural_renderer/cuda/rasterize_cuda.cpp', 
    'neural_renderer/cuda/rasterize_cuda_kernel.cu'], build_directory=build_dir, verbose=True)
create_texture_image = load(
    'create_texture_image', ['neural_renderer/cuda/create_texture_image_cuda.cpp', 
    'neural_renderer/cuda/create_texture_image_cuda_kernel.cu'], build_directory=build_dir, verbose=True)

help(load_textures)
help(rasterize)
help(create_texture_image)


INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setup(
    description='PyTorch implementation of "A 3D mesh renderer for neural networks"',
    author='Nikolaos Kolotouros',
    author_email='nkolot@seas.upenn.edu',
    license='MIT License',
    version='1.1.3',
    name='neural_renderer',
    test_suite='setup.test_all',
    packages=['neural_renderer', 'neural_renderer.cuda'],
    install_requires=INSTALL_REQUIREMENTS,
    # ext_modules=ext_modules,
    # cmdclass = {'build_ext': BuildExtension}
)