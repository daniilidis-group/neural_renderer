# CXX=g++-4.9 CC=gcc-4.9 python setup_jit.py
import unittest
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
