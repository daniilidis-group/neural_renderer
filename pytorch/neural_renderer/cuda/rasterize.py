import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

# it needs at least cuda 6.0 to compile because of some atomic instructions
cuda_cflags = ['-arch=sm_60', '-gencode=arch=compute_60,code=sm_60', '-gencode=arch=compute_70,code=sm_70']
rasterize_cuda = load(
        'rasterize_cuda', [os.path.join(dir_path,'rasterize_cuda.cpp'),
                          os.path.join(dir_path, 'rasterize_cuda_kernel.cu')],
        verbose=True, extra_cuda_cflags=cuda_cflags)

class RasterizeFunction(Function):

    @staticmethod
    def forward(ctx, inputs, image_size, near, far, eps, background_color,
                return_rgb=False, return_alpha=False, return_depth=False):
        ctx.image_size = image_size
        ctx.near = near
        ctx.far = far
        ctx.eps = eps
        ctx.background_color = background_color
        ctx.return_rgb = return_rgb
        ctx.return_alpha = return_alpha
        ctx.return_depth = return_depth

        ctx.inputs = inputs
        ctx.faces = ctx.inputs[0].copy()

        ctx.device = ctx.faces.device
        ctx.batch_size, ctx.num_faces = ctx.faces.shape[:2]

        if ctx.return_rgb:
            ctx.textures = inputs[1].contiguous()
            ctx.texture_size = ctx.textures.shape[2]
        else:
            # initializing with dummy values
            ctx.textures = torch.zeros(1)
            ctx.texture_size = None

        ctx.face_index_map = -1 * torch.ones((ctx.batch_size, ctx.image_size, ctx.image_size), dtype=torch.int32).to(ctx.device)
        ctx.weight_map = torch.zeros((ctx.batch_size, ctx.image_size, ctx.image_size, 3), dtype=torch.float32).to(ctx.device)
        ctx.depth_map = torch.zeros_like(ctx.face_index_map, dtype=torch.float32).to(ctx.device) + ctx.far

        if ctx.return_rgb:
            ctx.rgb_map = torch.zeros((ctx.batch_size, ctx.image_size, ctx.image_size, 3), dtype=torch.float32).to(ctx.device)
            ctx.sampling_index_map = torch.zeros((ctx.batch_size, ctx.image_size, ctx.image_size, 8), dtype=torch.int32).to(ctx.device)
            ctx.sampling_weight_map = torch.zeros((ctx.batch_size, ctx.image_size, ctx.image_size, 8), dtype=torch.float32).to(ctx.device)

        else:
            ctx.rgb_map = torch.zeros(1, 'float32')
            ctx.sampling_index_map = torch.zeros(1, dtype=torch.int32).to(ctx.device)
            ctx.sampling_weight_map = torch.zeros(1, dtype=torch.float32).to(ctx.device)
        if ctx.return_alpha:
            ctx.alpha_map = torch.zeros((ctx.batch_size, ctx.image_size, ctx.image_size), dtype=torch.float32).to(ctx.device)
        else:
            ctx.alpha_map = torch.zeros(1, dtype=torch.float32).to(ctx.device)
        if ctx.return_depth:
            ctx.face_inv_map = torch.zeros((ctx.batch_size, ctx.image_size, ctx.image_size, 3, 3), dtype=torch.float32).to(ctx.device)
        else:
            ctx.face_inv_map = torch.zeros(1, torch.float32).to(ctx.device)

        RasterizeFunction.forward_face_index_map(ctx)
        RasterizeFunction.forward_texture_sampling(ctx)
        RasterizeFunction.forward_background(ctx)
        RasterizeFunction.forward_alpha_map(ctx)

        rgb_r, alpha_r, depth_r = None, None, None
        if ctx.return_rgb:
            rgb_r = ctx.rgb_map
        if ctx.return_alpha:
            alpha_r = ctx.alpha_map.copy()
        if ctx.return_depth:
            depth_r = ctx.depth_map.copy()
        return rgb_r, alpha_r, depth_r

    @staticmethod
    def backward(ctx, grad_outputs):
        # initialize output buffers
        ctx.grad_faces = torch.zeros_like(ctx.faces, dtype=torch.float32).to(ctx.device).contiguous()
        if ctx.return_rgb:
            ctx.grad_textures = torch.zeros_like(ctx.faces, dtype=torch.float32).to(ctx.device).contiguous()
        else:
            ctx.grad_textures = torch.zeros(1, dtype=torch.float32).to(ctx.device)
        
        # get grad_outputs
        if ctx.return_rgb:
            if grad_outputs[0] is not None:
                ctx.grad_rgb_map = grad_outputs[0].contiguous()
            else:
                ctx.grad_rgb_map = torch.zeros_like(ctx.rgb_map)
        else:
            ctx.grad_rgb_map = torch.zeros(1, dtype=torch.float32)
        if ctx.return_alpha:
            if grad_outputs[1] is not None:
                ctx.grad_alpha_map = grad_outputs[1].copy()
            else:
                ctx.grad_alpha_map = torch.zeros_like(ctx.alpha_map)
        else:
            ctx.grad_alpha_map = torch.zeros(1, torch.float32)
        if ctx.return_depth:
            if grad_outputs[2] is not None:
                ctx.grad_depth_map = grad_outputs[2].contiguous()
            else:
                ctx.grad_depth_map = torch.zeros_like(ctx.depth_map)
        else:
            torch.grad_depth_map = torch.zeros(1, torch.float32)

        # backward pass
        RasterizeFunction.backward_pixel_map(ctx)
        RasterizeFunction.backward_textures(ctx)
        RasterizeFunction.backward_depth_map(ctx)

        # return
        if len(ctx.inputs) == 1:
            return ctx.grad_faces
        else:
            return ctx.grad_faces, ctx.grad_textures

    @staticmethod
    def forward_face_index_map(ctx):
        lock = torch.zeros(ctx.face_index_map.shape, dtype=torch.int32).to(ctx.device)
        rasterize_cuda.forward_face_index_map(ctx.faces, ctx.face_index_map,ctx.weight_map,
                                        ctx.depth_map, ctx.face_inv_map, lock,
                                        ctx.image_size, ctx.near, ctx.far,
                                        ctx.return_rgb, ctx.return_alpha,
                                        ctx.return_depth)

    @staticmethod
    def forward_texture_sampling(ctx):
        if not ctx.return_rgb:
            return
        rasterize_cuda.forward_texture_sampling(ctx.faces, ctx.textures, ctx.face_index_map,
                                           ctx.weight_map, ctx.depth_map, ctx.rgb_map,
                                           ctx.sampling_index_map, ctx.sampling_weight_map,
                                           ctx.image_size, ctx.eps)

    @staticmethod
    def forward_alpha_map(ctx):
        if not ctx.return_alpha:
            return
        ctx.alpha_map[ctx.face_index_map >= 0] = 1

    @staticmethod
    def forward_background(ctx):
        if not ctx.return_rgb:
            return
        background_color = torch.tensor(ctx.background_color, dtype=torch.float32).to(ctx.device)
        mask = (ctx.face_index_map >= 0).float()[:, :, :, None]
        if background_color.ndimension() == 1:
            ctx.rgb_map = ctx.rgb_map * mask + (1-mask) * background_color[None, None, None, :]
        elif background_color.ndimension() == 2:
            ctx.rgb_map = ctx.rgb_map * mask + (1-mask) * background_color[:, None, None, :]
        return

    @staticmethod
    def backward_pixel_map(ctx):
        if (not ctx.return_rgb) and (not ctx.return_alpha):
            return
        rasterize_cuda.backward_pixel_map(ctx.faces, ctx.face_index_map, ctx.rgb_map,
                                     ctx.alpha_map, ctx.grad_rgb_map, ctx.grad_alpha_map,
                                     ctx.grad_faces, ctx.image_size, ctx.eps, ctx.return_rgb,
                                     ctx.return_alpha)

    @staticmethod
    def backward_textures(ctx):
        if not ctx.return_rgb:
            return
        rasterize_cuda.backward_textures(ctx.face_index_map, ctx.sampling_weight_map, ctx.sampling_index_map,
                                    ctx.grad_rgb_map, ctx.grad_textures, ctx.num_faces)

    @staticmethod
    def backward_depth_map(ctx):
        if not ctx.return_depth:
            return
        rasterize_cuda.backward_depth_map(ctx.faces, ctx.depth_map, ctx.face_index_map,
                                     ctx.face_inv_map, ctx.weight_map,
                                     ctx.grad_depth_map, ctx.grad_faces, ctx.image_size)

class Rasterize(nn.Module):
    
    def __init__(self, image_size, near, far, eps, background_color,
                 return_rgb=False, return_alpha=False, return_depth=False):
        super(Rasterize, self).__init__()
        self.image_size = image_size
        # arguments
        self.image_size = image_size
        self.near = near
        self.far = far
        self.eps = eps
        self.background_color = background_color
        self.return_rgb = return_rgb
        self.return_alpha = return_alpha
        self.return_depth = return_depth

    def forward(self, input):
        return RasterizeFunction.apply(input, self.image_size, self.near, self.far,
                                       self.eps, self.background_color,
                                       self.return_rgb, self.return_alpha, self.return_depth)
