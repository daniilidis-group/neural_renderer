import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

cuda_cflags = []
rasterize_cuda = load(
        'rasterize_cuda', [os.path.join(dir_path,'rasterize_cuda.cpp'),
                          os.path.join(dir_path, 'rasterize_cuda_kernel.cu')],
        verbose=True, extra_cuda_cflags=cuda_cflags)

class RasterizeFunction(Function):

    @staticmethod
    def forward(ctx, faces, textures, image_size, near, far, eps, background_color,
                return_rgb=False, return_alpha=False, return_depth=False):
        ctx.image_size = image_size
        ctx.near = near
        ctx.far = far
        ctx.eps = eps
        ctx.background_color = background_color
        ctx.return_rgb = return_rgb
        ctx.return_alpha = return_alpha
        ctx.return_depth = return_depth

        faces = faces.clone()

        ctx.device = faces.device
        ctx.batch_size, ctx.num_faces = faces.shape[:2]

        if ctx.return_rgb:
            textures = textures.contiguous()
            ctx.texture_size = textures.shape[2]
        else:
            # initializing with dummy values
            textures = torch.cuda.FloatTensor(1).fill_(0)
            ctx.texture_size = None

        face_index_map = torch.cuda.IntTensor(ctx.batch_size, ctx.image_size, ctx.image_size).fill_(-1)
        weight_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_size, ctx.image_size, 3).fill_(0.0)
        depth_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_size, ctx.image_size).fill_(ctx.far)

        if ctx.return_rgb:
            rgb_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_size, ctx.image_size, 3).fill_(0)
            sampling_index_map = torch.cuda.IntTensor(ctx.batch_size, ctx.image_size, ctx.image_size, 8).fill_(0)
            sampling_weight_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_size, ctx.image_size, 8).fill_(0)
        else:
            rgb_map = torch.cuda.FloatTensor(1).fill_(0)
            sampling_index_map = torch.cuda.FloatTensor(1).fill_(0)
            sampling_weight_map = torch.cuda.FloatTensor(1).fill_(0)
        if ctx.return_alpha:
            alpha_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_size, ctx.image_size).fill_(0)
        else:
            alpha_map = torch.cuda.FloatTensor(1).fill_(0)
        if ctx.return_depth:
            face_inv_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_size, ctx.image_size, 3, 3).fill_(0)
        else:
            face_inv_map = torch.cuda.FloatTensor(1).fill_(0)

        face_index_map, weight_map, depth_map, face_inv_map =\
            RasterizeFunction.forward_face_index_map(ctx, faces, face_index_map,
                                                     weight_map, depth_map,
                                                     face_inv_map)
        rgb_map, sampling_index_map, sampling_weight_map =\
                RasterizeFunction.forward_texture_sampling(ctx, faces, textures,
                                                           face_index_map, weight_map,
                                                           depth_map, rgb_map,
                                                           sampling_index_map,
                                                           sampling_weight_map)
                
        rgb_map = RasterizeFunction.forward_background(ctx, face_index_map, rgb_map)
        alpha_map = RasterizeFunction.forward_alpha_map(ctx, alpha_map, face_index_map)

        ctx.save_for_backward(faces, textures, face_index_map, weight_map,
                              depth_map, rgb_map, alpha_map, face_inv_map,
                              sampling_index_map, sampling_weight_map)


        rgb_r, alpha_r, depth_r = torch.tensor([]), torch.tensor([]), torch.tensor([])
        if ctx.return_rgb:
            rgb_r = rgb_map
        if ctx.return_alpha:
            alpha_r = alpha_map.clone()
        if ctx.return_depth:
            depth_r = depth_map.clone()
        return rgb_r, alpha_r, depth_r

    @staticmethod
    def backward(ctx, grad_rgb_map, grad_alpha_map, grad_depth_map):
        faces, textures, face_index_map, weight_map,\
        depth_map, rgb_map, alpha_map, face_inv_map,\
        sampling_index_map, sampling_weight_map = \
                ctx.saved_tensors
        # initialize output buffers
        # no need for explicit allocation of cuda.FloatTensor because zeros_like does it automatically
        grad_faces = torch.zeros_like(faces, dtype=torch.float32).to(ctx.device).contiguous()
        grad_faces = torch.zeros_like(faces, dtype=torch.float32).to(ctx.device).contiguous()
        if ctx.return_rgb:
            grad_textures = torch.zeros_like(textures, dtype=torch.float32).to(ctx.device).contiguous()
        else:
            grad_textures = torch.FloatTensor(1).fill_(0.0)
        
        # get grad_outputs
        if ctx.return_rgb:
            if grad_rgb_map is not None:
                grad_rgb_map = grad_rgb_map.contiguous()
            else:
                grad_rgb_map = torch.zeros_like(rgb_map)
        else:
            grad_rgb_map = torch.FloatTensor(1).fill_(0.0)
        if ctx.return_alpha:
            if grad_alpha_map is not None:
                grad_alpha_map = grad_alpha_map.contiguous()
            else:
                grad_alpha_map = torch.zeros_like(alpha_map)
        else:
            grad_alpha_map = torch.FloatTensor(1).fill_(0.0)
        if ctx.return_depth:
            if grad_depth_map is not None:
                grad_depth_map = grad_depth_map.contiguous()
            else:
                grad_depth_map = torch.zeros_like(ctx.depth_map)
        else:
            grad_depth_map = torch.FloatTensor(1).fill_(0.0)

        # backward pass
        grad_faces = RasterizeFunction.backward_pixel_map(
                                        ctx, faces, face_index_map, rgb_map,
                                        alpha_map, grad_rgb_map, grad_alpha_map,
                                        grad_faces)
        grad_textures = RasterizeFunction.backward_textures(
                                        ctx, face_index_map, sampling_weight_map,
                                        sampling_index_map, grad_rgb_map, grad_textures)
        grad_faces = RasterizeFunction.backward_depth_map(
                                        ctx, faces, depth_map, face_index_map,
                                        face_inv_map, weight_map, grad_depth_map,
                                        grad_faces)

        if not textures.requires_grad:
            grad_textures = None

        return grad_faces, grad_textures, None, None, None, None, None, None, None, None

    @staticmethod
    def forward_face_index_map(ctx, faces, face_index_map, weight_map, 
                               depth_map, face_inv_map):
        lock = torch.zeros_like(face_index_map, dtype=torch.int32)
        return rasterize_cuda.forward_face_index_map(faces, face_index_map, weight_map,
                                        depth_map, face_inv_map, lock,
                                        ctx.image_size, ctx.near, ctx.far,
                                        ctx.return_rgb, ctx.return_alpha,
                                        ctx.return_depth)

    @staticmethod
    def forward_texture_sampling(ctx, faces, textures, face_index_map,
                                 weight_map, depth_map, rgb_map,
                                 sampling_index_map, sampling_weight_map):
        if not ctx.return_rgb:
            return rgb_map, sampling_index_map, sampling_weight_map
        else:
            return rasterize_cuda.forward_texture_sampling(faces, textures, face_index_map,
                                           weight_map, depth_map, rgb_map,
                                           sampling_index_map, sampling_weight_map,
                                           ctx.image_size, ctx.eps)

    @staticmethod
    def forward_alpha_map(ctx, alpha_map, face_index_map):
        if ctx.return_alpha:
            alpha_map[face_index_map >= 0] = 1
        return alpha_map

    @staticmethod
    def forward_background(ctx, face_index_map, rgb_map):
        if ctx.return_rgb:
            background_color = torch.cuda.FloatTensor(ctx.background_color)
            mask = (face_index_map >= 0).float()[:, :, :, None]
            if background_color.ndimension() == 1:
                rgb_map = rgb_map * mask + (1-mask) * background_color[None, None, None, :]
            elif background_color.ndimension() == 2:
                rgb_map = rgb_map * mask + (1-mask) * background_color[:, None, None, :]
        return rgb_map

    @staticmethod
    def backward_pixel_map(ctx, faces, face_index_map, rgb_map,
                           alpha_map, grad_rgb_map, grad_alpha_map, grad_faces):
        if (not ctx.return_rgb) and (not ctx.return_alpha):
            return grad_faces
        else:
            return rasterize_cuda.backward_pixel_map(faces, face_index_map, rgb_map,
                                     alpha_map, grad_rgb_map, grad_alpha_map,
                                     grad_faces, ctx.image_size, ctx.eps, ctx.return_rgb,
                                     ctx.return_alpha)

    @staticmethod
    def backward_textures(ctx, face_index_map, sampling_weight_map,
                          sampling_index_map, grad_rgb_map, grad_textures):
        if not ctx.return_rgb:
            return grad_textures
        else:
            return rasterize_cuda.backward_textures(face_index_map, sampling_weight_map,
                                                    sampling_index_map, grad_rgb_map,
                                                    grad_textures, ctx.num_faces)

    @staticmethod
    def backward_depth_map(ctx, faces, depth_map, face_index_map,
                           face_inv_map, weight_map, grad_depth_map, grad_faces):
        if not ctx.return_depth:
            return grad_faces
        else:
            return rasterize_cuda.backward_depth_map(faces, depth_map, face_index_map,
                                     face_inv_map, weight_map,
                                     grad_depth_map, grad_faces, ctx.image_size)

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

    def forward(self, faces, textures):
        return RasterizeFunction.apply(faces, textures, self.image_size, self.near, self.far,
                                       self.eps, self.background_color,
                                       self.return_rgb, self.return_alpha, self.return_depth)
