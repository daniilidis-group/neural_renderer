import math

import torch

import neural_renderer as nr


class Renderer():
    def __init__(self):
        # rendering
        self.image_size = 256
        self.anti_aliasing = True
        self.background_color = [0, 0, 0]
        self.fill_back = True

        # camera
        self.perspective = True
        self.viewing_angle = 30
        self.eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]
        self.camera_mode = 'look_at'
        self.camera_direction = [0, 0, 1]
        self.near = 0.1
        self.far = 100

        # light
        self.light_intensity_ambient = 0.5
        self.light_intensity_directional = 0.5
        self.light_color_ambient = [1, 1, 1]  # white
        self.light_color_directional = [1, 1, 1]  # white
        self.light_direction = [0, 1, 0]  # up-to-down

        # rasterization
        self.rasterizer_eps = 1e-3

    def render_silhouettes(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)

        # perspective transformation
        if self.perspective:
            vertices = nr.perspective(vertices, angle=self.viewing_angle)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def render_depth(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)

        # perspective transformation
        if self.perspective:
            vertices = nr.perspective(vertices, angle=self.viewing_angle)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render(self, vertices, faces, textures):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)

        # perspective transformation
        if self.perspective:
            vertices = nr.perspective(vertices, angle=self.viewing_angle)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return images
