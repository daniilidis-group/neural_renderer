import torch
import torch.nn as nn

import neural_renderer

class Mesh(nn.Module):

    def __init__(self, filename_obj, texture_size=4, normalization=True):
        super(Mesh, self).__init__()
        vertices, faces = neural_renderer.load_obj(filename_obj, normalization)
        self.vertices = nn.Parameter(torch.from_numpy(vertices))
        self.faces = nn.Parameter(torch.from_numpy(faces), requires_grad=False)
        self.num_vertices = self.vertices.shape[0]
        self.num_faces = self.faces.shape[0]

        # create textures
        shape = (self.num_faces, texture_size, texture_size, texture_size, 3)
        self.textures = nn.Parameter(0.05*torch.randn(*shape))
        self.texture_size = texture_size

    def forward(self):
        return self.vertices, self.faces, self.textures.sigmoid()

