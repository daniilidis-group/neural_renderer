import math

import torch

def perspective(vertices, angle=30.):
        assert(vertices.ndimension() == 3)
        device = vertices.device
        angle = torch.tensor(angle / 180 * math.pi).to(device)
        angle = angle[None]

        width = torch.tan(angle)
        width = width[:, None] 
        z = vertices[:, :, 2]
        x = vertices[:, :, 0] / z / width
        y = vertices[:, :, 1] / z / width
        vertices = torch.stack((x,y,z), dim=2)
        return vertices
