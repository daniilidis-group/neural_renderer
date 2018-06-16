import chainer
import chainer.functions as cf

import neural_renderer


class Mesh(chainer.Link):
    def __init__(self, filename_obj, texture_size=4, normalization=True):
        super(Mesh, self).__init__()

        with self.init_scope():
            # load .obj
            vertices, faces = neural_renderer.load_obj(filename_obj, normalization)
            self.vertices = chainer.Parameter(vertices)
            self.faces = faces
            self.num_vertices = self.vertices.shape[0]
            self.num_faces = self.faces.shape[0]

            # create textures
            init = chainer.initializers.Normal()
            shape = (self.num_faces, texture_size, texture_size, texture_size, 3)
            self.textures = chainer.Parameter(init, shape)
            self.texture_size = texture_size

    def to_gpu(self, device=None):
        super(Mesh, self).to_gpu(device)
        self.faces = chainer.cuda.to_gpu(self.faces, device)

    def get_batch(self, batch_size):
        # broadcast for minibatch
        vertices = cf.broadcast_to(self.vertices, [batch_size] + list(self.vertices.shape))
        faces = cf.broadcast_to(self.faces, [batch_size] + list(self.faces.shape)).data
        textures = cf.sigmoid(cf.broadcast_to(self.textures, [batch_size] + list(self.textures.shape)))
        return vertices, faces, textures

    def set_lr(self, lr_vertices, lr_textures):
        self.vertices.lr = lr_vertices
        self.textures.lr = lr_textures
