#include <torch/torch.h>

// CUDA forward declarations

at::Tensor load_textures_cuda(
        at::Tensor image,
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor is_update,
        int texture_wrapping,
        int use_bilinear);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor load_textures(
        at::Tensor image,
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor is_update,
        int texture_wrapping,
        int use_bilinear) {

    CHECK_INPUT(image);
    CHECK_INPUT(faces);
    CHECK_INPUT(is_update);
    CHECK_INPUT(textures);

    return load_textures_cuda(image, faces, textures, is_update, texture_wrapping, use_bilinear);
                                      
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_textures", &load_textures, "LOAD_TEXTURES (CUDA)");
}
