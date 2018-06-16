import setuptools

import neural_renderer

setuptools.setup(
    description='A 3D mesh renderer for neural networks',
    author='Hiroharu Kato',
    author_email='hiroharu.kato.1989.10.13@gmail.com',
    url='http://hiroharu-kato.com/projects_en/neural_renderer.html',
    license='MIT License',
    version=neural_renderer.__version__,
    name='neural_renderer',
    test_suite='tests',
    packages=['neural_renderer'],
    install_requires=['numpy', 'chainer', 'cupy', 'tqdm', 'scikit-image'],
)
