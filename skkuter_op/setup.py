from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='skkuter_op',
    ext_modules=[
        CUDAExtension(
            name='skkuter_op',
            sources=['skkuter_op.cpp', 'cuDecoder/decoder.cu'],

            extra_compile_args={'cxx': ['-O3'],'nvcc': ['-O3']},
            extra_link_flags=['-Wl,--no-as-needed', '-lcuda']
        )],
    cmdclass={'build_ext': BuildExtension}
)
