from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='skkuter_op',
    ext_modules=[
        CppExtension(
            name='skkuter_op',
            sources=['skkuter_op.cpp'],
            extra_compile_args=['-O3'],
        )],
    cmdclass={'build_ext': BuildExtension}
)