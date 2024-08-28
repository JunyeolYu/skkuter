from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='skkuter_op',
    ext_modules=[CppExtension('skkuter_op', ['skkuter_op.cpp'])],
    cmdclass={'build_ext': BuildExtension}
)