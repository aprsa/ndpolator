from setuptools import Extension, setup
import numpy

ext_modules = [
    Extension(
        'cndpolator',
        sources=[
            'src/ndpolator.c',
            'src/ndp_types.c',
        ],
        language='c',
        extra_compile_args=["-Werror", "-O2"],
        # extra_compile_args=["-Werror", "-O0", "-g"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    ext_modules=ext_modules,
)
