from setuptools import Extension, setup
import numpy

ext_modules = [
    Extension(
        'cndpolator',
        sources=[
            # ndpolator sources:
            'src/ndp_types.c',
            'src/ndpolator.c',
            'src/ndp_py.c',  # Python wrappers
            # vendored kdtree source:
            'external/kdtree/kdtree.c',
        ],
        language='c',
        extra_link_args=['-lm', '-pthread'],
        extra_compile_args=["-Werror", "-O3"],
        # extra_compile_args=["-Werror", "-O0", "-g"],
        include_dirs=['src', 'external/kdtree', numpy.get_include()],
    ),
]

setup(
    ext_modules=ext_modules,
)
