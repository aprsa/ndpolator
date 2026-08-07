from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import numpy


class BuildExtFlags(build_ext):
    """Custom build_ext command to add compiler-specific options."""

    def build_extensions(self):
        # we need to fiddle with compiler flags based on the compiler used (gcc, clang, msvc)
        compiler_type = self.compiler.compiler_type
        for ext in self.extensions:
            if compiler_type == 'msvc':
                ext.extra_compile_args = ['/O2', '/WX']
                ext.extra_link_args = []
            else:
                ext.extra_compile_args = ['-O3', '-Wall', '-Werror']
                ext.extra_link_args = ['-lm', '-pthread']
        super().build_extensions()


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
        include_dirs=['src', 'external/kdtree', numpy.get_include()],
    ),
]

setup(
    cmdclass={'build_ext': BuildExtFlags},
    ext_modules=ext_modules,
)
