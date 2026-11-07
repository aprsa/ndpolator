from .ndpolator import *

# The version is defined once, in pyproject.toml; setuptools records it in
# the package metadata at build time and we read it back from there.
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version('ndpolator')
except PackageNotFoundError:  # source tree that has not been installed
    __version__ = 'unknown'

del _version, PackageNotFoundError
