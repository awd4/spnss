from distutils.core import setup
from Cython.Build import cythonize
import numpy as np


setup(
    name = 'spnss',
    ext_modules = cythonize(['spnss/*.pyx','spnss/test/*.pyx']),
    include_dirs = [np.get_include()]
)

