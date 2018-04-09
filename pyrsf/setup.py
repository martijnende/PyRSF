from distutils.core import setup
from Cython.Build import cythonize
import os

os.environ['CFLAGS'] = '-O3 -Wall -std=c++11'

setup(
    ext_modules=cythonize("friction_rsf_opt.pyx", annotate=True)
)