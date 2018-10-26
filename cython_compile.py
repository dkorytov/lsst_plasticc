from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('util.pyx'))
#setup(ext_modules = cythonize('summary_statistics.pyx'))
