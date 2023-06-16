from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [Extension('helloworld', ["helloworld.pyx"])]
setup(ext_modules = cythonize(extensions))

extensions = [Extension('permutation_matrix', ["permutation_matrix.pyx"])]
setup(ext_modules = cythonize(extensions))



