from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy


setup(
    ext_modules=[
        Extension("Cspiketools", ["Cspiketools.pyx"],
                  include_dirs=[numpy.get_include()],
                  )
    ],
)

#extensions=[
#    Extension("Cspiketools",
#             ["Cspiketools.pyx"],
#             include_dirs=[numpy.get_include()],
#             #extra_compile_args=["-w"]
#
#            )
#]

#setup(
#    ext_modules=cythonize(extensions),
#

setup(
    ext_modules=cythonize("my_module.pyx"),
    include_dirs=[numpy.get_include()]
)