from distutils.core import setup, Extension

mat_module = Extension('mat', sources=['matmodule.cpp'])

setup(name='mat',
      version='0.0.1',
      description='description',
      ext_modules=[mat_module])
