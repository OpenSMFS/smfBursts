from setuptools import setup, Extension
import numpy as np
# from Cython.Build import cythonize

projectname = 'smfbursts/'

smfdir = projectname + "cfuncs/"

extension = [Extension("smfbursts.cfuncs", [smfdir+"burstsearch.c", smfdir+"smfbursts_parfuncs.c",
                                             smfdir+"smfbursts_cfuncs.c", smfdir+"kde.c"], 
                       include_dirs=[np.get_include()]),]

setup(name='smfbursts',
      ext_modules=extension,
      include_pakage_data = True,
      packages = ['smfbursts','smfbursts.cfuncs'])
