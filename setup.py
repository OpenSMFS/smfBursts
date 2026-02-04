from setuptools import setup, Extension
import numpy as np
# from Cython.Build import cythonize

projectname = 'fretbursts/'

fbcdir = projectname + "cfuncs/"

extension = [Extension("fretbursts.cfuncs", [fbcdir+"burstsearch.c", fbcdir+"fretbursts_parfuncs.c",
                                             fbcdir+"fretbursts_cfuncs.c", fbcdir+"kde.c"], 
                       include_dirs=[np.get_include()]),]
             # cythonize([Extension('fretbursts.phrates', [projectname+'phtools/phrates_cy.pyx'], include_dirs=[np.get_include(), '.'])])]

setup(name='fretbursts',
      version='1.0.1',
      ext_modules=extension,
      include_pakage_data = True,
      packages = ['fretbursts','fretbursts.cfuncs'])
