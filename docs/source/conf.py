# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import fretbursts

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from importlib.metadata import version as get_version
project = 'FRETBursts'
copyright = '2026, Paul David Harris'
author = 'Paul David Harris'
release:str = get_version('fretbursts')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.inheritance_diagram',
        'sphinx.ext.autosummary',
        'sphinx.ext.mathjax',
        'sphinx.ext.intersphinx',
        'sphinx.ext.napoleon',
        'sphinx_copybutton',
        'IPython.sphinxext.ipython_console_highlighting',
        'IPython.sphinxext.ipython_directive',
        'nbsphinx',
]
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
