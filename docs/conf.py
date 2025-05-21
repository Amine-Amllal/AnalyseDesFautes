# -- Project information -----------------------------------------------------

project = 'Analyse Des Fautes'
copyright = '2025, AMLLAL Amine, AKEBLI FatimaEzzahrae, ELHAKIOUI Asmae'
author = 'AMLLAL Amine, AKEBLI FatimaEzzahrae, ELHAKIOUI Asmae'
release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'myst_parser',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'fr'
source_suffix = ['.rst', '.md']

todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
