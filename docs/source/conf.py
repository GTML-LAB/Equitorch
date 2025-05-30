# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys, os
sys.path.insert(0,'../')
sys.path.insert(0,os.path.abspath('../'))
sys.path.insert(0,os.path.abspath('../..'))

import equitorch

project = 'equitorch'
copyright = '2025, Tong Wang'
author = 'Tong Wang'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
]

autoapi_options = [
    "members",
    # "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

# autodoc_default_options = {
#     'member-order': 'bysource',
#     'special-members': '__init__',
#     'undoc-members': False,
#     'private-members': False,
#     'inherited-members': False,
#     'show-inheritance': False,

# }

autoapi_member_order = 'bysource'

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_attr_annotations = True

autodoc_default_options = {
    'member-order': 'bysource',
    'undoc-members': False,
    'no_index': True,
    'show-inheritance': False,
    'inherited-members': False,
    'exclude-members': '__init__, extra_repr, forward',
    # 'imported-members': False,
    # 'members': False,
 }

# templates borrowed from https://github.com/pyg-team/pytorch_geometric/blob/master/docs/source/_templates/
templates_path = ['_templates']
exclude_patterns = []


# autosummary_generate = True
# autosummary_imported_members = True
autosummary_ignore_module_all = False
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo_.png'
html_css_files = [
    'css/custom.css'
]

def rstjinja(app, docname, source):
    r"""
    Render our pages as a jinja template for fancy templating goodness.
    """
    # Make sure we're outputting HTML
    # if app.builder.format != 'html':
        # return
    src = source[0]
    rst_context = {'equitorch': equitorch}
    rendered = app.builder.templates.render_string(
        src, rst_context | app.config.html_context
    )
    source[0] = rendered

def setup(app):
    app.connect("source-read", rstjinja)