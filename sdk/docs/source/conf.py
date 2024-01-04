# Configuration file for the Sphinx documentation builder.


# We need to import the module to get __doc__ strings in runtime.
import aleo


# -- Project information

project = 'aleo'
copyright = '2023, AleoHQ'
author = 'kpp'

version = '0.2.0'

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'autoapi.extension',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None)
}

autoapi_dirs = ['./', '../../python/aleo']
autoapi_file_patterns = ['__init__.pyi']
autoapi_generate_api_docs = True
autoapi_add_toctree_entry = False
autoapi_keep_files = False

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

autodoc_typehints = "signature"

# We need to use autoapi because autodoc
# doesn' twork with stub files.
#
# However when we use stub files, autodoc doesn't
# know anything about __doc__ string from the .so library.
#
# So we need to monkey patch the autoapi objects to update
# its obj._docstring to aleo.so.__doc__.
def monkey_patch_documentation(app, what, name, obj, skip, options):
    def transform_docstring(doc):
        if doc:
            return doc.replace("\n", " ") + "\n"
        else:
            return ''

    if name.startswith("aleo"):
        if what == "module":
            doc = aleo.__doc__
            obj._docstring = transform_docstring(doc)
        if what == "class":
            (module, klass) = name.split(".")
            klass = getattr(aleo, klass)
            doc = klass.__doc__
            obj._docstring = transform_docstring(doc)
        if what == "method":
            (module, klass, method_name) = name.split(".")
            klass = getattr(aleo, klass)
            method = getattr(klass, method_name)
            if method_name == "from_bytes":
                obj.args = "bytes: bytes"
            doc = method.__doc__
            obj._docstring = transform_docstring(doc)
        if what == "property":
            (module, klass, property_name) = name.split(".")
            klass = getattr(aleo, klass)
            property = getattr(klass, property_name)
            doc = property.__doc__
            obj._docstring = transform_docstring(doc)
    return None

def setup(sphinx):
    sphinx.connect("autoapi-skip-member", monkey_patch_documentation)

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
