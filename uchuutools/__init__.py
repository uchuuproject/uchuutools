# -*- coding: utf-8 -*-

"""
uchuutools
===========
Provides a collection of utility functions.
Recommended usage::

    import uchuutools as ut

Available modules
-----------------
:mod:`~converters`
    Provides a collection of functions that are core to **uchuutools** and are
    imported automatically.

:mod:`~ctrees_utils`
    Provides a collection of functions useful in manipulating *Consistent-Trees* catalogues.

:mod:`~utils`
    Provides several useful utility functions.

"""


# %% IMPORTS AND DECLARATIONS
# uchuutools imports
from .__version__ import __version__
from . import converters, ctrees_utils, utils
from .converters import *
from .readers import *

# All declaration
__all__ = ['converters', 'readers', 'ctrees_utils', 'utils', ]
__all__.extend(converters.__all__)
__all__.extend(readers.__all__)

# Author declaration
__author__ = "Manodeep Sinha (@manodeep)"
