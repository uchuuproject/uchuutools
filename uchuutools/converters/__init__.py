# -*- coding: utf-8 -*-

"""
Converters
===========
Provides a collection of routines to convert from Rockstar and Consistent-Tree
ascii catalogues into hdf5 files

Recommended usage::

    import uchuutools.converters as utconv

Available submodules
--------------------
:func:`~convert_ctrees_to_h5`
    Converts ascii Consistent-Trees catalogues to hdf5

:func:`~convert_halocat_to_h5`
    Converts ascii Rockstar and Consistent-Trees halo catalogues to hdf5

"""


# %% IMPORTS
# Module imports
from . import convert_ascii_ctrees_to_h5
from . import convert_ascii_halocat_to_h5

from .convert_ascii_ctrees_to_h5 import *
from .convert_ascii_halocat_to_h5 import *


# All declaration
__all__ = []
__all__.extend(convert_ascii_ctrees_to_h5.__all__)
__all__.extend(convert_ascii_halocat_to_h5.__all__)
