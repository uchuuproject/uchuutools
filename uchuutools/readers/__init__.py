# -*- coding: utf-8 -*-

"""
Readers
========
Provides a collection of routines to read Consistent-Tree
hdf5 files created with ``uchuutools``

Recommended usage::

    import uchuutools.readers as utread

Available submodules
--------------------
:func:`~read_forest_from_forestidx`
    Reads a single forest (with a known forest index) from an
    HDF5 Consistent-Trees catalogues

:func:`~read_forest_from_forestID`
    Reads a single forest (with a known forest ID) from an HDF5
    Consistent-Trees catalogues

:func:`~read_halocat`
    Reads a single halo catalogue file from an HDF5 file containing
    halos generated by Rockstar or Consistent-Trees

:func:`~get_idx_and_filename_from_forestID`
    Searches through a list of filenames to locate a forest ID and returns
    the matching forest index and filename in which the forest ID was found

:func:`~get_halo_dtype`
    Gets the numpy compound datatype for requested halo properties

"""


# %% IMPORTS
# Module imports
from . import read_uchuu_catalogs

from .read_uchuu_catalogs import *


# All declaration
__all__ = []
__all__.extend(read_uchuu_catalogs.__all__)