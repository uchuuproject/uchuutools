.. _getting_started:

##################
Getting started
##################

*************
Installation
*************

*uchuutools* can be installed by either cloning the `repository`_ and installing it manually::

    $ git clone https://github.com/uchuuproject/uchuutools
    $ cd uchuutools
    $ python -m pip install -e .

or by installing it directly from `PyPI`_ with::

    $ python -m pip install uchuutools

*uchuutools* can now be imported as a package with :pycode:`import uchuutools`.

.. _repository: https://github.com/uchuuproject/uchuutools
.. _PyPI: https://pypi.org/project/uchuutools

***********************
Running the converters
***********************


Running the tree catalogue converter
=====================================
You can use the :download:`the convenience wrapper script <../../../uchuutools/scripts/convert_ctrees_to_h5>` to convert Consistent-Trees ascii tree catalogues into hdf5.  This script that allows to specify some of the most common options on the command-line::

    $ convert_ctrees_to_h5 -h
        usage: convert_ctrees_to_h5 [-h] [-p | -np] [-m | -s] <output directory> <output file prefix> <CTrees filenames> [<CTrees filenames> ...]

        Convert ascii Consistent-Trees files into hdf5 (optionally in MPI parallel)

        positional arguments:
        <output directory>    the output directory for the hdf5 file(s)
        <output file prefix>  the prefix to use for the output hdf file(s). The fully qualified output filename is <outputdir>/<output_fileprefix>_0.h5
        <CTrees filenames>    list of input (ascii) Consistent-Trees filenames (must provide at least one file to convert)

        optional arguments:
        -h, --help            show this help message and exit
        -p, --progressbar     display a progressbar on rank=0
        -np, --no-progressbar
                                disable the progressbar
        -m, --multiple-dsets  write a separate dataset for each halo property
        -s, --single-dset     write a single dataset containing all halo properties

        Usage Scenario 1:
        -----------------
        If you are converting the output of the standard Consistent-Trees code, then
        please provide the full-path to the 'forests.list' and 'locations.dat'(order is unimportant).

        Usage Scenario 2:
        -----------------
        If you are converting the output of the parallel Consistent-Trees code
        from the Uchuu collaboration, then please provide all the tree filenames
        that you would like to convert (i.e., files ending with '<prefix>.tree').
        The names for relevant 'forests.list (<prefix>.forest)' and
        'locations.dat (<prefix>.loc)' will be automatically constructed.

If you want to access the entire API for the :func:`uchuutools.converters.convert_ctrees_to_h5`, then you will have to write some python code:

.. code-block:: python

    >>> from uchuutools.converters import convert_ctrees_to_h5
    >>> help(convert_ctrees_to_h5)
        Help on function convert_ctrees_to_h5 in module uchuutools.converters.convert_ascii_ctrees_to_h5:

        convert_ctrees_to_h5(filenames, standard_consistent_trees=None, outputdir='./',
                             output_filebase='forest', write_halo_props_cont=True, fields=None,
                             drop_fields=None, truncate=True, compression='gzip',
                             buffersize=None, use_pread=True, max_nforests=None, comm=None,
                             show_progressbar=False)
            Convert a set of forests from Consistent Trees ascii file(s) into an
            (optionally compressed) hdf5 file. Can be invoked with MPI.


Running the halo catalogue converter
=====================================
To convert Rockstar or Consistent-Trees ascii halo catalogues into hdf5, you might want to use
:download:`the convenience wrapper script <../../../uchuutools/scripts/convert_halocat_to_h5>` that allows to specify
some of the most common options on the command-line::

    $ convert_halocat_to_h5 -h
        usage: convert_halocat_to_h5 [-h] [-p | -np] [-m | -s] <output directory> <halo catalogues> [<halo catalogues> ...]

        Convert ascii halo catalogs from Rockstar or Consistent-Trees into hdf5 (optionally in MPI parallel)

        positional arguments:
        <output directory>    the output directory for the hdf5 file(s)
        <halo catalogues>     list of input (ascii) halo catalogues

        optional arguments:
        -h, --help            show this help message and exit
        -p, --progressbar     display a progressbar on rank=0
        -np, --no-progressbar
                                disable the progressbar
        -m, --multiple-dsets  write a separate dataset for each halo property
        -s, --single-dset     write a single dataset containing all halo properties

However, if you want to access the entire API for the :func:`uchuutools.converters.convert_halocat_to_h5`, then you will have to write some python code:

.. code-block:: python

    >>> from uchuutools.converters import convert_halocat_to_h5
    >>> help(convert_halocat_to_h5)
        Help on function convert_halocat_to_h5 in module uchuutools.converters.convert_ascii_halocat_to_h5:

        convert_halocat_to_h5(filenames, outputdir='./', write_halo_props_cont=True,
                              fields=None, drop_fields=None, chunksize=100000,
                              compression='gzip', comm=None, show_progressbar=False)
            Converts a list of Rockstar/Consistent-Trees halo catalogues from
            ascii to hdf5.

            Can be used with MPI but requires that the number of files to be larger
            than the number of MPI tasks spawned.

