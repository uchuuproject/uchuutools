# -*- coding: utf-8 -*-

"""
General Utilities
==================
Provides several utility functions.
"""

__author__ = "Manodeep Sinha"
__all__ = ["get_parser", "get_approx_totnumhalos", "generic_reader",
           "get_metadata", "resize_halo_datasets",
           "check_and_decompress", "distribute_array_over_ntasks",
           "write_halos", "update_container_h5_file", ]


# Majority of the code below is taken from
# Yao-Yuan Mao's SimulationAnalysis.py (part of YYM's helper package)
import re
import numpy as np
from builtins import range, zip
from contextlib import contextmanager


def sanitize_ctrees_header(headerline):
    import re

    header = [re.sub('\(\d+\)$', '', s) for s in headerline]
    # print("After normal sub: header = {}\n".format(header))
    header = [re.sub('[^a-zA-Z0-9 \n\.]', '_', s) for s in header]
    # print(f"After replacing special characters with _: header = {header}\n")
    header = [re.sub('_$', '', s) for s in header]
    # print(f"After replacing trailing underscore: header = {header}\n")
    header = [re.sub('(_)+', '_', s) for s in header]
    # print(f"After replacing multiple underscores: header = {header}")
    return header


def _isstring(s):
    try:
        s + ''
    except TypeError:
        return False
    return True


class BaseParseFields():
    def __init__(self, header, fields=None):
        if len(header) == 0:
            if all(isinstance(f, int) for f in fields):
                self._usecols = fields
                self._formats = [float]*len(fields)
                self._names = ['f%d' % f for f in fields]
            else:
                raise ValueError("header is empty, so fields must be "
                                 "a list of int.")
        else:
            # print(f"in baseparsefields: header = {header}")
            header_s = [self._name_strip(__) for __ in header]
            # print(f"header_s = {header_s}")
            if not fields:
                self._names = header
                names_s = header_s
                self._usecols = list(range(len(names_s)))
            else:
                # print(f"fields = {fields}")
                if _isstring(fields):
                    fields = [fields]
                self._names = [header[f] if isinstance(f, int)
                               else str(f) for f in fields]
                names_s = [self._name_strip(__) for __ in self._names]
                wrong_fields = [str(f) for s, f in zip(names_s, fields)
                                if s not in header_s]
                if wrong_fields:
                    msg = f"The following field(s) are not available "\
                          f"{', '.join(wrong_fields)}.\nAvailable fields: "\
                          f"{', '.join(header)}"
                    raise ValueError(msg)
                self._usecols = [header_s.index(__) for __ in names_s]

            self._formats = [self._get_format(__) for __ in names_s]
            self.dtype = np.dtype({'names': self._names,
                                   'formats': self._formats})

    def parse_line(self, line):
        items = line.split()
        try:
            return tuple(c(items[i]) for i, c in zip(self._usecols,
                                                     self._formats))
        except Exception as _error:
            msg = f'Something wrong when parsing this line:\n{line}'
            raise _error(msg)

    def pack(self, X):
        return np.array(X, self.dtype)

    def _name_strip(self, s):
        return self._re_name_strip.sub('', s).lower()

    def _get_format(self, s):
        return float if self._re_formats.search(s) is None else int

    _re_name_strip = re.compile('\W|_')
    _re_formats = re.compile('^phantom$|^mmp$|id$|^num|num$')


def get_parser(filename, fields=None, drop_fields=None):
    """
    Returns a parser that parses a single line from the input ascii file

    Parameters
    ----------

    filename: string, required
        A filename containg Rockstar/Consistent-Trees data. Can be a compressed
        file if the compression is one of the types supported by the
        ``generic_reader`` function.

    fields: list of strings, optional, default: None
        Describes which specific columns in the input file to carry across
        to the hdf5 file. Default action is to convert ALL columns.

    drop_fields: list of strings, optional, default: None
        Describes which columns are not carried through to the hdf5 file.
        Processed after ``fields``, i.e., you can specify ``fields=None`` to
        create an initial list of *all* columns in the ascii file, and then
        specify ``drop_fields = [colname2, colname7, ...]``, and those columns
        will not be present in the hdf5 output.

    Returns
    -------
    parser: an instance of BaseParseFields
        A parser that will parse a single line (read from a
        Rockstar/Consistent-Trees file) and create a tuple
        containing *only* the relevant columns.

    """

    with generic_reader(filename, 'rt') as f:
        # Taken from SimulationAnalysis.py
        hdrline = next(f)
        hdrline = hdrline.strip().lstrip('#').lstrip()
        # print(f"hdrline = {hdrline}")
        split_headerline = [s for s in hdrline.split()]
        header = sanitize_ctrees_header(split_headerline)
        # print(f"sanitised header = {header}")

        keep_fields = header[:]
        if fields:
            print(f"Only keeping the specified Fields = {fields}")
            fields = [s.upper() for s in fields]
            for fld in split_headerline:
                upfld = fld.upper()
                if upfld not in fields:
                    keep_fields.remove(fld)
                    print(f"Removing fld = {fld} since it does not exist "
                          "in 'fields'")
                else:
                    print(f"Keeping fld = {fld}")

        if drop_fields:
            for fld in drop_fields:
                try:
                    keep_fields.remove(fld)
                except ValueError:
                    msg = f"Field = '{fld}' was specified in the list of "\
                          f"fields to be dropped but could not find in the "\
                          f"list of current fields = {keep_fields}. Perhaps "\
                          f"the case needs to be fixed (the matching is "\
                          f"case-sensitive)?"
                    raise ValueError(msg)
        p = BaseParseFields(header, keep_fields)
        # Above taken from SimulationAnalysis.py

    return p


@contextmanager
def generic_reader(filename, mode='rt'):
    """
    Returns a file-reader with capability to read line-by-line
    for both compressed and normal text files.

    Parameters
    -----------

    filename: string, required
        The filename for the associated input/output. Can be a
        compressed (.bz2, .gz, .xz, .zip) file as well as a regular
        ascii file

    mode: string, optional, default: 'rt' (readonly-text mode)
        Controls the kind of i/o operation that will be
        performed

    Returns
    --------

    f: file handle, generator
        Yields a generator that has the ``readline`` feature (i.e.,
        supports the paradigm ``for line in f:``). This file-reader
        generator is suitable for use in ``with`` statements, e.g.,
        ``with generic_reader(<fname>) as f:``

    """
    if not isinstance(filename, str) and hasattr(filename, 'decode'):
        filename = filename.decode()

    if filename.endswith(".bz2"):
        import bz2
        f = bz2.open(filename, mode=mode)
    elif filename.endswith(".gz"):
        import gzip
        f = gzip.open(filename, mode=mode)
    elif filename.endswith('.xz'):
        import lzma
        f = lzma.open(filename, mode=mode)
    elif filename.endswith('.zip'):
        validmodes = set(['r', 'w'])
        if mode not in validmodes:
            msg = f"Error: filename = '{filename}' is a zipfile but the  "\
                  f"requested mode = '{mode}' is not the valid zipfile  "\
                  f"modes = '{validmodes}'"
            raise ValueError(msg)

        import zipfile
        f = zipfile.open(filename, mode)
    else:
        import io
        f = io.open(filename, mode=mode)

    yield f

    f.close()


def get_metadata(input_file):
    """
    Returns metadata information for ``input_file``. Includes all
    comment lines in the header, Rockstar/Consistent-Trees version,
    and the input catalog type (either Rockstar or Consistent-Trees).

    Assumes that the only comment lines in the file occur
    at the beginning. Comment lines are assumed to begin with '#'.

    Parameters
    -----------

    input_file: string, required
        The input filename for the Rockstar/Consistent Trees file
        Compressed files ('.bz2', '.gz', '.xz', '.zip') are also allowed
        as valid kinds of ``input_file``

    Returns
    --------

    metadata_dict: dictionary
        The dictionary contains four key-value pairs corresponding to the
        keys: ['metadata', 'version', 'catalog_type', 'headerline'].

    metadata: string
        All lines in the beginning of the file that start with the
        character '#'.

    version: string
        Rockstar or Consistent-Trees version that was used to generate
        ``input_file``

    catalog_type: string
        Is one of [``Rockstar``, ``Consistent Trees``,
        ``Consistent Trees (hlist)``] and indicates what kind of
        catalog is contained in ``input_file``

    headerline: string
        The first line in the input file with any leading/trailing white-space,
        and any leading '#' removed

    """

    def get_catalog_and_version(line):
        input_catalog_type = None
        version = None
        """
        The rockstar version has the pattern
        '#Rockstar Version: <VERSION STRING>'
        We split the input line on ':', then take the second chunk by
        indexing with [1] to retrieve the version string. The final step
        removes all white space from the string.

        The Consistent Trees version has the pattern
        '#Consistent Trees Version <VERION_STRING>'
        We split the line on ' ' to get four chunks, and take the last one.
        Remove the white-space as usual
        """
        catalog_types = [('Rockstar',
                          lambda line:(line.split(":")[1]).strip()),
                         ('Consistent Trees',
                          lambda line:(line.split(" ")[-1]).strip())]
        for (catalog, fn) in catalog_types:
            if catalog in line:
                input_catalog_type = catalog
                version = fn(line)
                break

        return input_catalog_type, version

    metadata = []
    ctrees_hlist = True
    with generic_reader(input_file, 'rt') as f:
        for line in f:
            if not line.startswith('#'):
                # Check if the first 'data' line contains
                # multiple columns. This is required to distinguish
                # between CTrees-generated halo catalogues ('hlist' files)
                # and the CTrees-generated tree files ('tree_?_?_?.dat' files)
                if len(line.split()) == 1:
                    ctrees_hlist = False

                break

            if 'VERSION' in line.upper():
                input_catalog_type, version = get_catalog_and_version(line)

            metadata.append(line)

    if not version:
        msg = f"Error: Could not locate version in the input "\
              f"file = '{input_file}'"
        raise ValueError(msg)

    if ctrees_hlist and 'Consistent' in input_catalog_type:
        input_catalog_type = f"{input_catalog_type} (hlist)"

    hdrline = metadata[0]
    hdrline = hdrline.strip().lstrip('#').lstrip()

    simulation_params = get_simulation_params_from_metadata(metadata)

    metadata_dict = dict()
    metadata_dict['metadata'] = metadata
    metadata_dict['version'] = version
    metadata_dict['catalog_type'] = input_catalog_type
    metadata_dict['headerline'] = hdrline
    metadata_dict['simulation_params'] = simulation_params

    return metadata_dict


def get_simulation_params_from_metadata(metadata):
    simulation_params = dict()
    for line in metadata[1:]:
        # cosmological parameters
        if "Omega_M" in line or ("Om" in line and "Ol" in line):
            pars = line[1:].split(";")
            for j, par in enumerate(["Omega_M",
                                     "Omega_L",
                                     "hubble"]):
                v = float(pars[j].split(" = ")[1])
                simulation_params[par] = v

        # box size
        elif "Full box size" in line or "Box size" in line:
            pars = line.split("=")[1].strip().split()
            box = float(pars[0])
            simulation_params['Boxsize'] = box

            # We can break because boxsize always occurs later than
            # the cosmological parameters (regardless of
            # CTrees/Rockstar origin)
            break

    return simulation_params


def get_approx_totnumhalos(input_file, ndatabytes=None):
    """
    Returns an (approximate) number of lines containing data
    in the ``input_file``.

    Assumes that the only comment lines in the file occur
    at the beginning. Comment lines are assumed to begin with '#'.

    Parameters
    -----------

    input_file: string, required
        The input filename for the Rockstar/Consistent Trees file

    ndatabytes: integer, optional
        The total number of bytes being processed. If not passed, the
        entire disk size of the ``input_file`` minus the initial header
        lines will be used (i.e. assumes that the entire file is being
        processed)

    Returns
    --------

    approx_totnumhalos: integer
        The approximate number of halos in the input file. The actual
        number of halos should be close but can be smaller/greater than
        the approximate value.

    """
    import os

    with generic_reader(input_file, 'rt') as f:
        hdr_bytes = 0
        for line in f:
            if line.startswith('#'):
                hdr_bytes += len(line)
            else:
                # The first line of a CTrees file is the number of trees
                # in that file. We should skip that line
                if len(line.split()) == 1:
                    hdr_bytes += len(line)
                    continue

                data_line_len = len(line)
                break

    if not ndatabytes:
        statinfo = os.stat(input_file)
        totbytes = statinfo.st_size
        ndatabytes = totbytes - hdr_bytes

    approx_nhalos = (ndatabytes // data_line_len) + 1
    return approx_nhalos


def check_and_decompress(fname):
    """
    Decompresses the input file (if necessary) and returns the
    decompressed filename

    Parameters
    ----------

    fname: string, required
        Input filename, can be compressed

    Returns
    -------

    decomp_fname: string
        The decompressed filename

    """

    decompressors = [('.bz2', 'bunzip2'),
                     ('.gz', 'gunzip'),
                     ('.zip', 'unzip')]
    if not isinstance(fname, str) and hasattr(fname, 'decode'):
        fname = fname.decode()

    for (ext, decomp) in decompressors:
        if fname.endswith(ext):
            import subprocess
            print(f"Uncompressing compressed file '{fname}'...")
            subprocess.run([decomp, fname], check=True)
            print(f"Uncompressing compressed file '{fname}'...done")
            fname = fname.rstrip(ext)
            break

    return fname


def distribute_array_over_ntasks(cost_array, rank, ntasks):
    """
    Calculates the subscript range for the ``rank``'th task
    such that the work-load is evenly distributed across
    ``ntasks``.

    Parameters
    -----------

    cost_array: numpy array, required
        Contains the cost associated with processing *each* element
        of the array

    rank: integer, required
        The integer rank for the task that we need to compute the
        work-load division for

    ntasks: integer, required
        Total number of tasks that the array should be (evenly) distributed
        across

    Returns
    --------

    (start, stop): A tuple of (np.int64, np.int64)
        Contains the initial and final subscripts that the ``rank``
        task should process.

        Note: start, stop are both inclusive, i.e., all elements from ``start``
        to ``stop`` should be included. For python array indexing with slices,
        this translates to arr[start:stop + 1].
    """

    if rank >= ntasks or rank < 0 or ntasks < 1:
        msg = f"Error: rank = {rank} and NTasks = {ntasks} must satisfy "\
               "   i) ThisTask < NTasks, "\
               "  ii) ThisTask > 0, and "\
               " iii) NTasks >= 1"
        raise ValueError(msg)

    ncosts = cost_array.shape[0]
    if ncosts < 0:
        msg = f"Error: On rank = {rank} total number of elements = {ncosts} "\
               "must be >= 0"
        raise ValueError(msg)

    if ncosts == 0:
        print(f"Warning: On rank = {rank}, got 0 elements to distribute "
              f"...returning")
        start, stop = None, None
        return (start, stop)

    if ntasks == 1:
        start, stop = 0, ncosts - 1
        print(f"Only a single task -- distributing the entire array "
              f"[start, stop] = [{start}, {stop}]")
        return (start, stop)

    cumul_cost_array = cost_array.cumsum()
    total_cost = cumul_cost_array[-1]
    target_cost = total_cost / ntasks

    cost_remaining, cost_assigned, ntasks_left = total_cost, 0.0, ntasks
    start = 0
    for icore in range(rank + 1):
        full_array_target_cost = cost_assigned + target_cost
        if icore == (ntasks - 1):
            stop = ncosts - 1
        else:
            stop = min(np.where(cumul_cost_array >= full_array_target_cost)[0])

        if stop < ncosts:
            cost_assigned_this_rank = cumul_cost_array[stop] - cost_assigned

            # Only print the work divisions on the last rank
            if rank == (ntasks - 1):
                # Yes, this is intentional that `icore` is used. The intent is
                # to show which forests ranges are assigned to which task
                # (i.e., rank) - MS 22/06/2020
                print(f"[Rank={icore}]: Assigning forests: start, stop = "
                      f"[{start}, {stop}]. Target cost = {target_cost}, "
                      f"cost actually assigned = {cost_assigned_this_rank}")
            if icore == rank:
                break

            cost_remaining -= cost_assigned_this_rank
            cost_assigned += cost_assigned_this_rank
            ntasks_left -= 1
            target_cost = cost_remaining / ntasks_left
            start = stop + 1
        else:
            msg = "Error: While determining optimal distribution of forests "\
                  "across MPI tasks. Could not locate an appropriate "\
                  f"stopping point for rank = {rank}. Expected stop = {stop} "\
                   "to be less than the number of elements in the cost "\
                  f"array = {ncosts}"
            raise ValueError(msg)

    return (start, stop)


def resize_halo_datasets(halos_dset, new_size, write_halo_props_cont, dtype):
    """
    Resizes the halo datasets

    Parameters
    -----------

    halos_dset: dictionary, required

    new_size: scalar integer, required

    write_halo_props_cont: boolean, required
        Controls if the individual halo properties are written as distinct
        datasets such that any given property for ALL halos is written
        contiguously (structure of arrays, SOA).

    dtype: numpy datatype


    Returns
    -------

        Returns ``True`` on successful completion
    """
    if write_halo_props_cont:
        for name in dtype.names:
            dset = halos_dset[name]
            dset.resize((new_size, ))
    else:
        halos_dset.resize((new_size, ))

    return True


def write_halos(halos_dset, halos_dset_offset, halos, nhalos_to_write,
                write_halo_props_cont):
    """
    Writes halos into the relevant dataset(s) within a hdf5 file

    Parameters
    -----------

    halos_dset: dictionary, required
        Contains the halos dataset(s) within a hdf5 file where either
        the entire halos array or the individual halo properties should
        be written to. See parameter ``write_halo_props_cont`` for
        further details

    halos_dset_offset: scalar integer, required
        Contains the index within the halos dataset(s) where the write
        should start

    halos: numpy structured array, required
        An array containing the halo properties that should be written out
        into the hdf5 file. The entire array may not be written out, see
        the parameter ``nhalos_to_write``

    nhalos_to_write: scalar integer, required
        Number of halos from the ``halos`` array that should be written
        out. Can be smaller than the shape of the ``halos`` array

    write_halo_props_cont: boolean, required
        Controls if the individual halo properties are written as distinct
        datasets such that any given property for ALL halos is written
        contiguously (structure of arrays, SOA).

    Returns
    -------

    Returns ``True`` on successful completion of the write

    """

    if write_halo_props_cont:
        for name in halos.dtype.names:
            dset = halos_dset[name]
            dset[halos_dset_offset:halos_dset_offset + nhalos_to_write] \
                = halos[name][0:nhalos_to_write]
    else:
        halos_dset[halos_dset_offset:halos_dset_offset + nhalos_to_write] = \
            halos[0:nhalos_to_write]

    return True


def update_container_h5_file(fname, h5files,
                             standard_consistent_trees=True):
    """
    Writes the container hdf5 file that has external links to
    the hdf5 datafiles with the mergertree information.

    Parameters
    -----------

    fname: string, required
        The name of the output container file (usually ``forest.h5``). A
        new file is always created, however, if the file ``fname`` previously
        existed then the external links are preserved.

    h5files: list of filenames, required
        The list of filenames that were either newly created or updated.

        If the container file ``fname`` exists, then the union of the filenames
        that already existed in ``fname`` and ``h5files`` will be used to
        create the external links

    standard_consistent_tree: boolean, optional, default: True
        Specifies whether the input files were from a parallel
        Consistent-Trees code or the standard Consistent-Trees code. Assumed
        to be standard (i.e., the public version) of the Consistent-Trees
        catalog

    Returns
    -------

    Returns ``True`` on successful completion of the write

    """
    import h5py

    outfiles = h5files
    if not isinstance(h5files, (list, tuple)):
        outfiles = [h5files]

    try:
        with h5py.File(fname, 'r') as hf:
            nfiles = hf['/'].attrs
            for ifile in range(nfiles):
                outfiles.append(hf[f'File{ifile}'].file)
                print(f"outfiles = {outfiles}")
    except OSError:
        pass

    outfiles = set(outfiles)
    nfiles = len(outfiles)
    with h5py.File(fname, 'w') as hf:
        hf['/'].attrs['Nfiles'] = nfiles
        hf['/'].attrs['TotNforests'] = 0
        hf['/'].attrs['TotNtrees'] = 0
        hf['/'].attrs['TotNhalos'] = 0
        attr_props = [('TotNforests', 'Nforests'),
                      ('TotNtrees', 'Ntrees'),
                      ('TotNhalos', 'Nhalos')]
        for ifile, outfile in enumerate(outfiles):
            with h5py.File(outfile, 'a') as hf_task:
                if standard_consistent_trees:
                    hf_task.attrs['consistent-trees-type'] = 'standard'
                else:
                    hf_task.attrs['consistent-trees-type'] = 'parallel'
                for (out, inp) in attr_props:
                    hf['/'].attrs[out] += hf_task['/'].attrs[inp]

            hf[f'File{ifile}'] = h5py.ExternalLink(outfile, '/')
    return
