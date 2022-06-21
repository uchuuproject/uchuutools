#!/usr/bin/env python
__author__ = "Manodeep Sinha"
__all__ = ["read_forest_from_forestidx", "read_forest_from_forestID",
           "get_idx_and_filename_from_forestID", "get_halo_dtype",
           "read_halocat"]

import warnings
import numpy as np
import h5py

from ..utils import generic_h5_file_opener


def _check_named_data_group(hf, named_data_group):
    """
    Internal utility function to check that the named data group exists
    """

    try:
        hf[named_data_group]
    except KeyError:
        msg = f"Error: Could not locate data group = '{named_data_group}'"\
              f"within the provided filename = {hf}. To fix this issue, "\
              f"please make sure the name of the data group is correct."
        raise ValueError(msg)

    return


def get_halo_dtype(hf, fields=None, named_data_group=None):
    """
    Returns the numpy datatype for reading the requested halo properties
    from the output hdf5 catalogues

    Parameters
    -----------

    hf: A valid ``h5py.File`` or ``h5py.Group`` object, required
        For tree catalogues, this file handle can refer to either
        the container file or an individual ``hdf5-treedata`` file.
        For halo catalogues, this file handle should refer to an
        ``hdf5-halocat`` file.
        Please refer to the :ref:`data format guide <data_formats>`
        for further details about the file formats.

    fields: an array of field names (strings), optional, default=None
        The default value is to return all available halo properties.
        A ValueError is raised if a passed field is not found in
        the hdf5 file. Repeated field names are only populated once.

    named_data_group: string, optional, default=None
        If set, then the code uses the provided group (or dataset)
        as the root group containing all the relevant data. This is useful
        when the relevant (halo/galaxy) data is stored in a non-standard
        location.

        If provided, this group/dataset *must* already exist within
        the hdf5 file. This provides the flexibility to read
        data from non-standard locations. For example, the
        halo properties are written to the root group in the initial
        Uchuu halo catalogues. Those halo catalogues can be read after
        setting ``named_data_group='/'``.

    Returns
    --------
    halo_dtype: A numpy datatype that has the requested fields
    """
    import h5py
    import numpy as np

    def _get_descr_with_fields(hf, base_dset, fields):

        if fields is None:
            raise ValueError("Please provide a list of fields to read")

        if not isinstance(fields, (tuple, list)):
            fields = (fields, )

        # Remove any repeated field names
        fields = set(fields)

        if isinstance(hf[base_dset], h5py.Group):
            try:
                descr = [(fld, hf[f'{base_dset}/{fld}'].dtype) for fld in fields
                         if isinstance(hf[f'{base_dset}/{fld}'], h5py.Dataset)]
            except KeyError as e:
                all_fields = [fld for fld in hf[base_dset].keys()
                              if isinstance(hf[f'{base_dset}/{fld}'], h5py.Dataset)]
                missing_fields = set(fields) - set(all_fields)
                msg = f"Error: Could not find field name(s) = {missing_fields}\n"\
                      f"Valid field names are = {all_fields}"
                raise ValueError(msg) from e

        elif isinstance(hf[base_dset], h5py.Dataset):
            try:
                descr = [(fld, hf[base_dset].dtype[fld]) for fld in fields]
            except KeyError as e:
                all_fields = hf[base_dset].dtype.names
                missing_fields = set(fields) - set(all_fields)
                msg = f"Error: Could not find field name(s) = {missing_fields}\n"\
                      f"Valid field names are = {all_fields}"
                raise ValueError(msg) from e
        else:
            msg = f"Error: Could not determine how to read from the dataset/group = {base_dset} "\
                  f"with type = {type(hf[base_dset])}. This should be either a group or a dataset."
            raise ValueError(msg)

        return np.dtype(descr)

    def _get_dtype_for_SOA(hf, base_dset, fields=None):
        # The halos were written as structure of arrays
        # -> need to get individual datatypes directly
        # from the datasets for those fields
        if not isinstance(hf[base_dset], h5py.Group):
            msg = f"Error: {base_dset} must be a group, but instead has type {type(hf[base_dset])})"
            raise ValueError(msg)

        if not fields:
            descr = [(ds, hf[f"{base_dset}/{ds}"].dtype) for ds in hf[base_dset].keys()
                     if isinstance(hf[f"{base_dset}/{ds}"], h5py.Dataset)]
            print(f"in _get_dtype_for_SOA: descr = {descr}\nfields = {fields}")
            return np.dtype(descr)

        return _get_descr_with_fields(hf, base_dset, fields)

    def _get_dtype_for_AOS(hf, base_dset, fields=None):
        # So the halos were written as a array of structures
        # -> need to get individual field datatypes from
        # the dataset
        if not isinstance(hf[base_dset], h5py.Dataset):
            msg = f"Error: {base_dset} must be a dataset"
            raise ValueError(msg)

        if not fields:
            # All the halo properties are requested -> we can directly
            # return the datatype for the dataset
            return hf[base_dset].dtype

        return _get_descr_with_fields(hf, base_dset, fields)


    # If the user has passed a specific group/dataset that
    # they want to use, then we only use that. This option
    # exists for backwards compatibility (first versions
    # of the Uchuu halo catalogues were written to the root
    # group `/`), as well as flexibility to allow users to
    # adapt the writing routines to their own naming conventions.
    # MS (22/02/2022)
    if named_data_group is not None:
        _check_named_data_group(hf, named_data_group)

        msg = f"A data group was provided = '{named_data_group}'.\n" \
              "The halo/galaxy properties are expected to have "\
              "been written to this data group"
        warnings.warn(msg)

        # MS (18/11/2021): This bothers me that setting the base dataset to '/'
        # results in a redundant '/' in the dataset name when using
        # formats of the type - {base_dset}/{fld} but letting this one go ...
        # (the extra '/' is ignored by h5py, and the correct dataset is used)
        base_dset = named_data_group

        # Now is this a group or a dataset
        if isinstance(hf[base_dset], h5py.Group):
            # This is a group -> get the keys within the group
            # as separate properties
            print("named data group is a group")
            return _get_dtype_for_SOA(hf, base_dset, fields=fields)
        elif isinstance(hf[base_dset], h5py.Dataset):
            print("named data group is a dataset")
            # This is a dataset -> get the dtype from the dataset
            return _get_dtype_for_AOS(hf, base_dset, fields=fields)
        else:
            msg = f"Error: {base_dset} with type {type(hf[base_dset])} "\
                  "is not a group or a dataset"
            raise ValueError(msg)


    # User has not specified a data group -> we need to figure out
    # where the data are located
    try:
        # Has the user passed in a file-handle to the container file?
        hf['File0']
    except KeyError:
        # if we are here, then we are working with an hdf5-treedata or an
        # hdf5-halocat file
        pass
    else:
        return get_halo_dtype(hf['File0'], fields=fields)

    # If we are here, then we are either working with an `hdf5-treedata`
    # file or an `hdf5-halocat` file. Let's set the dataset name
    base_dset = None
    possible_dsets = ['Forests', 'HaloCatalogue']
    for key in possible_dsets:
        try:
            hf[key]
        except KeyError:
            pass
        else:
            base_dset = key
            break

    if not base_dset:
        msg = "Error: Could not locate any of the "\
            f"possible datasets = {possible_dsets} "\
            "in the file. Please check that those datasets "\
            "are present in the file. For example, some of "\
            "the Uchuu halo catalogues do not contain the "\
            "``HaloCatalogue`` dataset - in which case you "\
            "can set the parameter ``named_data_group='/'`` "\
            "to read the halo (or galaxy) properties from an arbitrary "\
            "group (e.g., the group '/galaxies' within the hdf5 file),  "\
            "please provide the name of the group as the parameter "\
            "'named_data_group' (i.e., ``'named_data_group='/galaxies'``)"
        raise ValueError(msg)

    # Okay, we have the dataset name. Now, we need to figure out
    # how the halos were written and then dispatch to the appropriate
    # functions. There are two options:
    #   i) AOS -> array of structures, where all properties of one single
    #             halo are written together (under one single dataset)
    #  ii) SOA -> structure of arrays, each property is written as a
    #             separate dataset and this dataset contains the values
    #             for the specific halo property for all halos

    try:
        aos_base_dset = f'{base_dset}/halos'
        hf[aos_base_dset]
    except KeyError:
        return _get_dtype_for_SOA(hf, base_dset, fields=fields)
    else:
        return _get_dtype_for_AOS(hf, aos_base_dset, fields=fields)


def _get_fidx_from_forestID(h5_filehandle, track_forestID, rank=0):
    """
    Returns the index if a specific forest (identified by forestID)
    exists within a file, returns ``None`` otherwise.

    """
    def _get_fidx_from_dataset(dset, track_forestID, rank=0):
        idx = (np.where(dset == track_forestID))[0]
        if not idx:
            return None

        if len(idx) > 1:
            msg = f"[Rank {rank}]: Error: ForestID should be unique but "\
                  f"found {len(idx)} occurrences of "\
                  f"forestID = {track_forestID}"
            raise ValueError(msg)

        return idx[0]

    def _get_idx_from_fid_in_container_file(hf, track_forestID, rank=0):
        for ifile in range(hf.attrs['Nfiles']):
            idx = _get_fidx_from_dataset(hf[f'File{ifile}/Forest/ForestInfo'],
                                         track_forestID, rank)
            if idx:
                return idx

        return None

    with generic_h5_file_opener(h5_filehandle) as hf:
        try:
            hf['File0']
        except KeyError:
            # The file handle refers to an actual data file
            # (e.g., 'forest_0.h5') -> we can directly look within
            # the 'Forest/ForestInfo' dataset
            return _get_fidx_from_dataset(hf['Forest/ForestInfo'],
                                          track_forestID, rank)
        else:
            # So, we have the container file -> need to look through
            # all the subfiles and see if we can locate the forestID
            return _get_idx_from_fid_in_container_file(hf,
                                                       track_forestID,
                                                       rank)


def get_idx_and_filename_from_forestID(filenames, track_forestID, rank=0):
    """
    Returns the forest index and the filename for a specific forestID

    Parameters
    -----------

    filenames: list of filenames (string) or h5py.Group objects, required
        The list of hdf5 filenames or list of h5py.Group objects to
        search for the specific forest. If the container filename or a
        container file handle is passed, then all associated data files
        will be searched for the specified forest. Please refer to
        the :ref:`data format guide <data_formats>` for further details
        about the file formats.

    track_forestID: integer, required
        The ``forestID`` (originally generated by Consistent-Trees) that
        needs to be located

    rank: integer, optional, default=0
        An unique integer identifier for this task. Usually, the MPI
        rank of the task. Only used in debug messages at the moment.

    Returns
    --------

    (index, filename): A tuple containing the index for the forest and the
                       hdf5 filename within which the forest was located. If
                       the forest can not be found within all the provided
                       filenames, then a ``(None, None)`` is returned

    """
    if isinstance(filenames, str):
        filenames = (filenames, )

    for fname in filenames:
        idx = _get_fidx_from_forestID(fname, track_forestID, rank)
        if not idx:
            continue
        if isinstance(fname, h5py.Group):
            return (idx, fname.file.filename)
        else:
            return (idx, fname)

    return (None, None)


def read_forest_from_forestidx(h5_filehandle, forestindex, fields=None,
                               dtype=None, rank=0):
    """
    Returns a numpy structured array containing requested properties for
    all halos within a single forest

    Parameters
    -----------

    h5_filehandle: filename (string) or a valid ``h5py.File`` object, required
        The hdf5 filename or an open file handle containing the specified
        halos. The hdf5 file should be a valid Consistent-Trees hdf5
        catalogue similar to the files generated
        by `:func:uchuutools.convert_ctrees_to_h5`. External links embedded
        within the container file (e.g., ``hf['File0']`` where ``hf`` is
        an ``h5py.File`` handle for the container file) are also valid input

    forestindex: integer, required
        The index from which the forest is to be loaded. This index refers to
        the ``ForestInfo`` dataset.

    fields: list of column names (strings), optional, default=None
        The halo properties to be read. The default is to return all
        available halo properties.

    dtype: numpy datatype, optional, default=None
        The numpy datatype for the returned halos. By default, the datatype
        is constructed each time depending on the fields requested. But there
        are two use-cases where the datatype might need to be passed:

            i) when this function is being called repeatedly with the same
               ``fields``, then passing the datatype will likely result in
               faster load-times

            ii) the ``halos`` array has properties that will be populated
                externally by the user

        If a custom datatype is passed, then only the specified ``fields`` will
        have data. Any other columns that are present in the ``dtype``
        (i.e., those columns in ``dtype.names`` that are not in ``fields``)
        will have meaningless data (as created by ``np.empty``)

    rank: integer, optional
        The (MPI) rank for the process. Only used in debug messages

    Returns
    -------

    halos: numpy structured array
        A numpy structured array containing all the halos from the
        specified forest. Only the halo properties requested through
        the ``fields`` parameter are filled with meaningful data.

    """

    def _read_entire_forest(hf, start, nhalos, dtype, fields):
        halos = np.empty(nhalos, dtype=dtype)
        if hf.attrs['contiguous-halo-props']:
            for col in fields:
                dset_name = f'Forests/{col}'
                halos[col][:] = hf[dset_name][start:start+nhalos]
        else:
            dset_name = "Forests/halos"
            for col in fields:
                # The 'col' parameter must be the last one in the RHS
                # otherwise, the access is *extremely* slow
                halos[col][:] = hf[dset_name][start:start+nhalos][col]

        return halos

    with generic_h5_file_opener(h5_filehandle) as hf:
        if not dtype:
            dtype = get_halo_dtype(hf, fields)
        halo_start = hf['ForestInfo/ForestHalosOffset'][forestindex]
        nhalos = hf['ForestInfo/ForestNhalos'][forestindex]
        halos = _read_entire_forest(hf, halo_start, nhalos, dtype, fields)

    return halos


def read_forest_from_forestID(h5_filehandle, track_forestID,
                              fields=None, rank=0):
    """
    Returns a numpy structured array containing requested properties for
    all halos within a single forest

    Parameters
    -----------

    h5_filehandle: filename (string) or a valid ``h5py.File`` object, required
        The hdf5 filename or an open file handle containing the specified
        halos. The hdf5 file should be a valid Consistent-Trees hdf5
        catalogue similar to the files generated
        by `:func:uchuutools.convert_ctrees_to_h5`. External links embedded
        within the container file (e.g., ``hf['File0']`` where ``hf`` is
        an ``h5py.File`` handle for the container file) are also valid input

    track_forestID: integer, required
        The ``forestID`` (originally generated by Consistent-Trees) that
        needs to be loaded. All halos belonging to this forest will be read.

    fields: list of column names (strings), optional, default=None
        The halo properties to be read. The default is to return all
        available halo properties.

    rank: integer, optional
        The (MPI) rank for the process. Only used in debug messages

    Returns
    -------

    halos: numpy structured array
        A numpy structured array containing all the halos from the
        forest (with ``forestID==track_forestID``). Only the halo
        properties specified through the ``fields`` parameters are
        returned

        If the forestID can not be located in the supplied file,
        then a ValueError is raised.

    .. note:: This routine is meant to be used when arbitrary forests are
        being loaded. If you are looping sequentially through forests in the
        hdf5 tree data files, then it will be faster to directly invoke
        :func:`read_forest_from_forestidx`
    """

    with generic_h5_file_opener(h5_filehandle) as hf:
        idx, _ = get_idx_and_filename_from_forestID(hf, track_forestID, rank)
        if not idx:
            # Looks like the forestID does not exist in the file(s).
            msg = f"[Rank {rank}]: Error: Could not locate "\
                  f"forestID = {track_forestID} within the provided "\
                  f"filename = {h5_filehandle}. To fix this issue, try "\
                  f"passing the container file name (usually ``forest.h5``) "\
                  "and/or check that the forestID is correct."
            raise ValueError(msg)

        return read_forest_from_forestidx(hf, idx, fields, rank)


def read_halocat(h5_filehandle, fields=None, named_data_group=None, dtype=None,
                 chunksize=None,  rank=0):
    """
    Returns a numpy structured array containing requested properties for
    a chunk of halos within a single halo catalogue file.

    Parameters
    -----------

    h5_filehandle: filename (string) or a valid ``h5py.File`` object, required
        The hdf5 filename or an open file handle containing the specified
        halos. The hdf5 file should be a valid Consistent-Trees hdf5
        catalogue similar to the files generated
        by `:func:uchuutools.convert_ctrees_to_h5`. External links embedded
        within the container file (e.g., ``hf['File0']`` where ``hf`` is
        an ``h5py.File`` handle for the container file) are also valid input

    fields: list of column names (strings), optional, default=None
        The halo properties to be read. The default is to return all
        available halo properties.

    named_data_group: string, optional, default=None
        If set, then the code uses the provided group (or dataset)
        as the root group containing all the relevant data. This is useful
        when the relevant (halo/galaxy) data is stored in a non-standard
        location.

        If provided, this group/dataset *must* already exist within
        the hdf5 file

    dtype: numpy datatype, optional, default=None
        The numpy datatype for the returned halos. By default, the datatype
        is constructed each time depending on the fields requested. But there
        are two use-cases where the datatype might need to be passed:

            i) when this function is being called repeatedly with the same
               ``fields``, then passing the datatype will likely result in
               faster load-times

            ii) the ``halos`` array has properties that will be populated
                externally by the user

        If a custom datatype is passed, then only the specified ``fields`` will
        have data. Any other columns that are present in the ``dtype``
        (i.e., those columns in ``dtype.names`` that are not in ``fields``)
        will have meaningless data (as created by ``np.empty``)

    chunksize: integer, optional
        The number of halos returned at a time. The default value is
        to return 1% of the total number of halos present in the file

    rank: integer, optional
        The (MPI) rank for the process. Only used in debug messages

    Returns
    -------

    halos: numpy structured array
        A numpy structured array containing all the halos from the
        specified forest. Only the halo properties requested through
        the ``fields`` parameter are filled with meaningful data.

        This is a generator that will yield ``chunksize`` halos at a time.

    """
    def _read_SOA_halos(hf, base_dset, fields, start, nhalos, halos):
        for col in fields:
            dset_name = f'{base_dset}{col}' if base_dset.endswith("/") else f'{base_dset}/{col}'
            halos[col][:] = hf[dset_name][start:start+nhalos]
        yield halos

    def _read_AOS_halos(hf, dset_name, fields, start, nhalos, halos):
        for col in fields:
            # The 'col' parameter must be the last one in the RHS
            # otherwise, the access is *extremely* slow
            halos[col][:] = hf[dset_name][start:start+nhalos][col]
        yield halos

    def _read_halocat(hf, base_dset, start, nhalos, dtype, fields,
                      is_named_data_group=False, rank=rank):
        halos = np.empty(nhalos, dtype=dtype)
        if is_named_data_group:
            # Now is this a group or a dataset
            if isinstance(hf[base_dset], h5py.Group):
                # SOA case - read in all fields
                yield from _read_SOA_halos(hf, base_dset, fields, start, nhalos, halos)
            elif isinstance(hf[base_dset], h5py.Dataset):
                # AOS case
                yield from _read_AOS_halos(hf, base_dset, fields, start, nhalos, halos)
            else:
                msg = f"[Rank {rank}]Error: {base_dset} with type {type(hf[base_dset])} "\
                      "must be either a group or a dataset"
                raise ValueError(msg)


        # Not a named data group -> we need to figure out if this is a AOS/SOA
        try:
            hf[f'{base_dset}/halos']
        except KeyError:
            yield from _read_SOA_halos(hf, base_dset, fields, start, nhalos, halos)
        else:
            yield from _read_AOS_halos(hf, f"{base_dset}/halos", fields, start, nhalos, halos)

    base_dset = 'HaloCatalogue' if not named_data_group else named_data_group
    if dtype and not fields:
        msg = f"[Rank {rank}]: Error: If you pass a custom datatype, you must also pass "\
              f"the fields to be read. Valid field names (in the custom data-type) are: {dtype.names}"
        raise ValueError(msg)

    with generic_h5_file_opener(h5_filehandle) as hf:
        if named_data_group is not None:
            try:
                hf[named_data_group]
            except KeyError:
                msg = f"[Rank {rank}]: Error: Could not locate "\
                    f"named_data_group = '{named_data_group}' within the "\
                    f"provided filename = {h5_filehandle}.\n To fix this issue, "\
                    f"please make sure the name of the data group is correct."
                raise ValueError(msg)

        if not dtype:
            dtype = get_halo_dtype(hf, fields=fields, named_data_group=named_data_group)

        if not fields:
            fields = list(dtype.names)

        # This is a per-file attribute that shows the total number of halos in
        # the halo catalogue. Only noting because there is a similarly named
        # attributed in the mergertree data container file that refers to
        # the total number of halos *across ALL files*.  MS 24/02/2022
        totnhalos = hf.attrs['TotNhalos']
        # print(f"totnhalos = {totnhalos} fields = {fields} named_data_group = {named_data_group}")
        # print(f"dtype = {dtype}")

        if totnhalos <= 0:
            msg = f"[Rank {rank}]: Error: Total number of halos "\
                  f"must be at least 1, found = {totnhalos} halos "\
                  f"within the provided filename = {h5_filehandle}. "\
                  "Perhaps, the file got corrupted or was not written "\
                  "out fully?"
            raise ValueError(msg)

        if not chunksize:
            chunksize = int(0.01 * totnhalos) if totnhalos > 100 else 10

        # print(f"chunksize = {chunksize}")
        is_named_data_group = named_data_group is not None
        # print(f"is_named_data_group = {is_named_data_group} named_data_group = {named_data_group}")
        for halo_start in range(0, totnhalos, chunksize):
            nhalos = chunksize if (halo_start + chunksize) < totnhalos \
                               else (totnhalos - halo_start)
            yield from _read_halocat(hf, base_dset, halo_start, nhalos,
                                     dtype, fields,
                                     is_named_data_group=is_named_data_group,
                                     rank=rank)

