#!/usr/bin/env python
from __future__ import print_function

__author__ = "Manodeep Sinha"
__all__ = ["convert_ctrees_to_h5"]

import os
import time
import io

from ..utils import get_metadata, get_parser, \
                    distribute_array_over_ntasks, get_approx_totnumhalos, \
                    check_and_decompress, resize_halo_datasets, write_halos, \
                    update_container_h5_file

from ..ctrees_utils import read_locations_and_forests, \
                           get_aggregate_forest_info,\
                           get_all_parallel_ctrees_filenames, \
                           validate_inputs_are_ctrees_files, \
                           check_forests_locations_filenames, \
                           get_treewalk_dtype_descr, add_tree_walk_indices


def _create_and_validate_halos_dset(hf, dtype, write_halo_props_cont=True):
    """
    Internal utility function to check the existing halo dataset in the file,
    and return a reference where the halos dataset can be written to.

    """

    forests_grp = hf['Forests']
    if write_halo_props_cont:
        halos_dset = dict()
        # Create a dataset for every halo property
        # For any given halo property, the value
        # for halos will be written contiguously
        # (structure of arrays)
        for name in dtype.names:
            halos_dset[name] = forests_grp[name]
            if hf.attrs['Nhalos'] != halos_dset[name].shape[0]:
                msg = f"Error: The dataset for halo property = '{name}' "\
                    f"does not contain *exactly* the same number of halos "\
                    f"as specified in the file attribute. "\
                    "shape of halo property dataset "\
                    f" = '{halos_dset[name].shape}' "\
                    f"nhalos in file attribute = {hf.attrs['Nhalos']}"
                raise AssertionError(msg)
    else:
        # Create a single dataset that contains all properties
        # of a given halo, then all properties of the next halo,
        # and so on (array of structures)
        halos_dset = forests_grp['halos']
        if hf.attrs['Nhalos'] != halos_dset.shape[0]:
            msg = f"Error: The halos dataset does not contain *exactly* the "\
                  f"same number of halos as specified in the file attribute. "\
                  f"shape of halo property dataset = '{halos_dset.shape}' "\
                  f"nhalos in file attribute = {hf.attrs['Nhalos']}"
            raise AssertionError(msg)

    return halos_dset


def _convert_ctrees_forest_range(forest_info, trees_and_locations, rank,
                                 outputdir="./", output_filebase="forest",
                                 write_halo_props_cont=True,
                                 fields=None, drop_fields=None,
                                 truncate=True, compression='gzip',
                                 buffersize=1024*1024, use_pread=True,
                                 show_progressbar=False):
    """
    Convert a set of forests from Consistent Trees ascii file(s) into an
    (optionally compressed) hdf5 file.

    Parameters
    -----------

    forest_info: numpy structured array, required, shape: (nforests, )
        The numpy structured array containing the following info
        <ForestID ForestNhalos Input_ForestNbytes Ntrees> for the
        *forests* that are to be converted by this task

    trees_and_locations: numpy structured array, required, shape: (ntrees, )
        The numpy structured array containing the following info
        <ForestID TreeRootID FileName Offset TreeNbytes> for the
        *trees* that are to be converted by this task

    rank: integer, required
        The (MPI) rank for the process. The output filename is determined
        with this rank to ensure unique filenames when running in parallel.

    outputdir: string, optional, default: current working directory ('./')
        The directory where the converted hdf5 file will be written in. The
        output filename is obtained by appending '.h5' to the ``input_file``.

    output_filebase: string, optional, default: "forest"
        The output filename is constructed using
        '<outputdir>/<output_filebase>_<rank>.h5'

    write_halo_props_cont: boolean, optional, default: True
        Controls if the individual halo properties are written as distinct
        datasets such that any given property for *all* halos is written
        contiguously (structure of arraysA).

        When set to False, only one dataset ('halos') is created under the
        group 'Forests', and *all* properties of a halo is written out
        contiguously (array of structures).

    fields: list of strings, optional, default: None
        Describes which specific columns in the input file to carry across
        to the hdf5 file. Default action is to convert ALL columns.

    drop_fields: list of strings, optional, default: None
        Describes which columns are not carried through to the hdf5 file.
        Processed after ``fields``, i.e., you can specify ``fields=None`` to
        create an initial list of *all* columns in the ascii file, and then
        specify ``drop_fields = [colname2, colname7, ...]``, and those columns
        will not be present in the hdf5 output.

    truncate: boolean, default: True
        Controls whether a new file is created on this 'rank'. When set to
        ``True``, the header info file is written out. Otherwise, the file
        is appended to.

    compression: string, optional, default: 'gzip'
        Controls the kind of compression applied. Valid options are anything
        that ``h5py`` accepts.

    buffersize: integer, optional, default: 1 MB
        Controls the size of the buffer for how many halos are written out
        per write call to the hdf5 file. The number of halos written out is
        this buffersize divided the size of the datatype for individual halos.

    use_pread: boolean, optional, default: True
        Controls whether low-level i/o operations (through ``os.pread``) is
        used. Otherwise, higher-level i/o operations (via ``io.open``) is
        used. This option is only meaningful on linux systems (and python3+).
        Since ``pread`` does not change the file offset, additional
        parallelisation can be implemented reasonably easily.

    show_progressbar: boolean, optional, default: False
        Controls whether a progressbar is printed. Only enables progressbar
        on rank==0, the remaining ranks ignore this keyword.


    Returns
    -------

        Returns ``True`` on successful completion.

    """

    import numpy as np
    import h5py
    import sys
    from tqdm import tqdm

    if rank != 0:
        show_progressbar = False

    try:
        os.pread
    except NameError:
        use_pread = False

    sys.stdout.flush()
    tstart = time.perf_counter()

    # Set the datalen for strings
    string_dtype = 'S1024'

    if not os.path.isdir(outputdir):
        msg = f"Error: The first parameter (output directory) = "\
              f"'{outputdir}' should be of type directory"
        raise ValueError(msg)

    ntrees = trees_and_locations.shape[0]
    nforests = forest_info.shape[0]
    if nforests > ntrees:
        msg = f"Error: Expected the number of trees = '{ntrees}' "\
               "to be *at most* equal to the number of "\
              f"forests = '{nforests}'"
        raise AssertionError(msg)
    if ntrees <= 0:
        msg = f"[Rank={rank}] Error: ntrees = {ntrees} should be >= 0"
        raise AssertionError(msg)
    totnbytes = forest_info['Input_ForestNbytes'].sum()
    print(f"[Rank={rank}]: processing {totnbytes} bytes "
          f"(in {ntrees} trees) spread over {nforests} forests...")

    alltreedatafiles = list(set(trees_and_locations['Filename']))
    assert len(alltreedatafiles) > 0
    validate_inputs_are_ctrees_files(alltreedatafiles)

    metadata_dict = get_metadata(alltreedatafiles[0])
    metadata = metadata_dict['metadata']
    version_info = metadata_dict['version']
    input_catalog_type = metadata_dict['catalog_type']
    hdrline = metadata_dict['headerline']

    parser = get_parser(alltreedatafiles[0], fields=fields,
                        drop_fields=drop_fields)

    mergertree_descr = get_treewalk_dtype_descr()
    output_dtype = np.dtype(parser.dtype.descr + mergertree_descr)

    approx_totnumhalos = 0
    for fname in alltreedatafiles:
        ind = np.where(trees_and_locations['Filename'] == fname)
        nbytes = np.sum(trees_and_locations['TreeNbytes'][ind])
        approx_totnumhalos += get_approx_totnumhalos(fname, ndatabytes=nbytes)

    if show_progressbar:
        pbar = tqdm(total=ntrees, unit=' trees', disable=None)

    if (not buffersize) or (buffersize < output_dtype.itemsize):
        buffersize = 1024*1024  # 1 MB
    nbuffer_halos = buffersize // output_dtype.itemsize
    chunks = (nbuffer_halos, )

    output_file = f"{outputdir}/{output_filebase}_{rank}.h5"
    if truncate:
        with h5py.File(output_file, "w") as hf:
            # give the HDF5 root some more attributes
            hf.attrs['input_files'] = np.string_(alltreedatafiles)
            mtimes = [os.path.getmtime(f) for f in alltreedatafiles]
            hf.attrs['input_filedatestamp'] = np.array(mtimes)
            hf.attrs["input_catalog_type"] = np.string_(input_catalog_type)
            hf.attrs[f"{input_catalog_type}_version"] = np.string_(version_info)
            hf.attrs[f"{input_catalog_type}_columns"] = np.string_(hdrline)
            hf.attrs[f"{input_catalog_type}_metadata"] = np.string_(metadata)
            hf.attrs['contiguous-halo-props'] = write_halo_props_cont

            sim_grp = hf.create_group('simulation_params')
            simulation_params = metadata_dict['simulation_params']
            for k, v in simulation_params.items():
                sim_grp.attrs[f"{k}"] = v

            hf.attrs['HDF5_version'] = np.string_(h5py.version.hdf5_version)
            hf.attrs['h5py_version'] = np.string_(h5py.version.version)

            hf.attrs['Nforests'] = 0
            hf.attrs['Ntrees'] = 0
            hf.attrs['Nhalos'] = 0

            forest_dtype = np.dtype([('ForestID', np.int64),
                                     ('ForestHalosOffset', np.int64),
                                     ('ForestNhalos', np.int64),
                                     ('ForestNtrees', np.int64), ])
            hf.create_dataset('ForestInfo', (0,), dtype=forest_dtype,
                              chunks=True, compression=compression,
                              maxshape=(None,))

            tree_dtype = np.dtype([('ForestID', np.int64),
                                   ('TreeRootID', np.int64),
                                   ('TreeHalosOffset', np.int64),
                                   ('TreeNhalos', np.int64),
                                   ('Input_Filename', string_dtype),
                                   ('Input_FileDateStamp', np.float),
                                   ('Input_TreeByteOffset', np.int64),
                                   ('Input_TreeNbytes', np.int64), ])
            hf.create_dataset('TreeInfo', (0,), dtype=tree_dtype,
                              chunks=True, compression=compression,
                              maxshape=(None,))

            forests_grp = hf.create_group('Forests')
            if write_halo_props_cont:
                # Create a dataset for every halo property
                # For any given halo property, the value
                # for halos will be written contiguously
                # (structure of arrays)
                for name, dtype in output_dtype.descr:
                    forests_grp.create_dataset(name, (0,), dtype=dtype,
                                               chunks=chunks,
                                               compression=compression,
                                               maxshape=(None,))
            else:
                # Create a single dataset that contains all properties
                # of a given halo, then all properties of the next halo,
                # and so on (array of structures)
                forests_grp.create_dataset('halos', (0,),
                                           dtype=output_dtype,
                                           chunks=chunks,
                                           compression=compression,
                                           maxshape=(None,))

    halos_buffer = np.empty(nbuffer_halos, dtype=output_dtype)
    nhalos_in_buffer = 0
    with h5py.File(output_file, "a") as hf:
        # The filenames are written as byte-arrays (through np.string_)
        # into the hdf5 file. Therefore, we will need to convert back
        # into `str` objects
        existing_files = hf.attrs['input_files']
        target_all_files = np.unique(np.hstack((existing_files,
                                                alltreedatafiles)))

        # Strictly speaking this decode is not necessary but without
        # this extra decode we end up with an array that contains
        # both np.str_ and str -- MS 01/05/2020
        target_all_files = [x.decode() if isinstance(x, bytes)
                            else str(x) for x in target_all_files]

        if len(target_all_files) > len(existing_files):
            # Since we are appending to the hdf5 file, let's make
            # sure that *all* the files belong to the same setup of
            # simulation + mergertree. However, the ascii files
            # corresponding to the existing data might have been
            # deleted, so we should pass the metadata info directly
            # from the hdf5 file.
            base_input_catalog_type = hf.attrs['input_catalog_type'].decode()
            base_metadata = hf.attrs[f'{base_input_catalog_type}_metadata']
            base_version = hf.attrs[f'{base_input_catalog_type}_version'].decode()

            # Only validate the *current* files being processed
            assert len(alltreedatafiles) > 0
            validate_inputs_are_ctrees_files(alltreedatafiles,
                                             base_metadata=base_metadata,
                                             base_version=base_version,
                                             base_input_catalog_type=base_input_catalog_type)

        # We need to update how many *unique* input files have gone into
        # this hdf5 file
        hf.attrs['input_files'] = np.string_(target_all_files)
        hf.attrs['input_filedatestamp'] = np.array([os.path.getmtime(f)
                                                    for f in target_all_files])

        tree_dset = hf['TreeInfo']
        forest_dset = hf['ForestInfo']

        if forest_dset.shape[0] != hf.attrs['Nforests']:
            msg = "Error: The forest dataset does not contain *exactly* "\
                  "the same number of forests as specified in the file "\
                  f"attribute. Shape of forest dataset = "\
                  f"'{forest_dset.shape}', nforests in file attribute"\
                  f" = '{hf.attrs['Nforests']}'"
            raise AssertionError(msg)
        forest_offset = hf.attrs['Nforests']

        if tree_dset.shape[0] != hf.attrs['Ntrees']:
            msg = "Error: The tree dataset does not contain *exactly* "\
                  "the same number of trees as specified in the file "\
                  f"attribute. shape of tree dataset = '{tree_dset.shape}' "\
                  f"ntrees in file attribute = '{hf.attrs['Ntrees']}'"
            raise AssertionError(msg)
        tree_offset = hf.attrs['Ntrees']

        # resize both the datasets containing the forestlevel info and
        # treelevel info
        forest_dset.resize((forest_offset + nforests, ))
        tree_dset.resize((tree_offset + ntrees, ))

        # Now check the halos dataset
        halos_dset = _create_and_validate_halos_dset(hf, output_dtype,
                                                     write_halo_props_cont)

        # Okay - we have validated the halos offset
        halos_offset = hf.attrs['Nhalos']
        halos_dset_offset = halos_offset

        # resize the halos dataset so we don't have to resize at every step
        dset_size = halos_offset + approx_totnumhalos
        resize_halo_datasets(halos_dset, dset_size,
                             write_halo_props_cont, output_dtype)

        forest_dset[-nforests:, 'ForestID'] = forest_info['ForestID'][:]
        forest_dset[-nforests:, 'ForestNtrees'] = forest_info['Ntrees'][:]

        tree_dset[-ntrees:, 'ForestID', ] = trees_and_locations['ForestID'][:]
        tree_dset[-ntrees:, 'TreeRootID'] = trees_and_locations['TreeRootID'][:]

        # These quantities relate to the input files
        tree_dset[-ntrees:, 'Input_Filename'] = np.string_(trees_and_locations['Filename'][:])
        mtimes = [os.path.getmtime(fn) for fn in trees_and_locations['Filename']]
        tree_dset[-ntrees:, 'Input_FileDateStamp'] = np.array(mtimes)
        tree_dset[-ntrees:, 'Input_TreeByteOffset'] = trees_and_locations['Offset'][:]
        tree_dset[-ntrees:, 'Input_TreeNbytes'] = trees_and_locations['TreeNbytes'][:]

        alltreedatafiles = list(set(trees_and_locations['Filename']))
        if use_pread:
            filehandlers = {f: os.open(f, os.O_RDONLY)
                            for f in alltreedatafiles}
        else:
            filehandlers = {f: io.open(f, 'rt') for f in alltreedatafiles}

        ntrees_processed = 0
        treenhalos = np.empty(ntrees, dtype=np.int64)
        treehalos_offset = np.empty(ntrees, dtype=np.int64)
        forestnhalos = np.empty(nforests, dtype=np.int64)
        foresthalos_offset = np.empty(nforests, dtype=np.int64)
        for iforest in range(nforests):
            foresthalos_offset[iforest] = halos_offset
            forest_halos = np.empty(0, dtype=output_dtype)
            for _ in range(forest_info['Ntrees'][iforest]):
                treedata_file = trees_and_locations['Filename'][ntrees_processed]
                offset = trees_and_locations['Offset'][ntrees_processed]
                numbytes = trees_and_locations['TreeNbytes'][ntrees_processed]
                inp = filehandlers[treedata_file]

                if use_pread:
                    chunk = os.pread(inp, numbytes, offset)
                else:
                    inp.seek(offset, os.SEEK_SET)
                    chunk = inp.read(numbytes)

                parse_line = parser.parse_line
                halos = parser.pack([parse_line(line)
                                     for line in chunk.splitlines()])

                nhalos = halos.shape[0]
                forest_halos.resize(forest_halos.shape[0] + nhalos)

                # forest_halos have additional mergertree indices, therefore
                # the datatypes are not the same between halos (parser.dtype)
                # and forest_halos (output_dtype) -> assign by columns
                for name in parser.dtype.names:
                    forest_halos[name][-nhalos:] = halos[name][:]

                # Add the tree level info
                treenhalos[ntrees_processed] = nhalos
                treehalos_offset[ntrees_processed] = halos_offset
                ntrees_processed += 1
                if show_progressbar:
                    pbar.update(1)

                # Update the total number of halos read-in with
                # the number of halos in this tree
                halos_offset += nhalos

            # Entire forest has been loaded. Reset nhalos
            # to be the number of halos in the forest
            nhalos = forest_halos.shape[0]

            # Add the forest level info
            forestnhalos[iforest] = nhalos

            # Entire forest is now loaded -> add the mergertree indices
            add_tree_walk_indices(forest_halos, rank)

            # If there are not enough to trigger a write, simply fill up
            # the halos_buffer
            if (nhalos_in_buffer + nhalos) < nbuffer_halos:
                assert halos_buffer.dtype == forest_halos.dtype
                halos_buffer[nhalos_in_buffer:nhalos_in_buffer+nhalos] = forest_halos[:]
                nhalos_in_buffer += nhalos
                continue

            # Need to write to disk
            # Resize to make sure there is enough space to append the new halos
            if halos_offset > dset_size:
                resize_halo_datasets(halos_dset, halos_offset,
                                     write_halo_props_cont, output_dtype)
                dset_size = halos_offset

            # write the halos that are already in the buffer
            write_halos(halos_dset, halos_dset_offset, halos_buffer,
                        nhalos_in_buffer, write_halo_props_cont)
            halos_dset_offset += nhalos_in_buffer
            nhalos_in_buffer = 0

            # Now write the halos that have just been read-in
            # Note: The halos in the buffer *must* be written out before
            # the halos that have just been read-in. Otherwise, there will
            # sbe data corruption
            write_halos(halos_dset, halos_dset_offset, forest_halos,
                        nhalos, write_halo_props_cont)
            halos_dset_offset += nhalos
            if halos_offset != halos_dset_offset:
                msg = f"Error: After writing out halos into the hdf5 file, "\
                      f"expected to find that halos_offset = '{halos_offset}'"\
                      f" to be *exactly* equal to the offset in the hdf5 "\
                      f"dataset = '{halos_dset_offset}'"
                raise AssertionError(msg)

        if nhalos_in_buffer > 0:
            write_halos(halos_dset, halos_dset_offset, halos_buffer,
                        nhalos_in_buffer, write_halo_props_cont)
            halos_dset_offset += nhalos_in_buffer
            nhalos_in_buffer = 0

        if halos_offset != halos_dset_offset:
            msg = f"Error: After writing *all* the halos into the hdf5 file, "\
                  f"expected to find that halos_offset = '{halos_offset}'"\
                  f" to be *exactly* equal to the offset in the hdf5 "\
                  f"dataset = '{halos_dset_offset}'"
            raise AssertionError(msg)

        msg = f"Error: Expected to process {ntrees} trees but processed "\
              f"{ntrees_processed} trees instead"
        assert ntrees_processed == ntrees, msg

        # All the trees for this call have now been read in entirely -> Now
        # fix the actual dataset sizes to reflect the total number of
        # halos written
        resize_halo_datasets(halos_dset, halos_offset,
                             write_halo_props_cont, output_dtype)

        # all halos from all forests have been written out and the halo
        # dataset has been correctly resized. Now write the aggregate
        # quantities at the tree and forest levels
        tree_dset[-ntrees:, 'TreeNhalos'] = treenhalos[:]
        tree_dset[-ntrees:, 'TreeHalosOffset'] = treehalos_offset[:]
        forest_dset[-nforests:, 'ForestNhalos'] = forestnhalos[:]
        forest_dset[-nforests:, 'ForestHalosOffset'] = foresthalos_offset[:]

        hf.attrs['Nforests'] += nforests
        hf.attrs['Ntrees'] += ntrees
        hf.attrs['Nhalos'] = halos_offset
        if show_progressbar:
            pbar.close()

        # Close all the open file handlers
        if use_pread:
            for f in filehandlers.values():
                os.close(f)
        else:
            for f in filehandlers.values():
                f.close()

    totnumhalos = halos_offset

    t1 = time.perf_counter()
    print(f"[Rank {rank}]: processing {totnbytes} bytes "
          f"(in {ntrees} trees) spread over {nforests} forests...done. "
          f"Wrote {totnumhalos} halos in {t1-tstart:.2f} seconds")

    sys.stdout.flush()
    return True


def convert_ctrees_to_h5(filenames, standard_consistent_trees=None,
                         outputdir="./", output_filebase="forest",
                         write_halo_props_cont=True,
                         fields=None, drop_fields=None,
                         truncate=True, compression='gzip',
                         buffersize=None, use_pread=True,
                         max_nforests=None,
                         comm=None, show_progressbar=False):
    """
    Convert a set of forests from Consistent Trees ascii file(s) into an
    (optionally compressed) hdf5 file. Can be invoked with MPI.

    Parameters
    -----------

    filenames: list of strings for Consistent-Trees catalogues, required
        The input ascii files will be decompressed, if required.

    standard_consistent_tree: boolean, optional, default: None
        Whether the input filres were generated by the Uchuu collaboration's
        parallel Consistent-Trees code. If only two files are specified in
        ``filenames``, and these two filenames end with 'forests.list', and
        'locations.dat', then a standard Consistent-Trees output will be
        inferred. If all files specified in ``filenames`` end with '.tree',
        then parallel Consistent-Trees is inferred.

    outputdir: string, optional, default: current working directory ('./')
        The directory where the converted hdf5 file will be written in. The
        output filename is obtained by appending '.h5' to the ``input_file``.

    output_filebase: string, optional, default: "forest"
        The output filename is constructed using
        '<outputdir>/<output_filebase>_<rank>.h5'

    write_halo_props_cont: boolean, optional, default: True
        Controls if the individual halo properties are written as distinct
        datasets such that any given property for *all* halos is written
        contiguously (structure of arraysA).

        When set to False, only one dataset ('halos') is created under the
        group 'Forests', and *all* properties of a halo is written out
        contiguously (array of structures).

    fields: list of strings, optional, default: None
        Describes which specific columns in the input file to carry across
        to the hdf5 file. Default action is to convert ALL columns.

    drop_fields: list of strings, optional, default: None
        Contains a list of column names that will *not* be carried through
        to the hdf5 file. If ``drop_fields`` is not set for a
        parallel Consistent-Trees run, then [``Tidal_Force``, ``Tidal_ID``]
        will be used.

        ``drop_fields`` is processed after ``fields``, i.e., you can specify
        ``fields=None`` to create an initial list of *all* columns in the
        ascii file, and then specify
        ``drop_fields = [colname2, colname7, ...]``,
        and *only* those columns will not be present in the hdf5 output.

    truncate: boolean, default: True
        Controls whether a new file is created on this 'rank'. When set to
        ``True``, the header info file is written out. Otherwise, the file
        is appended to. The code checks to make sure that the existing metadata
        in the hdf5 file is identical to the new metadata in the ascii files
        being currently converted (i.e., tries to avoid different
        simulation + mergertree results being present in the same file)

    compression: string, optional, default: 'gzip'
        Controls the kind of compression applied. Valid options are anything
        that ``h5py`` accepts.

    buffersize: integer, optional, default: 1 MB
        Controls the size of the buffer how many halos are written out
        per write call to the hdf5 file. The number of halos written out is
        this buffersize divided the size of the datatype for individual halos.

    use_pread: boolean, optional, default: True
        Controls whether low-level i/o operations (through ``os.pread``) is
        used. Otherwise, higher-level i/o operations (via ``io.open``) is
        used. This option is only meaningful on linux systems (and python3+).
        Since ``pread`` does not change the file offset, additional
        parallelisation can be implemented reasonably easily.

    max_nforests: integer >= 1, optional, default: None
        The maximum number of forests to convert across all tasks. If a
        positive value is passed then the total number of forests converted
        will be ``min(totnforests, max_nforests)``. ValueError is raised
        if the passed parameter value is less than 1.

    comm: MPI communicator, optional, default: None
        Controls whether the conversion is run in MPI parallel. Should be
        compatible with `mpi4py.MPI.COMM_WORLD`.

    show_progressbar: boolean, optional, default: False
        Controls whether a progressbar is printed. Only enables progressbar
        on rank==0, the remaining ranks ignore this keyword.

    Returns
    -------

        Returns ``True`` on successful completion.

    """
    import os
    import sys
    import time
    import numpy as np

    rank = 0
    ntasks = 1
    if comm:
        rank = comm.Get_rank()
        ntasks = comm.Get_size()

    if not os.path.isdir(outputdir):
        msg = f"Error: Output directory = {outputdir} is not a valid directory"
        raise ValueError(msg)

    if max_nforests and max_nforests <= 0:
        msg = f"Error: The maximum number of forests to convert "\
              f"= {max_nforests} must be >= 1"
        raise ValueError(msg)

    tstart = time.perf_counter()
    sys.stdout.flush()

    if not standard_consistent_trees:
        standard_consistent_trees = True
        # The Uchuu collaboration has a special parallel version of the
        # Consistent-Tree developed by @Tomo. This code generates a set of
        # files equivalent to (forests.list, locations.dat, tree_*.dat) file
        # per CPU task. In that code, forests are guaranteed to be located
        # completely within one tree data file. However, the public version
        # of Consistent-Trees is different and there is exactly one
        # 'forests.list' and 'locations.dat' files for the entire simulation,
        # and as many tree_*.dat files as the number of BOX_DIVISIONS^3 set
        # in the Consistent-Trees config file at runtime.

        # This script aims to handle both scenarios with this logic -- if only
        # the "forests.list" and "locations.dat" files are supplied as
        # command-line arguments, then the public version of the
        # Consistent-Trees catalog is assumed. Otherwise, a list of
        # *all* the "XXXXXX.tree" files should be passed (i.e., for people
        # in the Uchuu collaboration). The correspoonding 'forest' and
        # 'location' file names will be automatically generated by assuming
        # the Uchuu convention:
        #  -> tree data files are called '<prefix>.tree'
        #  -> associated forest.list file is called '<prefix>.forest'
        #  -> associated locations.dat file is called '<prefix>.loc'
        #

        # If all files supplied at the command-line endwith
        # '.tree(.bz2,.zip,.gz)' then it is a parallel Consistent-Trees run.
        check_pctrees_files = [True if 'tree' in set(f.split('.'))
                               else False for f in filenames]
        if np.all(check_pctrees_files):
            standard_consistent_trees = False

        if standard_consistent_trees:
            if len(filenames) != 2:
                msg = "Error: To convert a standard Consistent-Trees output, "\
                      "please specify *exactly* two files -- the 'forests.list' and "\
                      "the 'locations.dat' files (order is unimportant). "\
                     f"Instead found filenames = '{filenames}'"
                raise ValueError(msg)

            filebasenames = set([os.path.basename(f) for f in filenames])
            expected_filebasenames = set(['forests.list', 'locations.dat'])
            if filebasenames != expected_filebasenames:
                msg = "Error: To convert a standard Consistent-Trees output, "\
                      "please specify *exactly* two files -- the "\
                      "'forests.list' and the 'locations.dat' files "\
                      "(order is unimportant). While exactly two files were "\
                      "specified, at least one of the 'forests.list' or "\
                      "'locations.dat' files were not present in the "\
                     f"supplied filenames = '{filenames}'"
                raise ValueError(msg)

    if standard_consistent_trees:
        forests_file, locations_file = check_forests_locations_filenames(filenames)
        forests_and_locations_fnames = [(forests_file, locations_file)]
    else:
        # We are converting parallel Ctrees files; however, these files might
        # still be compressed and we need to decompress them before processing.
        forests_and_locations_fnames = []
        decompressed_filenames = []
        for fname in filenames:
            decomp_fname = check_and_decompress(fname)
            extname = '.tree'
            if extname not in decomp_fname:
                msg = "Error: Should pass the tree data file names (i.e., "\
                      f"the filenames should end in '{extname}'. "\
                      f"Instead got filename = '{decomp_fname}'"
                raise ValueError(msg)

            decompressed_filenames.append(decomp_fname)
            forests_file, locations_file, _ = get_all_parallel_ctrees_filenames(decomp_fname)
            forests_and_locations_fnames.append((forests_file, locations_file))

        # Since multiple tree files are specified at the command-line,
        # let's make sure that all the files belong to the
        # same simulation + mergertree setup
        assert len(decompressed_filenames) > 0
        validate_inputs_are_ctrees_files(decompressed_filenames)

    if (not drop_fields) and (not standard_consistent_trees):
        # The Tidal fields can not be correctly calculated in the
        # parallel CTrees code and are dropped from the hdf5 file
        drop_fields = ['Tidal_Force', 'Tidal_ID']

    nfiles = len(forests_and_locations_fnames)
    if rank == 0:
        print(f"[Rank={rank}]: Converting {nfiles} sets of (forests, "
              f"locations) files over {ntasks} tasks ... ")

    nconverted = 0
    for (forests_file, locations_file) in forests_and_locations_fnames:
        t0 = time.perf_counter()
        print(f"[Rank={rank}]: Reading forests and locations files...")
        trees_and_locations = read_locations_and_forests(forests_file,
                                                         locations_file,
                                                         rank)
        ntrees = trees_and_locations.shape[0]
        t1 = time.perf_counter()
        print(f"[Rank={rank}]: Reading forests and locations files...done. "
              f"Time taken = {t1-t0:.2f} seconds")

        alltreedatafiles = list(set(trees_and_locations['Filename']))
        if standard_consistent_trees:
            # Validate that all the files are the same version and all
            # contain a Consistent Trees catalog
            assert len(alltreedatafiles) > 0
            validate_inputs_are_ctrees_files(alltreedatafiles)
        else:
            # Since we are processing parallel CTrees output, *every* tree data
            # file has an associated forests and locations file. Which means
            # that every locations file should only have one treedata file
            # Check that that is the case
            if len(alltreedatafiles) != 1:
                msg = "Error: Expected to find *exactly* one tree data file "\
                      "per locations file. However, while processing the "\
                      f"locations_file =  '{locations_file}', found "\
                      f"{len(alltreedatafiles)} tree data files. "\
                      f"The unique tree data files are: {alltreedatafiles}"
                raise AssertionError(msg)
            # No need to validate that the input files are valid CTrees files
            # That has already been done

        if rank == 0:
            print(f"[Rank={rank}]: Converting a single set of (forests, "
                  f"locations) over {ntasks} tasks...")

        # We have the tree-level info, let's create a similar info for
        # the forests (by grouping trees by ForestID)
        forest_info = get_aggregate_forest_info(trees_and_locations, rank)

        # If `max_nforests` was passed, then we convert at most
        # the first `max_nforests`  -- MS 27/07/2020
        if max_nforests:
            nforests = min(max_nforests, forest_info.shape[0])
            forest_info = forest_info[0:nforests]

        # Distribute the forests over ntasks
        (forest_start, forest_stop) = distribute_array_over_ntasks(forest_info['Input_ForestNbytes'], rank, ntasks)

        forest_ntrees_offset = forest_info['Ntrees'][:].cumsum()
        tree_start = 0
        if forest_start >= 1:
            tree_start = forest_ntrees_offset[forest_start - 1]

        # tree_stop is not-inclusive, and can be directly used in a
        # slicing context
        tree_stop = forest_ntrees_offset[forest_stop]
        print(f"[Rank={rank}]: Processing trees [tree_start, tree_stop) = "
              f"[{tree_start}, {tree_stop}) totntrees = {ntrees}")

        # Now we can start converting the forests
        # *Note* ``forest_stop`` is inclusive but since we are slicing,
        # we need to account for the python convention hence, the slice
        # on ``forest_info`` goes up to ``forest_stop + 1``
        _convert_ctrees_forest_range(forest_info[forest_start:forest_stop+1],
                                     trees_and_locations[tree_start:tree_stop],
                                     rank, outputdir=outputdir,
                                     output_filebase=output_filebase,
                                     write_halo_props_cont=write_halo_props_cont,
                                     fields=fields,
                                     drop_fields=drop_fields,
                                     truncate=truncate,
                                     compression=compression,
                                     buffersize=buffersize,
                                     use_pread=use_pread,
                                     show_progressbar=show_progressbar)

        truncate = False
        nconverted += 1
        if rank == 0:
            print(f"[Rank={rank}]: Converting a single set of (forests, "
                  f"locations) over {ntasks} tasks...done. Converted "
                  f"{nconverted} out of {nfiles} sets of files")

    # Done converting all files. The barrier is necessary to
    # ensure that the container file is created *after* all the
    # data have been converted
    if comm:
        comm.Barrier()

    if rank != 0:
        return True

    # Create the main file that contains the other files as (hdf5-sym-)links
    fname = f'{outputdir}/{output_filebase}.h5'
    outfiles = [f'{outputdir}/{output_filebase}_{itask}.h5'
                for itask in range(ntasks)]
    update_container_h5_file(fname, outfiles,
                             standard_consistent_trees,
                             rank)

    t1 = time.perf_counter()
    print(f"Converting {nfiles} sets of (forests, locations) files "
          f"over {ntasks} tasks ...done. Time taken = {t1-tstart:.2f} seconds")

    return True
