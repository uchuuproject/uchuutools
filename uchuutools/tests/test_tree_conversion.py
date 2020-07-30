#!/usr/bin/env python
from __future__ import print_function

__author__ = "Manodeep Sinha"
__all__ = ["test_ctrees_conversion"]

import numpy as np

from uchuutools.utils import get_parser
from uchuutools.ctrees_utils import get_treewalk_dtype_descr


def _loadtree_from_offset(fp, offset, parser):
    fp.seek(offset)
    X = []
    for line in fp:
        if line[0] == '#':
            break
        X.append(parser.parse_line(line))

    return parser.pack(X)


def _load_forest_columns(halo_props, starthalo, nhalos, columns):
    if not isinstance(columns, (list, tuple)):
        columns = [columns]

    colnames = set(columns)
    descr = [(name, halo_props[name].dtype) for name in colnames]
    forest = np.empty(nhalos, dtype=np.dtype(descr))
    for prop in columns:
        forest[prop][:] = halo_props[prop][starthalo:starthalo+nhalos]

    return forest


def _validate_forest_walk_indices(forest):
    from numpy.testing import assert_array_equal

    nhalos = forest.shape[0]
    # Check the primary tree-walking indices
    # (FirstHaloInFOFgroup, NextHaloInFOFgroup, PrevHaloInFOFgroup)
    # (Descendant, FirstProgenitor, NextProgenitor, PrevProgenitor)
    indices_fields = ['FirstHaloInFOFgroup', 'NextHaloInFOFgroup',
                      'PrevHaloInFOFgroup', 'Descendant',
                      'FirstProgenitor', 'NextProgenitor',
                      'PrevProgenitor']
    for fld in indices_fields:
        # these are all indices and therefore must be either -1
        # or have a value within [0, nhalos-1]
        valid = np.where(forest[fld] != -1)[0]
        if len(valid) > 0:
            msg = f"The min. {fld} index = {forest[fld][valid].min()} "\
                  "should be >= 0"
            assert forest[fld][valid].min() >= 0, msg
            msg = f"The max. {fld} index = {forest[fld][valid].max()} "\
                  f"should be < nhalos = {nhalos}"
            assert forest[fld][valid].max() < nhalos, msg

    prop = 'FirstHaloInFOFgroup'
    fof_idx = forest[prop][:]
    msg = f"Error: The halos do not correctly point to the "\
          f"host fof halo. ID of the halo pointed to by '{prop}' "\
          f"should match the 'FofID'"
    assert_array_equal(forest['id'][fof_idx], forest['FofID'][:])

    prop = 'Descendant'
    valid_desc = forest[prop][:] != -1
    msg = f"Error: The halos do not correctly point to the "\
          f"descendant halo. The ID of the halo pointed to by "\
          f"'{prop}' should match the 'desc_id'"
    desc = forest[prop][valid_desc]
    assert_array_equal(forest['id'][desc],
                       forest['desc_id'][valid_desc],
                       err_msg=msg)
    prop = 'scale'
    msg = f"Error: The halos do not correctly point to the "\
          f"descendant halo. The scale-factor of the halo pointed "\
          f"to by '{prop}' should match the 'desc_scale'"
    assert_array_equal(forest[prop][desc],
                       forest['desc_scale'][valid_desc],
                       err_msg=msg)
    msg = "Error: The scale-factor of the descendant halo "\
          "should be larger than the scale-factor of the "\
          "progenitor halo. "
    assert np.min(forest['scale'][desc] - forest['scale'][valid_desc]) > 0, msg

    prop = 'FirstProgenitor'
    valid_first_prog = forest[prop][:] != -1
    msg = f"Error: The halos do not correctly point to the "\
          f"descendant halo. The 'desc_id' of the halo pointed "\
          f"to by '{prop}' should match the id of the halo "\
          f"containing the '{prop}'"
    first_prog = forest[prop][valid_first_prog]
    assert_array_equal(forest['id'][valid_first_prog],
                       forest['desc_id'][first_prog],
                       err_msg=msg)

    return True


def _test_single_h5file(h5file, show_progressbar=True):
    import h5py
    from numpy.testing import assert_array_equal
    from tqdm import tqdm

    with h5py.File(h5file, 'r') as hf:
        cont_halo_props = False
        try:
            dtype = hf['Forests/halos'].dtype
        except KeyError:
            cont_halo_props = True

        if cont_halo_props:
            halo_props = dict()
            props = []
            for name in tqdm(hf['Forests'].keys()):
                if not isinstance(hf[f"Forests/{name}"], h5py.Dataset):
                    continue
                props.append(name)
                arr = np.asarray(hf[f"Forests/{name}"])
                halo_props[name] = arr
        else:
            props = dtype.names
            halo_props = np.asarray(hf["Forests/halos"])

        treefiles = [tf.decode() for tf in hf.attrs[u'input_files']]
        tree_file_handles = dict((tf, open(tf, 'rt')) for tf in treefiles)
        parser = get_parser(treefiles[0])
        mergertree_descr = get_treewalk_dtype_descr()
        mergertree_names = set([name for name, _ in mergertree_descr])

        if show_progressbar:
            nforests = hf['ForestInfo'].shape[0]
            pbar = tqdm(total=nforests, unit='forest', disable=None)

        treenum = 0
        for forestinfo in hf['ForestInfo']:
            ntrees_in_forest = forestinfo['ForestNtrees']
            for treeinfo in hf['TreeInfo'][treenum:treenum + ntrees_in_forest]:
                tfile = treeinfo['Input_Filename'].decode().replace('\x00', '')
                offset = treeinfo['Input_TreeByteOffset']
                fileptr = tree_file_handles[tfile]
                tree = _loadtree_from_offset(fileptr, offset, parser)

                h5_start = treeinfo['TreeHalosOffset']
                h5_end = h5_start + treeinfo['TreeNhalos']
                for prop in props:
                    if prop in mergertree_names:
                        continue

                    msg = f"Error: Halo property = '{prop}' is not the same "\
                        "between the hdf5 and ascii files"

                    assert_array_equal(tree[prop][:],
                                       halo_props[prop][h5_start:h5_end],
                                       err_msg=msg)
            nhalos_in_forest = forestinfo['ForestNhalos']
            start_halo = forestinfo['ForestHalosOffset']
            forest_columns = ['id', 'scale', 'desc_scale', 'desc_id']
            forest_columns.extend(mergertree_names)

            forest = _load_forest_columns(halo_props, start_halo,
                                          nhalos_in_forest, forest_columns)

            _validate_forest_walk_indices(forest)
            if show_progressbar:
                pbar.update(1)

            treenum += ntrees_in_forest

    if show_progressbar:
        pbar.close()

    return True


def test_ctrees_conversion(h5files, show_progressbar=True, comm=None):
    """
    Tests whether the list of hdf5 filenames correctly converted the input
    Consistent-Trees data

    Parameters
    ----------

    h5files: list of filenames, string, required
        The names of the converted files (e.g., [`forest_0.h5`, `forest_1.h5`])
        to be tested

    show_progressbar: boolean, optional
        Controls whether the progressbar is shown during testing

    comm: MPI communicator, optional, default: None
        Controls whether the conversion is run in MPI parallel. Should be
        compatible with `mpi4py.MPI.COMM_WORLD`.

    Returns
    -------

    trees_and_locations: A numpy structured array
        A numpy structured array containing the fields
        <TreeRootID ForestID Filename FileID Offset TreeNbytes>
        The array is sorted by ``(ForestID, Filename, Offset)`` in that order.
        This sorting means that *all* trees belonging to the same forest
        *will* appear consecutively regardless of the file that the
        corresponding tree data might appear in. The number of elements
        in the array is equal to the number of trees.

        Note: Sorting by ``Filename`` is implemented by an equivalent, but
        faster sorting on ``FileID``.

    """
    import time

    rank = 0
    ntasks = 1
    if comm:
        rank = comm.Get_rank()
        ntasks = comm.Get_size()

    # Protect against the case where a single file was passsed
    if not isinstance(h5files, (list, tuple)):
        h5files = [h5files]

    if rank != 0:
        show_progressbar = False

    tstart = time.perf_counter()
    nfiles = len(h5files)
    for ii in range(rank, nfiles, ntasks):
        h5file = h5files[ii]
        t0 = time.perf_counter()
        print(f"[Rank={rank}]: Testing trees in the {h5file} file ...")
        _test_single_h5file(h5file, show_progressbar=show_progressbar)
        t1 = time.perf_counter()
        print(f"[Rank={rank}:] Testing trees in the {h5file} file ...done. "
              f"Time taken = {t1-t0:0.2f} seconds")

    t1 = time.perf_counter()
    print(f"[Rank={rank}]: Tested {nfiles} in {t1-tstart:0.2f} seconds")

    return True


if __name__ == "__main__":
    import argparse
    descr = "Test the converted hdf5 forest-files against the input ascii "\
            "Consistent-Trees files (optionally in MPI parallel)"
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('h5files', metavar='<hdf5 file(s)>',
                        type=str, nargs='+',
                        help="the full path to the hdf5 files with the "
                             "forest data")

    parser.parse_args()
    args = parser.parse_args()

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        comm = None

    test_ctrees_conversion(args.h5files, comm)
