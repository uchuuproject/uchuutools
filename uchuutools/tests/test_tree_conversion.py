#!/usr/bin/env python
from __future__ import print_function

__author__ = "Manodeep Sinha"
__all__ = ["test_ctrees_conversion"]

import numpy as np


from utils import get_parser
from ctrees_utils import get_treewalk_dtype_descr


def _loadtree_from_offset(fp, offset, parser):
    fp.seek(offset)
    X = []
    for line in fp:
        if line[0] == '#':
            break
        X.append(parser.parse_line(line))

    return parser.pack(X)


# def _test_tree_walk_indices(forest):
#     print(f"forest.dtype.names = {forest.dtype.names}")
#     for prog, desc in enumerate(forest['Descendant']):
#         if desc == -1:
#             continue

#         if forest['scale'][desc] <= forest['scale'][prog]:
#             msg = f"Error: desc_scale = {forest['scale'][desc]} must be "\
#                   f"greater than progenitor scale = {forest['scale'][prog]}"\
#                   f"desc = {desc} prog = {prog} "\
#                   f"desc_id = {forest['desc_id'][prog]}"\
#                   f"id of desc = {forest['id'][desc]}, prog, "\
#                   f"Tree_root_ID = {forest['Tree_root_ID'][prog]} "\
#                   f"desc, Tree_root_ID = {forest['Tree_root_ID'][desc]}"

#             raise AssertionError(msg)
#         assert forest['id'][desc] == forest['desc_id'][prog]

#         if forest['FirstProgenitor'][desc] == prog:
#             # count the number of progenitors
#             num_prog = 1
#             nextprog = forest['NextProgenitor'][prog]
#             while nextprog != -1:
#                 num_prog += 1
#                 nextprog = forest['NextProgenitor'][prog]
#             assert forest['num_prog'][desc] == num_prog
#         else:
#             # This progenitor is not the primary progenitor
#             # -> we need to check the nextprog values -> nextprog
#             # *must* exist
#             firstprog = forest['FirstProgenitor'][desc]
#             nextprog = forest['NextProgenitor'][firstprog]
#             msg = f"firstprog = {firstprog} desc = {desc} "\
#                   f"prog = {prog} nextprog = {nextprog}"
#             assert nextprog != -1, msg
#             while nextprog != prog:
#                 assert forest['desc_id'][nextprog] == forest['id'][desc]
#                 ii = forest['NextProgenitor'][nextprog]
#                 assert ii != -1
#                 assert forest['Mvir'][nextprog] >= forest['Mvir'][ii]
#                 nextprog = ii

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
            for name in hf['Forests'].keys():
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
            ntrees = hf['TreeInfo'].shape[0]
            pbar = tqdm(total=ntrees, unit='tree', disable=None)

        for treeinfo in hf['TreeInfo']:
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

            # Check the primary tree-walking indices
            # (FirstHaloInFOFgroup, NextHaloInFOFgroup, PrevHaloInFOFgroup)
            # (Descendant, FirstProgenitor, NextProgenitor, PrevProgenitor)
            prop = 'FirstHaloInFOFgroup'
            fof_idx = tree[prop][:]
            msg = f"Error: The halos do not correctly point to the "\
                  f"host fof halo. ID of the halo pointed to by '{prop}' "\
                  f"should match the 'FofID'"
            assert_array_equal(tree['id'][fof_idx], tree['FofID'][:])

            prop = 'Descendant'
            valid_desc = tree[prop][:] != -1
            msg = f"Error: The halos do not correctly point to the "\
                  f"descendant halo. The ID of the halo pointed to by "\
                  f"'{prop}' should match the 'desc_id'"
            desc = tree['Descendant'][valid_desc]
            assert_array_equal(tree['id'][desc],
                               tree['desc_id'][valid_desc],
                               err_msg=msg)

            prop = 'FirstProgenitor'
            valid_first_prog = tree[prop][:] != -1
            msg = f"Error: The halos do not correctly point to the "\
                  f"descendant halo. The 'desc_id' of the halo pointed "\
                  f"to by '{prop}' should match the id of the halo "\
                  f"containing the '{prop}'"
            first_prog = tree[prop][valid_first_prog]
            assert_array_equal(tree['id'][valid_first_prog],
                               tree['desc_id'][first_prog],
                               err_msg=msg)

            if show_progressbar:
                pbar.update(1)

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
    base = ""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        comm = None

    base = "/home/msinha/scratch/simulations/uchuu/U2000/trees/hdf5/with-tree-indices"
    h5files = [f"{base}/forest_{rank}.h5" for rank in range(5)]
    test_ctrees_conversion(h5files, comm)
