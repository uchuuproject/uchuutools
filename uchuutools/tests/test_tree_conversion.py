#!/usr/bin/env python
__author__ = "Manodeep Sinha"
__all__ = ["test_ctrees_conversion"]

import numpy as np
import os

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
        columns = (columns, )

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


def _test_single_h5file(h5file, asciidir=None, show_progressbar=True):
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
                fld = f"Forests/{name}"
                if not isinstance(hf[fld], h5py.Dataset):
                    continue
                props.append(name)
                arr = np.asarray(hf[fld])
                halo_props[name] = arr
        else:
            props = dtype.names
            halo_props = np.asarray(hf["Forests/halos"])

        treefiles = [tf.decode() for tf in hf.attrs[u'input_files']]

        ## MS (08/03/2021): The concise (and less resilient) version
        # of replacing the input tree filenames with the alternate directory
        # treefiles = [tf if (os.path.isfile(tf)) else
        #                 f"{asciidir}/{os.path.basename(tf)}"
        #                 for tf in treefiles]

        ## MS: The verbose version of replacing the input ascii tree filenames
        for tf in treefiles:
            if os.path.isfile(tf):
                continue
            else:
                if not asciidir or (not os.path.isdir(asciidir)):
                    msg = "Error: Could not locate the original ascii "
                    msg += f"Consistent-Trees file (filename = {tf}).\n"
                    msg += "If the files have been moved to another directory,"
                    msg += "please pass that directory name via the"
                    msg += "`asciidir` parameter"
                    raise FileNotFoundError(msg)

                basename = os.path.basename(tf)
                newname = f"{asciidir}/{basename}"
                if os.path.isfile(newname):
                    # replace the filename in the list of (unique) filenames
                    print(f"LOG: Replacing old filename '{tf}' with the "
                          f"new filename '{newname}'")
                    treefiles.remove(tf)
                    treefiles.append(newname)
                else:
                    msg = f"Error: Could not locate filename = {newname} as"
                    msg += "the alternate file based on the alternate \nascii "
                    msg += f"directory = {asciidir}.\nThe original "
                    msg += f"filename specified in the hdf5 file is {tf}\n"
                    raise FileNotFoundError(msg)

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

                try:
                    fileptr = tree_file_handles[tfile]
                except KeyError:
                    basename = os.path.basename(tfile)
                    newname = f"{asciidir}/{basename}"
                    fileptr = tree_file_handles[newname]

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


def test_ctrees_conversion(h5files, asciidir=None,
                           show_progressbar=True, comm=None):
    """
    Tests whether the list of hdf5 filenames correctly converted the input
    Consistent-Trees data

    Parameters
    ----------

    h5files: list of filenames, string, required
        The names of the converted files (e.g., [`forest_0.h5`, `forest_1.h5`])
        to be tested

    asciidir: directory name, string, optional, default:None
        The name of an alternate directory containing the original ascii files.
        This will be only be used where the ascii filenames in hdf5-file metadata
        can not be accessed

    show_progressbar: boolean, optional, default: True
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
        h5files = (h5files, )

    if rank != 0:
        show_progressbar = False

    tstart = time.perf_counter()
    nfiles = len(h5files)
    for ii in range(rank, nfiles, ntasks):
        h5file = h5files[ii]
        t0 = time.perf_counter()
        print(f"[Rank={rank}]: Testing trees in the {h5file} file ...")
        _test_single_h5file(h5file, asciidir=asciidir, show_progressbar=show_progressbar)
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

    parser.add_argument("-a", '--asciidir', metavar='<alternate CTrees directory>',
                        type=str, nargs='?', default=None,
                        help="alternate directory containing the "\
                             "ascii Consistent-Trees files")

    prog_group = parser.add_mutually_exclusive_group()
    prog_group.add_argument("-p", "--progressbar", dest='show_progressbar',
                            action="store_true", default=True,
                            help="display a progressbar on rank=0")
    prog_group.add_argument("-np", "--no-progressbar", dest='show_progressbar',
                            action='store_false', help="disable the progressbar")

    parser.parse_args()
    args = parser.parse_args()

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        comm = None

    test_ctrees_conversion(args.h5files,
                           asciidir=args.asciidir,
                           show_progressbar=args.show_progressbar,
                           comm=comm)
