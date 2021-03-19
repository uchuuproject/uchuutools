.. _data_formats:

#########################
HDF5 Output Data Format
#########################


Within the following sections we will assume that the generated hdf5 file has been opened with the following code:

.. code-block:: python

    import h5py
    # if using the defaults, ``h5_filename`` could be
    # - ``./forest_0.h5`` (for Consistent-Trees tree catalogue)
    # - ``./out_0.list.h5`` (for Rockstar halo catalogue)
    # - ``./hlist_<scale_factor>.list.h5`` (for Consistent-Trees halo catalogue)
    hf = h5py.File(h5_filename, 'r')

Consistent-Trees HDF5 format for Tree catalogues
*************************************************

The Consistent-Trees tree hdf5 format consists of two types of files:

#. one container hdf5 file,
#. one or more hdf5 files that contain the forest/tree/halo level information (we will refer to this as ``hdf5-treecat`` file)

Container HDF5 file format
===========================

The following attributes of the container hdf5 file may be useful to the user:

#. **Nfiles**: Total number of hdf5 data files that are associated with this container file (``np.int64``)
#. **TotNforests**: Total number of forests stored across all associated ``Nfiles`` (``np.int64``)
#. **TotNtrees**: Total number of trees stored across all associated ``Nfiles`` (``np.int64``)
#. **TotNhalos**: Total number of halos stored across all associated ``Nfiles`` (``np.int64``)

The individual hdf5 files containing the halo-level information are embedded as ``ExternalLinks`` within the container
hdf5 file under ``File<ifile>``, where ``ifile`` ranges from ``[0, Nfiles)``. Users can loop over these external links and transparently read all the halos.

.. code-block:: python

    import h5py
    with h5py.File(container_file, 'r') as hf:
        nfiles = hf.attrs['Nfiles']
        for i in range(nfiles):
            # usage will depend on the value of ``write_halo_props_cont``
            # used during the creation of these files.
            mvir = hf[f"File{i}/Forests/Mvir"]

.. raw:: html

   <details>
   <summary><a>Source code that creates the container hdf5 file</a></summary>

.. code-block:: python

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
                nfiles = hf['/'].attrs['Nfiles']
                for ifile in range(nfiles):
                    outfiles.append(hf[f'File{ifile}'].file)
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

.. raw:: html

   </details>

HDF5-treecat file format
==========================

There may be one or more hdf5 data-files written as part of the conversion process. These files contain the actual halo information, as well as tree-level and forest-level information contained in the original ascii Consistent-Trees tree catalogues. In this section, we will describe this ``hdf5-treecat`` file format.

.. note:: The total number of hdf5 data-files associated with the container file is simply the number of parallel tasks used during the ascii->hdf5 conversion. For serial conversions, there will be *exactly* one hdf5 data-file (by defaut, named ``./forest_0.h5``)


File-level Attributes (``list(hf.attrs)``)
-------------------------------------------
The ``hdf5-treecat`` file has attributes at the root-level to store metadata about the input ascii Consistent-trees catalogues. The following attributes of the container hdf5 file facilitate reading the hdf5 file:

#. **Nforests**: Total number of forests stored in this file(``np.int64``)
#. **Ntrees**: Total number of trees stored in this file (``np.int64``)
#. **Nhalos**: Total number of halos stored in this file (``np.int64``)
#. **simulation\_params**: An hdf5 group that contains cosmological parameters (``Omega_M``, ``Omega_L``, ``hubble``) and the simulation boxsize (``Boxsize``)

.. raw:: html

   <details>
   <summary><a>Source code that creates the file-level attributes</a></summary>

.. code-block:: python

        # give the HDF5 root some attributes
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

        ### These two lines are executed at the end, while creating
        ### the container file :func:`update_container_h5_file`.
        ### ``hf_task`` here refers to ``hf`` in the preceeding
        ### chunk of code
        if standard_consistent_trees:
            hf_task.attrs['consistent-trees-type'] = 'standard'
        else:
            hf_task.attrs['consistent-trees-type'] = 'parallel'

.. raw:: html

   </details>


Halo-level info (``hf['Forests']``)
------------------------------------

Halos are written under a ``Forests`` group within the hdf5 file. If each selected halo property is written separately (i.e., with the default option of ``write_halo_props_cont=True``), then individual halo properties are written as a separate dataset as ``Forests/<property_name>`` (e.g., ``Forests/M200c``). If all selected properties of a halo are written contiguously (i.e., with the user-specified option of ``write_halo_props_cont=False``), then the halos are written as a single dataset ``Forests/halos``.

For each forest, all halos are written contiguously. Further, within each forest, all halos from the same tree are written contiguously. Hence the starting index and number of halos stored in the ``TreeInfo`` and ``ForestInfo`` datasets can be directly used to read all halos from the same tree/forest.

.. raw:: html

   <details>
   <summary><a>Source code that creates the dataset containing the halos</a></summary>

.. code-block:: python

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


.. raw:: html

   </details>

By design, the halo properties are written as chunked and compressed. If you plan to read these hdf5 files repeatedly, then you will get faster read-times if you re-write the hdf5 files as unchunked. If you intend to keep the compression, then you will likely get a better compression ratio as well (compression in hdf5 only works on the chunks). You can accomplish that by running the following on the command-line:

.. code-block:: bash

    h5repack -i forest_0.h5 -o forest_0_conti.h5 -l CONTI
    h5repack -i forest_0_conti.h5 -o forest_0_conti_gz4.h5 -f GZIP=4
    ## if the previous two are successfull
    mv forest_0_conti_gz4.h5 forest_0.h5 && rm forest_0_conti.h5


.. note::
    Any special characters in the Consistent-Trees halo property name are replaced with a single underscore ``_``. For example, ``A[x](500c)`` in the input ascii file is written as ``A_x_500c`` in the hdf5 file. This name conversion is done by the function :func:`uchuutools.utils.sanitize_ctrees_header`.

.. raw:: html

   <details>
   <summary><a>Source code that sanitizes the names of halo properties in the Consistent-Trees catalogue</a></summary>

.. code-block:: python

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

.. raw:: html

   </details>


Forest-level info (``hf['Forestinfo]``)
----------------------------------------

Since all halos from the same forest are written contiguously, the forest level info is there to allow easy access to entire forests. This info is stored in the dataset ``ForestInfo`` and contains the following fields:

#. **ForestID**: Contains the ``ForestID`` as assigned by Consistent-Trees (``np.int64``)
#. **ForestHalosOffset**: Contains the index of the first halo contained within each forest
#. **ForestNhalos**: Contains the total number of halos within each forest (``np.int64``)
#. **ForestNtrees**: Contains the total number of trees within each forest (``np.int64``)

The number of entries in this ``ForestInfo`` dataset (i.e., the shape) equals the number of forests stored in the hdf5 file.

.. raw:: html

   <details>
   <summary><a>Source code that creates the dataset with the forest-level info</a></summary>

.. code-block:: python

            forest_dtype = np.dtype([('ForestID', np.int64),
                                     ('ForestHalosOffset', np.int64),
                                     ('ForestNhalos', np.int64),
                                     ('ForestNtrees', np.int64), ])
            hf.create_dataset('ForestInfo', (0,), dtype=forest_dtype,
                              chunks=True, compression=compression,
                              maxshape=(None,))
.. raw:: html

   </details>


Tree-level info (``hf['TreeInfo']``)
-------------------------------------

Since the halos are stored on a **per tree** basis in the input ascii Consistent-Trees catalogue, data provenance requires that we store that original information at a tree level as well. In addition, this allows us to quickly read a single tree for visualisation/testing (rather than the entire forest). This info is stored in the dataset ``TreeInfo`` and contains the following fields:

#. **ForestID**: Contains the ``ForestID`` as assigned by Consistent-Trees (``np.int64``)
#. **TreeRootID**: Contains the ``TreeRootID`` as assigned by Consistent-Trees (``np.int64``)
#. **TreeHalosOffset**: Contains the index of the first halo contained within each tree (``np.int64``)
#. **TreeNhalos**: Contains the total number of halos within each tree (``np.int64``)
#. **Input_Filename**: Contains the input ascii Consistent-Trees filename(string, ``'S1024'``)
#. **Input_FileDateStamp**: Contains the modification time of the input ascii Consistent-Trees file (``np.float``)
#. **Input_TreeByteOffset**: Contains the byte offset of the first halo within the input ascii Consistent-Trees file (``np.int64``)
#. **Input_TreeNbytes**: Contains the total number of bytes for this tree within the input ascii Consistent-Trees file (``np.int64``)

Fields prefixed with ``Input_`` are there solely for tracking back to the original files or ease of access (``Input_TreeNbytes``). The number of entries in this ``TreeInfo`` dataset (i.e., the shape) equals the number of trees stored in the hdf5 file.


.. raw:: html

   <details>
   <summary><a>Source code that creates the dataset with the tree-level info</a></summary>

.. code-block:: python

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

.. raw:: html

   </details>



------------


Rockstar/Consistent-Trees HDF5 format for halo catalogues
**********************************************************
Each Rockstar ``out_*.list``, or Consistent-Trees ``hlist_*.list`` files is converted
into a single hdf5 file (``hdf5-halocat`` file). The halos in the hdf5 files are written
in the exact same order as the input ascii files.

HDF5-halocat file format
==========================

File-level Attributes
----------------------
The ``hdf5-halocat`` file has attributes at the root-level to store metadata about the input ascii Consistent-trees catalogues. The following attributes of the container hdf5 file facilitate reading the hdf5 file:

#. **TotNhalos**: Total number of halos stored in this file (``np.int64``)
#. **scale\_factor**: Total number of forests stored in this file(``np.float``)
#. **redshift**: The redshift for the halo catalogue (``np.float``)
#. **redshift\_params**: An hdf5 group that contains cosmological parameters (``Omega_M``, ``Omega_L``, ``hubble``) and the simulation boxsize (``Boxsize``)

.. raw:: html

   <details>
   <summary><a>Source code that creates the file-level attributes</a></summary>

.. code-block:: python

        line_with_scale_factor = ([line for line in metadata
                                   if line.startswith("#a")])[0]
        scale_factor = float((line_with_scale_factor.split('='))[1])
        redshift = 1.0/scale_factor - 1.0

        # give the HDF5 root some attributes
        hf.attrs[u"input_filename"] = np.string_(input_file)
        hf.attrs[u"input_filedatestamp"] = np.array(os.path.getmtime(input_file))
        hf.attrs[u"input_catalog_type"] = np.string_(input_catalog_type)
        hf.attrs[f"{input_catalog_type}_version"] = np.string_(version_info)
        hf.attrs[f"{input_catalog_type}_columns"] = np.string_(hdrline)
        hf.attrs[f"{input_catalog_type}_metadata"] = np.string_(metadata)
        sim_grp = hf.create_group('simulation_params')
        simulation_params = metadata_dict['simulation_params']
        for k, v in simulation_params.items():
            sim_grp.attrs[f"{k}"] = v

        hf.attrs[u"HDF5_version"] = np.string_(h5py.version.hdf5_version)
        hf.attrs[u"h5py_version"] = np.string_(h5py.version.version)
        hf.attrs[u"TotNhalos"] = -1
        hf.attrs[u"scale_factor"] = scale_factor
        hf.attrs[u"redshift"] = redshift


.. raw:: html

   </details>


Halo-level info
-----------------


.. raw:: html

   <details>
   <summary><a>Source code that creates the dataset containing halos</a></summary>

.. code-block:: python

        halos_grp = hf.create_group('HaloCatalogue')
        halos_grp.attrs['scale_factor'] = scale_factor
        halos_grp.attrs['redshift'] = redshift

        dset_size = approx_totnumhalos
        if write_halo_props_cont:
            halos_dset = dict()
            # Create a dataset for every halo property
            # For any given halo property, the value
            # for halos will be written contiguously
            # (structure of arrays)
            for name, dtype in parser.dtype.descr:
                halos_dset[name] = halos_grp.create_dataset(name,
                                                            (dset_size, ),
                                                            dtype=dtype,
                                                            chunks=True,
                                                            compression=compression,
                                                            maxshape=(None,))
        else:
            # Create a single dataset that contains all properties
            # of a given halo, then all properties of the next halo,
            # and so on (array of structures)
            halos_dset = halos_grp.create_dataset('halos', (dset_size,),
                                                  dtype=parser.dtype,
                                                  chunks=True,
                                                  compression=compression,
                                                  maxshape=(None,))

.. raw:: html

   </details>

