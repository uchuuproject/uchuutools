.. _using_output_hdf5_files:

############################
Using the Output HDF5 files
############################

One complete forest catalogue consists of a single container hdf5 file (containing
metadata), and at least one ``hdf5-treedata`` file (containing forest-level  information). For further details, please refer to the data format :ref:`documentation <data_formats>`.

**********************************
Using the output tree hdf5 files
**********************************
We recommend using the publicly available `ytree <https://github.com/ytree-project/ytree/>`_ python package to inspect and visualise the output hdf5 files. Please refer to the ytree documentation on `reading <https://ytree.readthedocs.io/en/latest/Loading.html#consistent-trees-hdf5>`_, `searching <https://ytree.readthedocs.io/en/latest/Arbor.html#searching-through-merger-trees-accessing-like-a-database>`_ and `visualising <https://ytree.readthedocs.io/en/latest/Plotting.html>`_ mergertrees within `ytree <https://github.com/ytree-project/ytree/>`_.


If you want to use the hdf5 files directly, then here are a few eexamples:

.. raw:: html

   <details>
   <summary><a>Loading an entire forest based on ForestID</a></summary>

.. code-block:: python

    h5_filename = 'forest_0.h5'
    track_forestID = 'XXXXXXXX'
    fields = ['mvir', 'conc', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'vmax']

    from uchuutools.readers import read_forest_from_forestID
    forest_halos = read_forest_from_forestID(h5_filename, track_forestID, fields=fields)

.. raw:: html

   </details>


.. raw:: html

   <details>
   <summary><a>Loop through all forests in a single hdf5-treedata file</a></summary>

.. code-block:: python

    h5_filename = 'forest_0.h5'
    fields = ['mvir', 'conc', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'vmax']

    from uchuutools.readers import read_forest_from_forestidx, get_halo_dtype
    with h5py.File(h5_filename, 'r') as hf:
        dtype = get_halo_dtype(hf, fields)
        for fidx in range(hf.attrs['Nforests']):
            forest_halos = read_forest_from_forestidx(hf, fidx, fields=fields, dtype=dtype)
            # do something with ``forest_halos``

.. raw:: html

   </details>


.. raw:: html

   <details>
   <summary><a>Loop through all forests associated with a container file (i.e., the entire catalogue)</a></summary>

.. code-block:: python

    h5_filename = 'forest.h5'
    fields = ['mvir', 'conc', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'vmax']

    from uchuutools.readers import read_forest_from_forestidx, get_halo_dtype
    with h5py.File(h5_filename, 'r') as hf:
        dtype = get_halo_dtype(hf, fields)
        totnforests = hf.attrs['TotNforests']
        nfiles = hf.attrs['Nfiles']
        print(f"Reading {totnforests} spread over {nfiles} files ...")
        for ifile in range(nfiles):
            file_grp = f'File{ifile}'
            nforests_this_file = hf[file_grp].attrs['Nforests']
            print(f"Reading {nforests_this_file} from file#{ifile} ...")
            for fidx in range(nforests_this_file):
                forest_halos = read_forest_from_forestidx(hf[file_grp], fidx, fields=fields, dtype=dtype)
                # do something with ``forest_halos``

            print(f"Reading {nforests_this_file} from file#{ifile} ...done")

        print(f"Reading {totnforests} spread over {nfiles} files ...done")

.. raw:: html

   </details>

**********************************
Using the output halo hdf5 files
**********************************
The ``hdf5-halocat`` files contain halo information and each entry is a separate halo, without any imposed ordering or
association between the halos. Therefore, the halos can be processed either completely or in chunks.

.. raw:: html

   <details>
   <summary><a>Loop through all halos in a single hdf5-halocat file</a></summary>

.. code-block:: python

    h5_filename = 'RockstarExtended/halodir_050/halolist_z0p00_0.h5'
    fields = ['mvir', 'conc', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'vmax']

    from uchuutools.readers import read_halocat, get_halo_dtype
    with h5py.File(h5_filename, 'r') as hf:
        dtype = get_halo_dtype(hf, fields, named_data_group='/')
        for halos in read_halocat(hf, fields=fields, dtype=dtype, named_data_group='/'):
            # do something with ``halos``

.. raw:: html

   </details>

.. note:: The Uchuu halo catalogues were generated with an earlier version of the uchuutools script and you will
          need to set the ``named_data_group='/'`` parameter when reading the halo catalogues.
