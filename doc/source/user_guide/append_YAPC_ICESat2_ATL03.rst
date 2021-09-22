============================
append_YAPC_ICESat2_ATL03.py
============================

- Reads ICESat-2 `ATL03 geolocated photon height product files <https://nsidc.org/data/ATL03>`_ and appends photon classification flags from YAPC (*Yet Another Photon Classifier*)

    * ``yapc_snr_norm``: segment level photon weight normalization
    * ``yapc_snr``: the photon level normalized YAPC signal-to-noise ratio
    * ``yapc_conf``: YAPC-based confidence levels

 Calling Sequence
 ================

 .. code-block:: bash

    python append_YAPC_ICESat2_ATL03.py --verbose --mode 0o775 <path_to_ATL03_file>

`Source code`__

.. __: https://github.com/tsutterley/yapc/blob/main/scripts/append_YAPC_ICESat2_ATL03.py

Inputs
######

1. ``ATL03_file``: full path to ATL03 file

Command Line Options
####################

- ``--K X``, ``-k X``: number of values for KNN algorithm
- ``--min-ph X``: minimum number of photons for a major frame to be valid
- ``--min-x-spread X``: minimum along-track spread of photon events
- ``--min-h-spread X``: minimum window of heights for photon events
- ``--aspect X``: aspect ratio of x and h window
- ``--method X``: algorithm for computing photon event weights

    * ``'ball_tree'``: use scikit.learn.BallTree with custom distance metric
    * ``'linear'``: use a brute-force approach with linear algebra
    * ``'brute'``: use a brute-force approach
- ``-V``, ``--verbose``: output module information for process
- ``-M X``, ``--mode X``: permissions mode of output HDF5 datasets
