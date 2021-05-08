============================
append_YAPC_ICESat2_ATL03.py
============================

- Reads ICESat-2 `ATL03 geolocated photon height product files <https://nsidc.org/data/ATL03>`_ and appends photon classification flags from YAPC (*Yet Another Photon Classifier*)

    * ``snr_norm_ph``: segment level photon weight normalization
    * ``snr_ph``: the photon level normalized YAPC signal-to-noise ratio
    * ``snr_conf_ph``: YAPC-based confidence levels

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

- ``-V``, ``--verbose``: output module information for process
- ``-M X``, ``--mode X``: permissions mode of output HDF5 datasets
