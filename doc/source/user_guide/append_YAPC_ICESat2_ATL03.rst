============================
append_YAPC_ICESat2_ATL03.py
============================

- Reads ICESat-2 `ATL03 geolocated photon height product files <https://nsidc.org/data/ATL03>`_ and appends photon classification flags from YAPC (*Yet Another Photon Classifier*) to granules

    * ``weight_ph_norm``: photon weight normalization
    * ``weight_ph``: normalized YAPC signal-to-noise ratio for each photon event
    * ``yapc_conf``: YAPC-based confidence levels for each photon event

`Source code`__

.. __: https://github.com/tsutterley/yapc/blob/main/scripts/append_YAPC_ICESat2_ATL03.py

Calling Sequence
################

.. argparse::
    :filename: ../../scripts/append_YAPC_ICESat2_ATL03.py
    :func: arguments
    :prog: append_YAPC_ICESat2_ATL03.py
    :nodescription:
    :nodefault:

    --K -k : @after
        * Use ``0`` for dynamic selection of neighbors

    --aspect : @after
        * Use ``0`` for pre-defined window dimensions

    --method : @after
        * ``'ball_tree'``: use ``scikit.learn.BallTree`` with custom distance metric
        * ``'linear'``: use a brute-force approach with linear algebra
        * ``'brute'``: use a brute-force approach

    --metric : @after
        * ``'height'``: height differences
        * ``'manhattan'``: manhattan distances

    --norm : @after
        * ``'segment'``: normalize photon weights for each segment
        * ``'granule'``: normalize photon weights over complete granule

    --output -O : @after
        * ``'append'``: add photon classification flags to original ATL03 file
        * ``'copy'``: add photon classification flags to a copied ATL03 file
        * ``'reduce'``: create a new file with the photon classification flags
