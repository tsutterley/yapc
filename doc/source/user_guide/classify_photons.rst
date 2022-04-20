===================
classify_photons.py
===================

Yet Another Photon Classifier for ATL03 Global Geolocated Photon Data

Calling Sequence
================

.. code-block:: python

    from yapc.classify_photons import classify_photons

    # calculate photon event weights
    pe_weights = classify_photons(x_atc[i1], h_ph[i1],
        h_win_width, i2, K=0, min_ph=3, min_xspread=1.0,
        min_hspread=0.01, win_x=15, win_h=3, method='linear')


`Source code`__

.. __: https://github.com/tsutterley/yapc/blob/main/yapc/classify_photons.py


General Methods
===============

.. autofunction:: yapc.classify_photons.classify_photons

.. autofunction:: yapc.classify_photons.windowed_manhattan

.. autofunction:: yapc.classify_photons.distance_matrix
