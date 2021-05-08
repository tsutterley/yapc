=============
YAPC Overview
=============

``YAPC`` is a prototype photon classifer for the NASA ICESat-2
ATL03 Global Geolocated Photon product.
It was developed by Jeff Lee (GSFC) with the goal of supporting and
simplifying science applications for the NASA Ice Cloud and
land Elevation Satellite-2 (ICESat-2).

``YAPC`` is a customized inverse-distance kNN algorithm developed to
determine the significance (or weight) of individual photon events.
The weight of each photon is indicative of localized density based
on it surrounding K neighbors and the inverse distances.
For this prototype product:

- Signal confidence is no longer surface-type specific
- There is no histogramming or linear fitting of the photon events
- There is no penalty for sloped surfaces

YAPC Goals
==========

- Separate photon events that are likely signal from likely noise
- Improve processing efficiency compared to the current histogram-based algorithm
- Be significantly easier to maintain and extend for products and applications
- Reduce the amount of sensitivity studies required for analysis
- Possibly reduce the size of ATL03 product files

YAPC Algorithm
==============

For each major frame:

- Calculate ``h_win_width`` as the sum of the (possibly) 2 telemetry band widths
- Initialize photon weights to 0.0 and perform initial checks

  * Number of photons >= ``MIN_PH``
  * Along-track spread >= ``MIN_XSPREAD``
  * Height spread >= ``MIN_HSPREAD``
- Calculate the size of a dynamic selection window

  * Calculate ``density`` using the number of photons in the major frame, ``h_win_width`` and the span of the along-track distance
  * From the ``density``, calculate the area necessary to contain at least ``MIN_PH`` photons
  * Calculate the horizontal (``win_x``) and vertical (``win_h``) window sizes
- For each source photon in the center major frame, calculate inverse distances from its neighbors
- Select and count target photons whose along-track distance and height are within ``win_x/2`` and ``win_h/2`` of the source photon
- Calculate and record the inverse distances for the ``k`` nearest photons

.. code-block:: python

    inv_dist =  win_x/2 - abs(delta(x)) + win_h/2 - abs(delta(h))

- Calculate the weight of each source photon as the sum of the ``k`` largest ``inv_dist`` values
- Divide the weight by the product of the window sizes ``(win_x * win_h)``

For the python version, the distances between photons can be calculated using:

- `Balltree data structures from scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html>`_
- Brute force with a linear algebra approach
- Brute force with an iterative approach

Credits
=======
These notes were modified from a presentation by Jeff Lee to the ICESat-2 Land Ice Science Team.