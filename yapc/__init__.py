u"""
Yet Another Photon Classifier (YAPC) for Python
===============================================

yapc contains Python tools for classifying geolocated photon data
from the NASA Ice, Cloud and land Elevation Satellite-2 (ICESat-2)
using a k-Nearest Neighbors (kNN) algorithm

The package works using Python packages (numpy, scikit-learn)

Documentation is available at https:/yapc.readthedocs.io
"""
import yapc._dist_metrics
import yapc.utilities
import yapc.version
from yapc.classify_photons import classify_photons
# set version
__version__ = yapc.version.version
