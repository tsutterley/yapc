#!/usr/bin/env python
u"""
test_distance.py
Test distance metrics match

Metrics:
    ball_tree: scikit.learn.BallTree with custom distance metric
    linear: brute-force approach with linear algebra
    brute: iterated brute-force approach
"""
import numpy as np
import sklearn.neighbors
import yapc._dist_metrics as _dist_metrics
from yapc.classify_photons import distance_matrix

# PURPOSE: test that distance metrics match
def test_distance():
    # number of photon events
    n_pe = 100
    # photon event for distance
    i = 0
    # along-track x and height
    x = 0.1*np.arange(n_pe)
    h = np.sin(x*np.pi)
    # window
    win_x = 4.0
    win_h = 1.0
    window = np.array([win_x/2.0, win_h/2.0])
    # number of neighbors
    K = 5

    # method 1: ball_tree
    tree = sklearn.neighbors.BallTree(np.c_[x,h],
        metric=_dist_metrics.windowed_manhattan, window=window)
    dist,_ = tree.query(np.c_[x[i],h[i]], k=(K+1), return_distance=True)
    d1 = np.sort(dist[:,1:])
    # method 2: linear
    dist = distance_matrix(np.c_[x,h], np.c_[x[i],h[i]], p=1, window=window)
    d2 = np.sort(dist, axis=0)[1:K+1].T
    # method 3: brute
    # all photon events in buffer excluding source photon
    ii = sorted(set(np.arange(n_pe)) - set([i]))
    # distance of photon events to source photon
    dx = np.abs(x[ii] - x[i])
    dh = np.abs(h[ii] - h[i])
    n, = np.nonzero((dx < (win_x/2.0)) & (dh < (win_h/2.0)))
    d3 = np.sort(dx[n] + dh[n])
    # assert distance metrics are congruent
    assert np.all(np.isclose(d1,d2))
    assert np.all(np.isclose(d2,d3))
    assert np.all(np.isclose(d1,d3))
