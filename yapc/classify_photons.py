#!/usr/bin/env python
u"""
classify_photons.py
Written by Aimee Gibbons and Tyler Sutterley (05/2021)
Yet Another Photon Classifier for ATL03 Geolocated Photon Data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    scikit-learn: Machine Learning in Python
        http://scikit-learn.org/stable/index.html
        https://github.com/scikit-learn/scikit-learn

UPDATE HISTORY:
    Written 05/2021
"""
import numpy as np
import sklearn.neighbors
import yapc._dist_metrics as _dist_metrics

# PURPOSE: create distance metric for windowed classifier
def windowed_manhattan(u, v, window=[], w=None):
    """
    Create a windowed Manhattan distance metric

    Arguments
    ---------
    u: Input array
    v: Input array for distance

    Keyword arguments
    -----------------
    window: distance window for reducing neighbors
    w: weights for each value
    """
    # verify dimensions
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)
    # calculate Manhattan (rectilinear) distances
    l1_diff = np.abs(u - v)
    # weight differences
    if w is not None:
        w = np.atleast_1d(w)
        l1_diff = w * l1_diff
    # broadcast window to dimensions if using a square window
    window = np.broadcast_to(np.atleast_1d(window),l1_diff.shape)
    for d,wnd in enumerate(window):
        if (l1_diff[d] >= wnd):
            l1_diff[d] = np.inf
    return l1_diff.sum()

# PURPOSE: calculate distances between points as matrices
def distance_matrix(u, v, p=1, window=[]):
    """
    Calculate distances between points as matrices

    Arguments
    ---------
    u: Input array
    v: Input array for distance

    Keyword arguments
    -----------------
    p: power for calculating distance
        1: Manhattan distances
        2: Euclidean distances
    window: distance window for reducing neighbors
    """
    M,s = np.shape(u)
    N,s = np.shape(v)
    # allocate for output distance matrix
    D = np.zeros((M,N))
    # broadcast window to dimensions if using a square window
    window = np.broadcast_to(np.atleast_1d(window),(s,))
    for d in range(s):
        ii, = np.dot(d,np.ones((1,N))).astype(np.int)
        jj, = np.dot(d,np.ones((1,M))).astype(np.int)
        dx = np.abs(u[:,ii] - v[:,jj].T)
        # window differences for dimension
        dx[dx >= window[d]] = np.inf
        # add differences to total distance matrix
        D += np.power(dx,p)
    # convert distances to output units
    return np.power(D,1.0/p)

# PURPOSE: use the GSFC YAPC k-nearest neighbors algorithm to determine
# weights for each photon event within an ATL03 major frame
def classify_photons(x, h, h_win_width, indices, K=5, MIN_PH=5,
    MIN_XSPREAD=1.0, MIN_HSPREAD=0.01, METHOD='linear'):
    """
    Use the NASA GSFC YAPC k-nearest neighbors algorithm to determine
    weights for each photon event within an ATL03 major frame

    Arguments
    ---------
    x: along-track x coordinates for photon events for 3 major frames
    h: photon event heights for 3 major frames
    h_win_width: height of (possibly 2) telemetry bands
    indices: indices of photon events in ATL03 major frame

    Keyword arguments
    -----------------
    K: number of values for KNN algorithm
    MIN_PH: minimum number of photons for a major frame to be valid
    MIN_XSPREAD: minimum along-track spread of photon events
    MIN_HSPREAD: minimum window of heights for photon events
    METHOD: algorithm for computing photon event weights
        `'ball_tree'`: use scikit.learn.BallTree with custom distance metric
        `'linear'`: use a brute-force approach with linear algebra
        `'brute'`: use a brute-force approach
    """
    # number of photon events in a major frame
    n_pe = len(h[indices])
    # output photon weights for major frame
    pe_weights = np.zeros((n_pe))
    # check that number of photons is greater than criteria
    # number of points but be greater than or equal to k
    if (n_pe < MIN_PH) | (n_pe < (K+1)):
        return pe_weights
    # along-track spread of photon events
    xspread = np.max(x[indices]) - np.min(x[indices])
    # height spread of photon events
    hspread = np.max(h[indices]) - np.min(h[indices])
    # check that spread widths are greater than criteria
    if (xspread < MIN_XSPREAD) | (hspread < MIN_HSPREAD):
        return pe_weights
    # photon density
    density = n_pe/(xspread*h_win_width)
    # minimum area to contain minimum number of photon events
    area_min = MIN_PH/density
    # calculate horizontal and vertical window sizes
    win_x = 0.75*MIN_PH*np.sqrt(density)
    win_h = 0.25*MIN_PH*np.sqrt(density)
    # reduce to a buffered window around major frame
    xmin = np.min(x[indices]) - win_x
    xmax = np.max(x[indices]) + win_x
    hmin = np.min(h[indices]) - win_h
    hmax = np.max(h[indices]) + win_h
    iwin, = np.nonzero((x >= xmin) & (x <= xmax) & (h >= hmin) & (h <= hmax))
    # method of calculating photon event weights
    if (METHOD == 'ball_tree'):
        # use BallTree with custom metric to calculate photon event weights
        # window for nearest neighbors
        window = np.array([win_x/2.0,win_h/2.0])
        # create ball tree with photon events in the 3 major frames
        # using a cythonized callable distance metric
        tree = sklearn.neighbors.BallTree(np.c_[x[iwin],h[iwin]],
            metric=_dist_metrics.windowed_manhattan, window=window)
        # K nearest neighbors with windowed manhattan metric
        # use K+1 to remove identity distances (d=0)
        dist,_ = tree.query(np.c_[x[indices],h[indices]], k=K+1,
            return_distance=True)
        # calculate photon event weights and normalize by window size
        valid = np.all(np.isfinite(dist),axis=1)
        inv_dist = np.sum(win_x/2.0 + win_h/2.0 - dist[:,1:],axis=1)
        pe_weights[valid] = inv_dist[valid]/(win_x*win_h)
    elif (METHOD == 'linear'):
        # use brute force with linear algebra to calculate photon event weights
        # window for nearest neighbors
        window = np.array([win_x/2.0,win_h/2.0])
        # calculate distance matrix between points
        dist = distance_matrix(np.c_[x[indices],h[indices]],
            np.c_[x[iwin],h[iwin]], p=1, window=window)
        # sort distances and get K nearest neighbors
        # use K+1 to remove identity distances (d=0)
        dist_sort = np.sort(dist, axis=1)[:,1:K+1]
        # calculate inverse distance of photon events in window
        inv_dist = win_x/2.0 + win_h/2.0 - dist_sort
        # calculate photon event weights and normalize by window size
        valid = np.all(np.isfinite(dist_sort),axis=1)
        pe_weights[valid] = np.sum(inv_dist[valid,:],axis=1)/(win_x*win_h)
    elif (METHOD == 'brute'):
        # use brute force approach to calculate photon event weights
        # for each photon in the major frame
        for j,i in enumerate(indices):
            # all photon events in buffer excluding source photon
            ii = sorted(set(iwin) - set([i]))
            # distance of photon events to source photon
            dx = np.abs(x[ii] - x[i])
            dh = np.abs(h[ii] - h[i])
            # indices of photons within window
            n, = np.nonzero((dx < (win_x/2.0)) & (dh < (win_h/2.0)))
            # skip iteration if there are less than K within window
            if (len(n) < K):
                continue
            # calculate inverse distance of photon events in window
            inv_dist = win_x/2.0 - dx[n] + win_h/2.0 - dh[n]
            # sort distances and get K nearest neighbors
            k_sort = np.argsort(dx[n] + dh[n])[:K]
            # sum of the K largest weights (normalized by the window size)
            pe_weights[j] = np.sum(inv_dist[k_sort])/(win_x*win_h)
    # return the weights for the major frame
    return pe_weights
