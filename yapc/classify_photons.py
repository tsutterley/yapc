#!/usr/bin/env python
u"""
classify_photons.py
Written by Aimee Gibbons and Tyler Sutterley (10/2021)
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
    Updated 10/2021: half the perimeter for weighting the distances
        scale weights by the selected number of neighbors (K)
        check that the telemetry band height is positive
    Updated 09/2021: add option for setting aspect ratio of window
        add option to return selection window dimensions
    Updated 08/2021: update algorithm to match current GSFC version
    Updated 05/2021: use int64 to fix numpy deprecation warning
    Written 05/2021
"""
import numpy as np
import sklearn.neighbors
import yapc._dist_metrics as _dist_metrics

# PURPOSE: create distance metric for windowed classifier
def windowed_manhattan(u, v, window=[], w=None):
    """
    Create a windowed Manhattan distance metric

    Parameters
    ----------
    u: float
        Input array
    v: float
        Input array for distance
    window: float or list, default []
        distance window for reducing neighbors
    w: float or NoneType, default None
        weights for each value
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
    Calculate distances between two collections of points

    Arguments
    ---------
    u: float
        First collection of coordinates
    v: float
        Second collection of coordinates
    p: int, default 1
        Power for calculating distance

            - ``1``: Manhattan distances
            - ``2``: Euclidean distances
    window: float or list, default []
        Distance window for reducing neighbors
    """
    M,s = np.shape(u)
    N,s = np.shape(v)
    # allocate for output distance matrix
    D = np.zeros((M,N))
    # broadcast window to dimensions if using a square window
    window = np.broadcast_to(np.atleast_1d(window),(s,))
    for d in range(s):
        ii, = np.dot(d,np.ones((1,N))).astype(np.int64)
        jj, = np.dot(d,np.ones((1,M))).astype(np.int64)
        dx = np.abs(u[:,ii] - v[:,jj].T)
        # window differences for dimension
        dx[dx >= window[d]] = np.inf
        # add differences to total distance matrix
        D += np.power(dx,p)
    # convert distances to output units
    return np.power(D,1.0/p)

# PURPOSE: use the GSFC YAPC k-nearest neighbors algorithm to determine
# weights for each photon event within an ATL03 major frame
def classify_photons(x, h, h_win_width, indices, **kwargs):
    """
    Use the NASA GSFC YAPC k-nearest neighbors algorithm to determine
    weights for each photon event within an ATL03 major frame

    Arguments
    ---------
    x: float
        along-track x coordinates for photon events for 3 major frames
    h: float
        photon event heights for 3 major frames
    h_win_width: float
        height of (possibly 2) telemetry bands
    indices: int
        indices of photon events in ATL03 major frame
    K: int, default 3
        number of values for KNN algorithm
    min_ph: int, default 3
        minimum number of photons for a major frame to be valid
    min_xspread: float, default 1.0
        minimum along-track spread of photon events
    min_hspread: float, default 0.01
        minimum window of heights for photon events
    aspect: float, default 3.0
        aspect ratio of x and h window
    method: str, default 'linear'
        algorithm for computing photon event weights

            - ``'ball_tree'``: use scikit.learn.BallTree with custom distance metric
            - ``'linear'``: use a brute-force approach with linear algebra
            - ``'brute'``: use a brute-force approach
    return_window: bool, default False
        return the width and height of the selection window
    """
    # set default keyword arguments
    kwargs.setdefault('K',3)
    kwargs.setdefault('min_ph',3)
    kwargs.setdefault('min_xspread',1.0)
    kwargs.setdefault('min_hspread',0.01)
    kwargs.setdefault('aspect',3.0)
    kwargs.setdefault('method','linear')
    kwargs.setdefault('return_window',False)
    # number of values for KNN algorithm
    K = np.copy(kwargs['K'])
    # number of photon events in major frame
    n_pe = len(h[indices])
    # output photon weights for major frame
    pe_weights = np.zeros((n_pe))
    # check that number of photons is greater than criteria
    # number of points but be greater than or equal to k
    min_ph_check = (n_pe >= kwargs['min_ph']) & (n_pe >= (K+1))
    if np.logical_not(min_ph_check) and kwargs['return_window']:
        #-- return empty weights and window sizes
        return (pe_weights,0.0,0.0)
    elif np.logical_not(min_ph_check):
        #-- return empty weights
        return pe_weights
    # along-track spread of photon events
    xspread = np.max(x[indices]) - np.min(x[indices])
    # height spread of photon events
    hspread = np.max(h[indices]) - np.min(h[indices])
    # check that spread widths are greater than criteria
    spread_check = (xspread >= kwargs['min_xspread']) & \
        (hspread >= kwargs['min_hspread']) & \
        (h_win_width >= 0.0)
    if np.logical_not(spread_check) and kwargs['return_window']:
        #-- return empty weights and window sizes
        return (pe_weights,0.0,0.0)
    elif np.logical_not(spread_check):
        #-- return empty weights
        return pe_weights
    # photon density
    density = n_pe/(xspread*h_win_width)
    # minimum area to contain minimum number of photon events
    area_min = kwargs['min_ph']/density
    # minimum length of a square containing minimum number of photons
    length_min = np.sqrt(area_min)
    # calculate horizontal and vertical window sizes with aspect ratio
    # x window length will be aspect times the h window length
    win_x = np.power(kwargs['aspect'],0.5)*length_min
    win_h = np.power(kwargs['aspect'],-0.5)*length_min
    # reduce to a buffered window around major frame
    xmin = np.min(x[indices]) - win_x/2.0
    xmax = np.max(x[indices]) + win_x/2.0
    hmin = np.min(h[indices]) - win_h/2.0
    hmax = np.max(h[indices]) + win_h/2.0
    iwin, = np.nonzero((x >= xmin) & (x <= xmax) & (h >= hmin) & (h <= hmax))
    # normalization for weights
    dist_norm = K*(win_x/2.0 + win_h/2.0)
    # method of calculating photon event weights
    if (kwargs['method'] == 'ball_tree'):
        # use BallTree with custom metric to calculate photon event weights
        # window for nearest neighbors
        window = np.array([win_x/2.0,win_h/2.0])
        # create ball tree with photon events in the buffered major frame
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
        pe_weights[valid] = inv_dist[valid]/dist_norm
    elif (kwargs['method'] == 'linear'):
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
        pe_weights[valid] = np.sum(inv_dist[valid,:],axis=1)/dist_norm
    elif (kwargs['method'] == 'brute'):
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
            pe_weights[j] = np.sum(inv_dist[k_sort])/dist_norm
    # check if returning both the weights and the window size
    if kwargs['return_window']:
        # return the weights and window size for the major frame
        return (pe_weights,win_x,win_h)
    else:
        # return the weights for the major frame
        return pe_weights
