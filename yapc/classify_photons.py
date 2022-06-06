#!/usr/bin/env python
u"""
classify_photons.py
Written by Aimee Gibbons and Tyler Sutterley (06/2022)
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
    Updated 06/2022: added option for setting the minimum KNN value
    Updated 04/2022: can weight using only height differences
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
def windowed_manhattan(u, v, window=[], w=[]):
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
    w: float or list, default []
        weights for each value
    """
    # verify dimensions
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)
    # calculate Manhattan (rectilinear) distances
    l1_diff = np.abs(u - v)
    # broadcast window to dimensions if using a square window
    window = np.broadcast_to(np.atleast_1d(window),l1_diff.shape)
    w = np.broadcast_to(np.atleast_1d(w),l1_diff.shape)
    for d,wnd in enumerate(window):
        if (l1_diff[d] >= wnd):
            l1_diff[d] = np.inf
        # scale differences by weights
        with np.errstate(invalid='ignore'):
            l1_diff[d] *= w[d]
    return l1_diff.sum()

# PURPOSE: calculate distances between points as matrices
def distance_matrix(u, v, p=1, window=[], w=[]):
    """
    Calculate distances between two collections of points

    Parameters
    ----------
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
    w: float or list, default []
        weights for each value
    """
    M,s = np.shape(u)
    N,s = np.shape(v)
    # allocate for output distance matrix
    D = np.zeros((M,N))
    # broadcast window to dimensions if using a square window
    window = np.broadcast_to(np.atleast_1d(window),(s,))
    w = np.broadcast_to(np.atleast_1d(w),(s,))
    for d in range(s):
        ii, = np.dot(d,np.ones((1,N))).astype(np.int64)
        jj, = np.dot(d,np.ones((1,M))).astype(np.int64)
        dx = np.abs(u[:,ii] - v[:,jj].T)
        # window differences for dimension
        dx[dx >= window[d]] = np.inf
        # scale differences by weights
        # add differences to total distance matrix
        with np.errstate(invalid='ignore'):
            D += np.power(w[d]*dx, p)
    # convert distances to output units
    return np.power(D, 1.0/p)

# PURPOSE: use the GSFC YAPC k-nearest neighbors algorithm to determine
# weights for each photon event
def classify_photons(x, h, h_win_width, indices, **kwargs):
    """
    Use the NASA GSFC YAPC k-nearest neighbors algorithm to determine
    weights for each photon event

    Parameters
    ----------
    x: float
        along-track x coordinates for photon events
    h: float
        photon event heights
    h_win_width: float
        height of (possibly 2) telemetry bands
    indices: int
        indices of photon events to classify
    K: int, default 0
        number of values for KNN algorithm
    min_knn: int, default 5
        minimum number of values for KNN algorithm
    min_ph: int, default 3
        minimum number of photons to be valid
    min_xspread: float, default 1.0
        minimum along-track spread of photon events
    min_hspread: float, default 0.01
        minimum window of heights for photon events
    win_x: float, default 15.0
        along-track length of window
    win_h: float, default 6.0
        height of window
    aspect: float, default 0.0
        aspect ratio of x and h window
    method: str, default 'linear'
        algorithm for computing photon event weights

            - ``'ball_tree'``: use scikit.learn.BallTree with custom distance metric
            - ``'linear'``: use a brute-force approach with linear algebra
            - ``'brute'``: use a brute-force approach
    metric: str, default 'height'
        metric for computing distances

            - ``'height'``: height differences
            - ``'manhattan'``: manhattan distances
    return_window: bool, default False
        return the width and height of the selection window
    return_K: bool, default False
        return the dynamically selected number of values
    """
    # set default keyword arguments
    kwargs.setdefault('K', 0)
    kwargs.setdefault('min_knn', 5)
    kwargs.setdefault('min_ph', 3)
    kwargs.setdefault('min_xspread', 1.0)
    kwargs.setdefault('min_hspread', 0.01)
    kwargs.setdefault('win_x', 15.0)
    kwargs.setdefault('win_h', 6.0)
    kwargs.setdefault('aspect', 0.0)
    kwargs.setdefault('method', 'linear')
    kwargs.setdefault('metric', 'height')
    kwargs.setdefault('return_window', False)
    kwargs.setdefault('return_K', False)
    # number of photon events
    n_pe = len(h[indices])
    # output photon weights
    pe_weights = np.zeros((n_pe))
    # number of values for KNN algorithm
    if (kwargs['K'] == 0):
        K = np.max([kwargs['min_knn'], np.sqrt(n_pe)/2]).astype(int)
    else:
        K = np.copy(kwargs['K'])
    # check that number of photons is greater than criteria
    # number of points but be greater than or equal to k
    min_ph_check = (n_pe >= kwargs['min_ph']) & (n_pe >= (K+1))
    if np.logical_not(min_ph_check) and kwargs['return_window'] and kwargs['return_K']:
        # return empty weights, window sizes and selection window
        return (pe_weights, 0.0, 0.0, K)
    if np.logical_not(min_ph_check) and kwargs['return_window']:
        # return empty weights and window sizes
        return (pe_weights, 0.0, 0.0)
    if np.logical_not(min_ph_check) and kwargs['return_K']:
        # return empty weights and selection window
        return (pe_weights, K)
    elif np.logical_not(min_ph_check):
        # return empty weights
        return pe_weights
    # along-track spread of photon events
    xspread = np.max(x[indices]) - np.min(x[indices])
    # height spread of photon events
    hspread = np.max(h[indices]) - np.min(h[indices])
    # check that spread widths are greater than criteria
    spread_check = (xspread >= kwargs['min_xspread']) & \
        (hspread >= kwargs['min_hspread']) & \
        (h_win_width >= 0.0)
    if np.logical_not(spread_check) and kwargs['return_window'] and kwargs['return_K']:
        # return empty weights, window sizes and selection window
        return (pe_weights, 0.0, 0.0, K)
    if np.logical_not(spread_check) and kwargs['return_window']:
        # return empty weights and window sizes
        return (pe_weights, 0.0, 0.0)
    if np.logical_not(spread_check) and kwargs['return_K']:
        # return empty weights and selection window
        return (pe_weights, K)
    elif np.logical_not(spread_check):
        # return empty weights
        return pe_weights
    # use pre-defined window size or adaptive window size
    if (kwargs['aspect'] > 0.0):
        # photon density
        density = n_pe/(xspread*h_win_width)
        # minimum area to contain minimum number of photon events
        area_min = kwargs['min_ph']/density
        # minimum length of a square containing minimum number of photons
        length_min = np.sqrt(area_min)
        # calculate horizontal and vertical window sizes with aspect ratio
        # x window length will be aspect times the h window length
        win_x = np.power(kwargs['aspect'], 0.5)*length_min
        win_h = np.power(kwargs['aspect'], -0.5)*length_min
    else:
        # pre-defined window size
        win_x = np.copy(kwargs['win_x'])
        win_h = np.copy(kwargs['win_h'])
    # reduce to a buffered window
    xmin = np.min(x[indices]) - win_x/2.0
    xmax = np.max(x[indices]) + win_x/2.0
    hmin = np.min(h[indices]) - win_h/2.0
    hmax = np.max(h[indices]) + win_h/2.0
    iwin, = np.nonzero((x >= xmin) & (x <= xmax) & (h >= hmin) & (h <= hmax))
    # normalization for weights
    if (kwargs['metric'] == 'height'):
        dist_norm = K*win_h/2.0
        dist_weights = np.array([0.0, 1.0])
    else:
        dist_norm = K*(win_x/2.0 + win_h/2.0)
        dist_weights = np.array([1.0, 1.0])
    # method of calculating photon event weights
    if (kwargs['method'] == 'ball_tree'):
        # use BallTree with custom metric to calculate photon event weights
        # window for nearest neighbors
        window = np.array([win_x/2.0,win_h/2.0])
        # create ball tree with photon events
        # using a cythonized callable distance metric
        tree = sklearn.neighbors.BallTree(np.c_[x[iwin],h[iwin]],
            metric=_dist_metrics.windowed_manhattan, window=window,
            w=dist_weights)
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
            np.c_[x[iwin],h[iwin]], p=1, window=window, w=dist_weights)
        # sort distances and get K nearest neighbors
        # use K+1 to remove identity distances (d=0)
        dist_sort = np.sort(dist, axis=1)[:,1:K+1]
        # calculate inverse distance of photon events in window
        inv_dist = win_x/2.0 + win_h/2.0 - dist_sort
        # calculate photon event weights and normalize by window size
        valid = np.all(np.isfinite(dist_sort),axis=1)
        pe_weights[valid] = np.sum(inv_dist[valid,:], axis=1)/dist_norm
    elif (kwargs['method'] == 'brute'):
        # use brute force approach to calculate photon event weights
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
            # sort distances and get K nearest neighbors
            if (kwargs['metric'] == 'height'):
                inv_dist = win_h/2.0 - dh[n]
                k_sort = np.argsort(dh[n])[:K]
            else:
                inv_dist = win_x/2.0 - dx[n] + win_h/2.0 - dh[n]
                k_sort = np.argsort(dx[n] + dh[n])[:K]
            # sum of the K largest weights (normalized by the window size)
            pe_weights[j] = np.sum(inv_dist[k_sort])/dist_norm
    # check if returning both the weights and auxiliary data
    if kwargs['return_window'] and kwargs['return_K']:
        # return the weights, window size and dynamically selected K
        return (pe_weights, win_x, win_h, K)
    elif kwargs['return_window']:
        # return the weights and window size
        return (pe_weights, win_x, win_h)
    elif kwargs['return_K']:
        # return the weights and dynamically selected K
        return (pe_weights, K)
    else:
        # return the weights
        return pe_weights
