#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

# PURPOSE: create distance metric for windowed classifier
def windowed_manhattan(u, v, window=None):
    """
    Calculate distances between two collections of points
       using a windowed Manhattan metric

    Arguments
    ---------
    u: float
        First collection of coordinates
    v: float
        Second collection of coordinates
    window: float or NoneType, default None
        Distance window for reducing neighbors
    """
    # calculate manhattan (rectilinear) distances
    cdef double d = 0.0
    d = dist(u, v, window)
    return d
