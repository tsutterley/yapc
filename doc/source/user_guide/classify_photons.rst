===================
classify_photons.py
===================

Yet Another Photon Classifier for ATL03 Global Geolocated Photon Data

Calling Sequence
================

.. code-block:: python

    from yapc.classify_photons import classify_photons

    # geolocated photon data for beam gtx
    val = IS2_atl03_mds[gtx]
    # ATL03 Segment ID
    Segment_ID[gtx] = val['geolocation']['segment_id']
    # number of photon events
    n_pe, = val['heights']['delta_time'].shape
    # first photon in the segment (convert to 0-based indexing)
    Segment_Index_begin[gtx] = val['geolocation']['ph_index_beg'] - 1
    # number of photon events in the segment
    Segment_PE_count[gtx] = val['geolocation']['segment_ph_cnt'].copy()
    # along-track distance for each ATL03 segment
    Segment_Distance[gtx] = val['geolocation']['segment_dist_x'].copy()

    # along-track and across-track distance for photon events
    x_atc = val['heights']['dist_ph_along'].copy()
    y_atc = val['heights']['dist_ph_across'].copy()
    # photon event heights
    h_ph = val['heights']['h_ph'].copy()
    # flag denoting photon events as possible TEP
    isTEP = (val[gtx]['heights']['quality_ph'] == 3)
    # for each 20m segment
    for j,_ in enumerate(Segment_ID[gtx]):
        # index for 20m segment j
        idx = Segment_Index_begin[gtx][j]
        # skip segments with no photon events
        if (idx < 0):
            continue
        # number of photons in 20m segment
        cnt = Segment_PE_count[gtx][j]
        # add segment distance to along-track coordinates
        x_atc[idx:idx+cnt] += Segment_Distance[gtx][j]

    # iterate over ATLAS major frames
    photon_mframes = val['heights']['pce_mframe_cnt'].copy()
    pce_mframe_cnt = val['bckgrd_atlas']['pce_mframe_cnt'].copy()
    unique_major_frames,unique_index = np.unique(pce_mframe_cnt,return_index=True)
    major_frame_count = len(unique_major_frames)
    tlm_height_band1 = val['bckgrd_atlas']['tlm_height_band1'].copy()
    tlm_height_band2 = val['bckgrd_atlas']['tlm_height_band2'].copy()
    # photon event weights and signal-to-noise ratio
    pe_weights = np.zeros((n_pe),dtype=np.float)
    Segment_Photon_SNR[gtx] = np.zeros((n_pe),dtype=np.int)
    # run for each major frame
    for iteration,idx in enumerate(unique_index):
        # sum of 2 telemetry band widths for major frame
        h_win_width = tlm_height_band1[idx] + tlm_height_band2[idx]
        # photon indices for major frame (buffered by 1 on each side)
        # do not use possible TEP photons in photon classification
        i1, = np.nonzero((photon_mframes >= unique_major_frames[iteration]-1) &
            (photon_mframes <= unique_major_frames[iteration]+1) &
            np.logical_not(isTEP))
        # indices for the major frame within the buffered window
        i2, = np.nonzero(photon_mframes[i1] == unique_major_frames[iteration])
        # calculate photon event weights
        pe_weights[i1[i2]] = classify_photons(x_atc[i1], h_ph[i1],
            h_win_width, i2, K=3, min_ph=3, min_xspread=1.0,
            min_hspread=0.01, method='linear')


`Source code`__

.. __: https://github.com/tsutterley/yapc/blob/main/yapc/classify_photons.py


General Methods
===============

.. autofunction:: yapc.classify_photons.classify_photons

.. autofunction:: yapc.classify_photons.windowed_manhattan

.. autofunction:: yapc.classify_photons.distance_matrix
