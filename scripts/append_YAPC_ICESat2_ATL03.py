#!/usr/bin/env python
u"""
append_YAPC_ICESat2_ATL03.py (09/2021)
Reads ICESat-2 ATL03 data files and appends photon classification flags
    from YAPC (Yet Another Photon Classifier)

CALLING SEQUENCE:
    python append_YAPC_ICESat2_ATL03.py ATL03_file

COMMAND LINE OPTIONS:
    --K X, -k X: number of values for KNN algorithm
    --min-ph X: minimum number of photons for a major frame to be valid
    --min-x-spread X: minimum along-track spread of photon events
    --min-h-spread X: minimum window of heights for photon events
    --aspect X: aspect ratio of x and h window
    --method X: algorithm for computing photon event weights
        `'ball_tree'`: use scikit.learn.BallTree with custom distance metric
        `'linear'`: use a brute-force approach with linear algebra
        `'brute'`: use a brute-force approach
    -V, --verbose: Verbose output to track progress
    -M X, --mode X: Permission mode of files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://h5py.org
    scikit-learn: Machine Learning in Python
        http://scikit-learn.org/stable/index.html
        https://github.com/scikit-learn/scikit-learn

PROGRAM DEPENDENCIES:
    classify_photons.py: Yet Another Photon Classifier for Geolocated Photon Data

UPDATE HISTORY:
    Updated 09/2021: added more YAPC options for photon distance classifier
    Updated 08/2021: using YAPC HDF5 variable names to match ASAS version
    Written 05/2021
"""
from __future__ import print_function

import os
import re
import h5py
import argparse
import numpy as np
from yapc.classify_photons import classify_photons

#-- PURPOSE: reads ICESat-2 ATL03 HDF5 files
#-- computes photon classifications heights over 20m segments
def append_YAPC_ICESat2_ATL03(ATL03_file, **kwargs):
    kwargs.setdefault('K',3)
    kwargs.setdefault('MIN_PH',3)
    kwargs.setdefault('MIN_XSPREAD',1.0)
    kwargs.setdefault('MIN_HSPREAD',0.01)
    kwargs.setdefault('ASPECT',3.0)
    kwargs.setdefault('METHOD','linear')
    kwargs.setdefault('VERBOSE',False)
    kwargs.setdefault('MODE',0o775)

    #-- Open the HDF5 file for appending
    fileID = h5py.File(ATL03_file, 'a')
    #-- output information for file
    print('{0} -->'.format(ATL03_file)) if kwargs['VERBOSE'] else None

    #-- attributes for the output variables
    attrs = dict(yapc_snr_norm={}, yapc_snr={}, yapc_conf={})
    #-- normalization for photon event weights
    attrs['yapc_snr_norm']['units'] = 1
    attrs['yapc_snr_norm']['long_name'] = "Maximum Weight"
    attrs['yapc_snr_norm']['description'] = ("Maximum weight from the photon "
        "event classifier used as normalization for calculating the"
        "signal-to-noise ratio")
    attrs['yapc_snr_norm']['source'] = "YAPC"
    attrs['yapc_snr_norm']['contentType'] = "qualityInformation"
    attrs['yapc_snr_norm']['coordinates'] = ("delta_time reference_photon_lat "
        "reference_photon_lon")
    #-- signal-to-noise ratio for each photon
    attrs['yapc_snr']['units'] = 100
    attrs['yapc_snr']['long_name'] = "Signal-to-Noise Ratio"
    attrs['yapc_snr']['description'] = ("Signal-to-Noise ratio calculated using "
        "the photon event classifier, normalized using the maximum weight "
        "in an ATL03 segment")
    attrs['yapc_snr']['source'] = "YAPC"
    attrs['yapc_snr']['contentType'] = "qualityInformation"
    attrs['yapc_snr']['coordinates'] = "delta_time lat_ph lon_ph"
    #-- photon signal-to-noise confidence from photon classifier
    attrs['yapc_conf']['units'] = 1
    attrs['yapc_conf']['valid_min'] = 0
    attrs['yapc_conf']['valid_max'] = 4
    attrs['yapc_conf']['flag_values'] = [0,2,3,4]
    attrs['yapc_conf']['flag_meanings'] = "noise low medium high"
    attrs['yapc_conf']['long_name'] = "Photon Signal Confidence"
    attrs['yapc_conf']['description'] = ("Confidence level associated with "
        "each photon event selected as signal from the photon classifier "
        "(0=noise; 2=low; 3=med; 4=high).")
    attrs['yapc_conf']['source'] = "YAPC"
    attrs['yapc_conf']['contentType'] = "qualityInformation"
    attrs['yapc_conf']['coordinates'] = "delta_time lat_ph lon_ph"

    #-- read each input beam within the file
    IS2_atl03_beams = []
    for gtx in [k for k in fileID.keys() if bool(re.match(r'gt\d[lr]',k))]:
        #-- check if subsetted beam contains data
        #-- check in both the geolocation and heights groups
        try:
            fileID[gtx]['geolocation']['segment_id']
            fileID[gtx]['heights']['delta_time']
        except KeyError:
            pass
        else:
            IS2_atl03_beams.append(gtx)

    #-- for each input beam within the file
    for gtx in sorted(IS2_atl03_beams):
        print(gtx) if kwargs['VERBOSE'] else None
        #-- ATL03 Segment ID
        Segment_ID = fileID[gtx]['geolocation']['segment_id'][:]
        #-- number of ATL03 20 meter segments
        n_seg = len(Segment_ID)
        #-- number of photon events
        n_pe, = fileID[gtx]['heights']['delta_time'].shape

        #-- first photon in the segment (convert to 0-based indexing)
        Segment_Index_begin = fileID[gtx]['geolocation']['ph_index_beg'][:] - 1
        #-- number of photon events in the segment
        Segment_PE_count = fileID[gtx]['geolocation']['segment_ph_cnt'][:]
        #-- along-track distance for each ATL03 segment
        Segment_Distance = fileID[gtx]['geolocation']['segment_dist_x'][:]

        #-- along-track and across-track distance for photon events
        x_atc = fileID[gtx]['heights']['dist_ph_along'][:].copy()
        #-- photon event heights
        h_ph = fileID[gtx]['heights']['h_ph'][:].copy()
        #-- for each 20m segment
        for j,_ in enumerate(Segment_ID):
            #-- index for 20m segment j
            idx = Segment_Index_begin[j]
            #-- skip segments with no photon events
            if (idx < 0):
                continue
            #-- number of photons in 20m segment
            cnt = Segment_PE_count[j]
            #-- add segment distance to along-track coordinates
            x_atc[idx:idx+cnt] += Segment_Distance[j]

        #-- iterate over ATLAS major frames
        photon_mframes = fileID[gtx]['heights']['pce_mframe_cnt'][:].copy()
        pce_mframe_cnt = fileID[gtx]['bckgrd_atlas']['pce_mframe_cnt'][:].copy()
        unique_major_frames,unique_index = np.unique(pce_mframe_cnt,return_index=True)
        major_frame_count = len(unique_major_frames)
        tlm_height_band1 = fileID[gtx]['bckgrd_atlas']['tlm_height_band1'][:].copy()
        tlm_height_band2 = fileID[gtx]['bckgrd_atlas']['tlm_height_band2'][:].copy()
        #-- photon event weights
        pe_weights = np.zeros((n_pe),dtype=np.float64)
        #-- photon signal-to-noise ratios from classifier
        photon_snr = np.zeros((n_pe),dtype=int)
        #-- photon confidence levels from classifier
        pe_sig_conf = np.zeros((n_pe),dtype=int)
        #-- run for each major frame (distributed over comm.size # of processes)
        for iteration in range(major_frame_count):
            #-- background atlas index for iteration
            idx = unique_index[iteration]
            #-- sum of 2 telemetry band widths for major frame
            h_win_width = tlm_height_band1[idx] + tlm_height_band2[idx]
            #-- photon indices for major frame (buffered by 1 on each side)
            i1, = np.nonzero((photon_mframes >= unique_major_frames[iteration]-1) &
                (photon_mframes <= unique_major_frames[iteration]+1))
            #-- indices for the major frame within the buffered window
            i2, = np.nonzero(photon_mframes[i1] == unique_major_frames[iteration])
            #-- calculate photon event weights
            pe_weights[i1[i2]] = classify_photons(x_atc[i1], h_ph[i1],
                h_win_width, i2, **kwargs)

        #-- for each 20m segment
        snr_norm = np.zeros((n_seg),dtype=np.float64)
        for j,_ in enumerate(Segment_ID):
            #-- index for 20m segment j
            idx = Segment_Index_begin[j]
            #-- skip segments with no photon events
            if (idx < 0):
                continue
            #-- number of photons in 20m segment
            cnt = Segment_PE_count[j]
            #-- photon event weights from photon classifier
            segment_weights = pe_weights[idx:idx+cnt]
            snr_norm[j] = np.max(segment_weights)
            #-- photon event signal-to-noise ratio from photon classifier
            if (snr_norm[j] > 0):
                photon_snr[idx:idx+cnt] = 100.0*segment_weights/snr_norm[j]

        #-- calculate confidence levels from photon classifier
        pe_sig_conf[photon_snr >= 25] = 2
        pe_sig_conf[photon_snr >= 60] = 3
        pe_sig_conf[photon_snr >= 80] = 4

        #-- segment signal-to-noise ratio normalization from photon classifier
        val = '{0}/{1}/{2}'.format(gtx,'geolocation','yapc_snr_norm')
        h5 = fileID.create_dataset(val, np.shape(snr_norm), data=snr_norm,
            dtype=snr_norm.dtype, compression='gzip')
        #-- attach dimensions
        for i,dim in enumerate(['delta_time']):
            h5.dims[i].attach_scale(fileID[gtx]['geolocation'][dim])
        #-- add HDF5 variable attributes
        for att_name,att_val in attrs['yapc_snr_norm'].items():
            h5.attrs[att_name] = att_val

        #-- photon signal-to-noise ratio from photon classifier
        val = '{0}/{1}/{2}'.format(gtx,'heights','yapc_snr')
        h5 = fileID.create_dataset(val, np.shape(photon_snr), data=photon_snr,
            dtype=photon_snr.dtype, compression='gzip')
        #-- attach dimensions
        for i,dim in enumerate(['delta_time']):
            h5.dims[i].attach_scale(fileID[gtx]['heights'][dim])
        #-- add HDF5 variable attributes
        for att_name,att_val in attrs['yapc_snr'].items():
            h5.attrs[att_name] = att_val

        #-- photon signal-to-noise confidence from photon classifier
        val = '{0}/{1}/{2}'.format(gtx,'heights','yapc_conf')
        h5 = fileID.create_dataset(val, np.shape(pe_sig_conf), data=pe_sig_conf,
            dtype=pe_sig_conf.dtype, compression='gzip')
        #-- attach dimensions
        for i,dim in enumerate(['delta_time']):
            h5.dims[i].attach_scale(fileID[gtx]['heights'][dim])
        #-- add HDF5 variable attributes
        for att_name,att_val in attrs['yapc_conf'].items():
            h5.attrs[att_name] = att_val

    #-- close the HDF5 file
    fileID.close()
    #-- change the permissions mode
    os.chmod(ATL03_file, kwargs['MODE'])

#-- Main program that calls append_YAPC_ICESat2_ATL03()
def main():
    #-- Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Reads ICESat-2 ATL03 data files and appends
            photon classification flags from YAPC
            """
    )
    #-- command line parameters
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL03 file to run')
    #-- number of values for KNN algorithm
    parser.add_argument('--K','-k',
        type=int, default=3,
        help='Number of values for KNN algorithm')
    #-- minimum number of photons for a major frame to be valid
    parser.add_argument('--min-ph',
        type=int, default=3,
        help='Minimum number of photons for a major frame to be valid')
    #-- minimum along-track spread of photon events
    parser.add_argument('--min-x-spread',
        type=float, default=1.0,
        help='Minimum along-track spread of photon events')
    #-- minimum window of heights for photon events
    parser.add_argument('--min-h-spread',
        type=float, default=0.01,
        help='Minimum window of heights for photon events')
    #-- aspect ratio of x and h window
    parser.add_argument('--aspect',
        type=float, default=3.0,
        help='Aspect ratio of x and h window')
    #-- algorithm for computing photon event weights
    choices = ('ball_tree','linear','brute')
    parser.add_argument('--method',
        type=str, default='linear', choices=choices,
        help='Algorithm for computing photon event weights')
    #-- verbosity settings
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    #-- permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='permissions mode of output files')
    args = parser.parse_args()

    #-- run the program for the ATL03 file
    for ATL03_file in args.infile:
        append_YAPC_ICESat2_ATL03(ATL03_file,
            K=args.K,
            MIN_PH=args.min_ph,
            MIN_XSPREAD=args.min_x_spread,
            MIN_HSPREAD=args.min_h_spread,
            ASPECT=args.aspect,
            METHOD=args.method,
            VERBOSE=args.verbose,
            MODE=args.mode)

#-- run main program
if __name__ == '__main__':
    main()
