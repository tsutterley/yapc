#!/usr/bin/env python
u"""
append_YAPC_ICESat2_ATL03.py (10/2021)
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
    -O X, --output X: output file type
        append: add photon classification flags to original ATL03 file
        copy: add photon classification flags to a copied ATL03 file
        reduce: create a new file with the photon classification flags
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
    Updated 10/2021: using python logging for handling verbose output
        do not use possible TEP photons in YAPC calculation
        using new YAPC HDF5 variable names to match ASAS version
        added parsing for converting file lines to arguments
    Updated 09/2021: added more YAPC options for photon distance classifier
        create YAPC group and output major frame variables to HDF5
        add output options for creating copied or reduced files
    Updated 08/2021: using YAPC HDF5 variable names to match ASAS version
    Written 05/2021
"""
from __future__ import print_function

import os
import re
import h5py
import yapc
import logging
import argparse
import numpy as np

#-- PURPOSE: reads ICESat-2 ATL03 HDF5 files
#-- computes photon classifications heights over 20m segments
def append_YAPC_ICESat2_ATL03(input_file, **kwargs):
    kwargs.setdefault('K',3)
    kwargs.setdefault('min_ph',3)
    kwargs.setdefault('min_xspread',1.0)
    kwargs.setdefault('min_hspread',0.01)
    kwargs.setdefault('aspect',3.0)
    kwargs.setdefault('method','linear')
    kwargs.setdefault('return_window',True)
    kwargs.setdefault('output','append')
    kwargs.setdefault('verbose',False)
    kwargs.setdefault('mode',0o775)

    #-- create logger for verbosity level
    loglevel = logging.INFO if kwargs['verbose'] else logging.CRITICAL
    logger = yapc.utilities.build_logger('yapc',level=loglevel)

    #-- compile regular expression operator for extracting data from ATL03 files
    rx = re.compile(r'(processed_)?(ATL\d+)_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})'
        r'(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    #-- read ICESat-2 ATL03 HDF5 files (extract base parameters)
    SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX=rx.findall(input_file).pop()

    #-- input/output directory
    directory = os.path.dirname(input_file)
    #-- Open the input HDF5 file for reading or appending
    assert kwargs['output'] in ('append','copy','reduce')
    mode = 'a' if (kwargs['output'] == 'append') else 'r'
    f_in = h5py.File(input_file, mode=mode)

    #-- output information for file
    logger.info('{0} -->'.format(input_file))

    #-- attributes for the output variables
    attrs = {}
    #-- major frame delta_time
    attrs['delta_time'] = {}
    attrs['delta_time']['units'] = "seconds since 2018-01-01"
    attrs['delta_time']['long_name'] = "Elapsed GPS seconds"
    attrs['delta_time']['standard_name'] = "time"
    attrs['delta_time']['calendar'] = "standard"
    attrs['delta_time']['description'] = ("Number of GPS seconds since the "
        "ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
        "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number "
        "of GPS seconds between the GPS epoch (1980-01-06T00:00:00.000000Z "
        "UTC) and the ATLAS SDP epoch. By adding the offset contained within "
        "atlas_sdp_gps_epoch to delta time parameters, the time in gps_seconds "
        "relative to the GPS epoch can be computed.")
    attrs['delta_time']['contentType'] = "physicalMeasurement"
    #-- index of first photon in each major frame
    attrs['ph_index_beg'] = {}
    attrs['ph_index_beg']['units'] = 1
    attrs['ph_index_beg']['long_name'] = "Photon Index Begin"
    attrs['ph_index_beg']['description'] = ("Index (1-based) of first photon "
        "within each major frame. Use in conjunction with mf_ph_cnt.")
    attrs['ph_index_beg']['source'] = "derived"
    attrs['ph_index_beg']['contentType'] = "referenceInformation"
    attrs['ph_index_beg']['coordinates'] = "delta_time"
    #-- number of photons in each major frame
    attrs['mf_ph_cnt'] = {}
    attrs['mf_ph_cnt']['units'] = 1
    attrs['mf_ph_cnt']['long_name'] = "Number of photons"
    attrs['mf_ph_cnt']['description'] = ("Number of photon events in "
        "each major frame")
    attrs['mf_ph_cnt']['source'] = "derived"
    attrs['mf_ph_cnt']['contentType'] = "referenceInformation"
    attrs['mf_ph_cnt']['coordinates'] = "delta_time"
    #-- width in meters of the selection window
    attrs['win_x'] = {}
    attrs['win_x']['units'] = "meters"
    attrs['win_x']['long_name'] = "Window width"
    attrs['win_x']['description'] = "Width in meters of the selection window"
    attrs['win_x']['source'] = "YAPC"
    attrs['win_x']['contentType'] = "referenceInformation"
    attrs['win_x']['coordinates'] = "delta_time"
    #-- height in meters of the selection window
    attrs['win_h'] = {}
    attrs['win_h']['units'] = "meters"
    attrs['win_h']['long_name'] = "Window height"
    attrs['win_h']['description'] = "Height in meters of the selection window"
    attrs['win_h']['source'] = "YAPC"
    attrs['win_h']['contentType'] = "referenceInformation"
    attrs['win_h']['coordinates'] = "delta_time"
    #-- normalization for photon event weights
    attrs['weight_ph_norm'] = {}
    attrs['weight_ph_norm']['units'] = 1
    attrs['weight_ph_norm']['long_name'] = "Maximum Weight"
    attrs['weight_ph_norm']['description'] = ("Maximum weight from the photon "
        "event classifier used as normalization for calculating the"
        "signal-to-noise ratio")
    attrs['weight_ph_norm']['source'] = "YAPC"
    attrs['weight_ph_norm']['contentType'] = "qualityInformation"
    attrs['weight_ph_norm']['coordinates'] = ("delta_time reference_photon_lat "
        "reference_photon_lon")
    #-- signal-to-noise ratio for each photon
    attrs['weight_ph'] = {}
    attrs['weight_ph']['units'] = 100
    attrs['weight_ph']['long_name'] = "Signal-to-Noise Ratio"
    attrs['weight_ph']['description'] = ("Signal-to-Noise ratio calculated using "
        "the photon event classifier, normalized using the maximum weight "
        "in an ATL03 segment")
    attrs['weight_ph']['source'] = "YAPC"
    attrs['weight_ph']['contentType'] = "qualityInformation"
    attrs['weight_ph']['coordinates'] = "delta_time lat_ph lon_ph"
    #-- photon signal-to-noise confidence from photon classifier
    BACKG,L_CONF,M_CONF,H_CONF = (0.0,25.0,60.0,80.0)
    attrs['yapc_conf'] = {}
    attrs['yapc_conf']['units'] = 1
    attrs['yapc_conf']['valid_min'] = 0
    attrs['yapc_conf']['valid_max'] = 4
    attrs['yapc_conf']['flag_values'] = [0,2,3,4]
    attrs['yapc_conf']['confidences'] = [BACKG,L_CONF,M_CONF,H_CONF]
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
    for gtx in [k for k in f_in.keys() if bool(re.match(r'gt\d[lr]',k))]:
        #-- check if subsetted beam contains data
        #-- check in both the geolocation and heights groups
        try:
            f_in[gtx]['geolocation']['segment_id']
            f_in[gtx]['heights']['delta_time']
        except KeyError:
            pass
        else:
            IS2_atl03_beams.append(gtx)

    #-- open output file
    if (kwargs['output'] == 'append'):
        #-- copy filename and file object for output variables
        output_file = input_file
        f_out = f_in
    elif (kwargs['output'] == 'copy'):
        #-- Copy input file to a new appended ATL03 file
        fargs=(SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
        file_format='{0}{1}_{2}{3}{4}{5}{6}{7}_{8}{9}{10}_{11}_{12}{13}_YAPC.h5'
        output_file = os.path.join(directory,file_format.format(*fargs))
        yapc.utilities.copy(input_file,output_file)
        #-- Open the output HDF5 files for appending
        f_out = h5py.File(output_file, 'a')
    elif (kwargs['output'] == 'reduce'):
        #-- create a new reduced file with only YAPC parameters
        fargs=(SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
        file_format='{0}{1}_YAPC_{2}{3}{4}{5}{6}{7}_{8}{9}{10}_{11}_{12}{13}.h5'
        output_file = os.path.join(directory,file_format.format(*fargs))
        #-- Open the output HDF5 files for writing
        f_out = h5py.File(output_file, 'w')
        #-- copy file-level attributes from input to output
        for att_name,att_val in f_in.attrs.items():
            f_out.attrs[att_name] = att_val
        #-- for each valid beam group in the input file
        for gtx in IS2_atl03_beams:
            copy_ATL03_beam_group(f_in, f_out, gtx)

    #-- print output file if verbose
    logger.info('{0} ({1})'.format(output_file,kwargs['output']))

    #-- for each input beam within the file
    for gtx in sorted(IS2_atl03_beams):
        logger.info(gtx)
        #-- ATL03 Segment ID
        Segment_ID = f_in[gtx]['geolocation']['segment_id'][:]
        #-- number of ATL03 20 meter segments
        n_seg = len(Segment_ID)
        #-- number of photon events
        n_pe, = f_in[gtx]['heights']['delta_time'].shape

        #-- first photon in the segment (convert to 0-based indexing)
        Segment_Index_begin = f_in[gtx]['geolocation']['ph_index_beg'][:] - 1
        #-- number of photon events in the segment
        Segment_PE_count = f_in[gtx]['geolocation']['segment_ph_cnt'][:]
        #-- along-track distance for each ATL03 segment
        Segment_Distance = f_in[gtx]['geolocation']['segment_dist_x'][:]

        #-- along-track and across-track distance for photon events
        x_atc = f_in[gtx]['heights']['dist_ph_along'][:].copy()
        #-- photon event heights
        h_ph = f_in[gtx]['heights']['h_ph'][:].copy()
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
        photon_mframes = f_in[gtx]['heights']['pce_mframe_cnt'][:].copy()
        #-- background ATLAS group variables are based upon 50-shot summations
        #-- PCE Major Frames are based upon 200-shot summations
        pce_mframe_cnt = f_in[gtx]['bckgrd_atlas']['pce_mframe_cnt'][:].copy()
        #-- find unique major frames and their indices within background ATLAS group
        #-- (there will 4 background ATLAS time steps for nearly every major frame)
        unique_major_frames,unique_index = np.unique(pce_mframe_cnt,return_index=True)
        #-- number of unique major frames in granule for beam
        major_frame_count = len(unique_major_frames)
        #-- height of each telemetry band for a major frame
        tlm_height_band1 = f_in[gtx]['bckgrd_atlas']['tlm_height_band1'][:].copy()
        tlm_height_band2 = f_in[gtx]['bckgrd_atlas']['tlm_height_band2'][:].copy()
        #-- delta times of each major frame
        delta_time = f_in[gtx]['bckgrd_atlas']['delta_time'][:].copy()
        #-- flag denoting photon events as possible TEP
        if (int(RL) < 4):
            isTEP = np.any((f_in[gtx]['heights']['signal_conf_ph'][:]==-2),axis=1)
        else:
            isTEP = (f_in[gtx]['heights']['quality_ph'][:] == 3)

        #-- photon event variables
        heights = {}
        #-- photon event weights
        pe_weights = np.zeros((n_pe),dtype=np.float64)
        #-- photon signal-to-noise ratios from classifier
        heights['weight_ph'] = np.zeros((n_pe),dtype=np.uint8)
        #-- photon confidence levels from classifier
        heights['yapc_conf'] = np.zeros((n_pe),dtype=np.int8)

        #-- output major frame variables
        yapc_window = {}
        #-- selection window sizes
        yapc_window['win_x'] = np.zeros((major_frame_count))
        yapc_window['win_h'] = np.zeros((major_frame_count))
        #-- index of first photon in major frame
        yapc_window['ph_index_beg'] = np.zeros((major_frame_count),dtype=int)
        #-- number of photons in major frame
        yapc_window['mf_ph_cnt'] = np.zeros((major_frame_count),dtype=int)
        #-- average delta time of major frame
        yapc_window['delta_time'] = np.zeros((major_frame_count))

        #-- run for each major frame
        for i in range(major_frame_count):
            #-- background atlas index for iteration
            idx = unique_index[i]
            #-- sum of 2 telemetry band widths for major frame
            h_win_width = tlm_height_band1[idx] + tlm_height_band2[idx]
            #-- calculate average delta time of major frame
            yapc_window['delta_time'][i] = np.mean(delta_time[idx])
            #-- photon indices for major frame (buffered by 1 on each side)
            #-- do not use possible TEP photons in photon classification
            i1, = np.nonzero((photon_mframes >= unique_major_frames[i]-1) &
                (photon_mframes <= unique_major_frames[i]+1) &
                np.logical_not(isTEP))
            #-- indices for the major frame within the buffered window
            i2, = np.nonzero(photon_mframes[i1] == unique_major_frames[i])
            #-- number of photons in major frame
            yapc_window['mf_ph_cnt'][i] = len(np.atleast_1d(i2))
            #-- check if there are (non-TEP) photons in major frame
            if (yapc_window['mf_ph_cnt'][i] > 0):
                #-- calculate photon event weights
                pe_weights[i1[i2]],win_x,win_h = yapc.classify_photons(
                    x_atc[i1], h_ph[i1], h_win_width, i2, **kwargs)
                #-- index of first photon in major frame (1-based)
                yapc_window['ph_index_beg'][i] = np.atleast_1d(i1[i2])[0] + 1
                #-- selection window sizes
                yapc_window['win_x'][i] = np.copy(win_x)
                yapc_window['win_h'][i] = np.copy(win_h)
            else:
                #-- print debug message
                logger.debug('No photons in major frame {0:d}'.format(i))

        #-- photon event weights scaled to a single byte
        heights['weight_ph'] = 255*pe_weights
        #-- verify photon event weights
        np.clip(heights['weight_ph'], 0, 255, out=heights['weight_ph'])

        #-- for each 20m segment
        weight_ph_norm = np.zeros((n_seg),dtype=np.uint8)
        for j,_ in enumerate(Segment_ID):
            #-- index for 20m segment j
            idx = Segment_Index_begin[j]
            #-- skip segments with no photon events
            if (idx < 0):
                continue
            #-- number of photons in 20m segment
            cnt = Segment_PE_count[j]
            #-- scaled photon event weights from photon classifier
            segment_weights = heights['weight_ph'][idx:idx+cnt]
            #-- verify segment weights
            np.clip(segment_weights, 0, 255, out=segment_weights)
            #-- calculate normalization for 20m segment
            weight_ph_norm[j] = np.max(segment_weights).astype(np.uint8)
            #-- photon event signal-to-noise ratio from photon classifier
            if (weight_ph_norm[j] > 0):
                #-- calculate PE signal-to-noise ratio
                scaled_SNR = (100.0*segment_weights)/weight_ph_norm[j]
                #-- verify PE SNR values and add to output array
                np.clip(scaled_SNR, 0, 100, out=scaled_SNR)
                #-- calculate confidence levels from photon classifier
                segment_class = np.zeros((cnt),dtype=np.int8)
                segment_class[scaled_SNR >= L_CONF] = 2
                segment_class[scaled_SNR >= M_CONF] = 3
                segment_class[scaled_SNR >= H_CONF] = 4
                #-- copy segment classification to output heights variable
                heights['yapc_conf'][idx:idx+cnt] = np.copy(segment_class)

        #-- major frame variables
        f_out[gtx].create_group('yapc_window')
        f_out[gtx]['yapc_window'].attrs['description'] = ('Dynamic '
            'selection window parameters for photon classifier')
        f_out[gtx]['yapc_window'].attrs['version'] = yapc.version.full_version
        for att_name in ['K','min_ph','min_xspread','min_hspread','aspect']:
            f_out[gtx]['yapc_window'].attrs[att_name] = kwargs[att_name]

        #-- major frame delta_time
        #-- index of first photon in each major frame
        #-- number of photons in each major frame
        #-- width in meters of the selection window
        #-- height in meters of the selection window
        for key in ['delta_time','ph_index_beg','mf_ph_cnt','win_x','win_h']:
            val = '{0}/{1}/{2}'.format(gtx,'yapc_window',key)
            logger.info(val)
            h5 = f_out.create_dataset(val, np.shape(yapc_window[key]),
                data=yapc_window[key], dtype=yapc_window[key].dtype,
                compression='gzip')
            #-- create or attach dimensions
            if (key == 'delta_time'):
                #-- make dimension
                h5.make_scale('delta_time')
            else:
                #-- attach dimensions
                for i,dim in enumerate(['delta_time']):
                    h5.dims[i].attach_scale(f_out[gtx]['yapc_window'][dim])
            #-- add HDF5 variable attributes
            for att_name,att_val in attrs[key].items():
                h5.attrs[att_name] = att_val

        #-- segment signal-to-noise ratio normalization from photon classifier
        val = '{0}/{1}/{2}'.format(gtx,'geolocation','weight_ph_norm')
        logger.info(val)
        h5 = f_out.create_dataset(val, np.shape(weight_ph_norm),
            data=weight_ph_norm, dtype=weight_ph_norm.dtype,
            compression='gzip')
        #-- attach dimensions
        for i,dim in enumerate(['delta_time']):
            h5.dims[i].attach_scale(f_out[gtx]['geolocation'][dim])
        #-- add HDF5 variable attributes
        for att_name,att_val in attrs['weight_ph_norm'].items():
            h5.attrs[att_name] = att_val

        #-- photon signal-to-noise ratio from photon classifier
        for key in ['weight_ph','yapc_conf']:
            val = '{0}/{1}/{2}'.format(gtx,'heights',key)
            logger.info(val)
            h5 = f_out.create_dataset(val, np.shape(heights[key]),
                data=heights[key], dtype=heights[key].dtype,
                compression='gzip')
            #-- attach dimensions
            for i,dim in enumerate(['delta_time']):
                h5.dims[i].attach_scale(f_out[gtx]['heights'][dim])
            #-- add HDF5 variable attributes
            for att_name,att_val in attrs[key].items():
                h5.attrs[att_name] = att_val

    #-- close the HDF5 files
    f_in.close()
    f_out.close()
    #-- change the permissions mode
    os.chmod(output_file, kwargs['mode'])

#-- PURPOSE: copy ATL03 variables and attributes to output file
def copy_ATL03_beam_group(f_in, f_out, gtx):
    #-- group variables to copy for each beam group
    groups = {}
    groups['geolocation'] = ['delta_time','segment_id',
        'reference_photon_lat','reference_photon_lon']
    groups['heights'] = ['delta_time','lat_ph','lon_ph']
    #-- attributes to not copy from input HDF5 file
    invalid_attributes = ['CLASS','DIMENSION_LIST','NAME','REFERENCE_LIST']
    #-- create the beam group
    f_out.create_group(gtx)
    #-- copy group attributes from input to output
    for att_name,att_val in f_in[gtx].attrs.items():
        f_out[gtx].attrs[att_name] = att_val
    #-- create data groups and copy necessary variables
    for group,keys in groups.items():
        f_out[gtx].create_group(group)
        #-- for each variable to copy from the input file
        for key in keys:
            val = '{0}/{1}/{2}'.format(gtx,group,key)
            h5 = f_out.create_dataset(val, data=f_in[val],
                dtype=f_in[val].dtype, compression='gzip')
            #-- create or attach dimensions
            if (key == 'delta_time'):
                #-- make dimension
                h5.make_scale('delta_time')
            else:
                #-- attach dimensions
                for i,dim in enumerate(['delta_time']):
                    h5.dims[i].attach_scale(f_out[gtx][group][dim])
            #-- copy group attributes from input to output
            for att_name,att_val in f_in[val].attrs.items():
                if att_name not in invalid_attributes:
                    f_out[val].attrs[att_name] = att_val

#-- Main program that calls append_YAPC_ICESat2_ATL03()
def main():
    #-- Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Reads ICESat-2 ATL03 data files and appends
            photon classification flags from YAPC
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = yapc.utilities.convert_arg_line_to_args
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
        type=str.lower, default='linear', choices=choices,
        help='Algorithm for computing photon event weights')
    #-- output file type (appended, copied or reduced)
    choices = ('append','copy','reduce')
    parser.add_argument('--output','-O',
        type=str.lower, default='append', choices=choices,
        help='Output file type')
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
            min_ph=args.min_ph,
            min_xspread=args.min_x_spread,
            min_hspread=args.min_h_spread,
            aspect=args.aspect,
            method=args.method,
            output=args.output,
            verbose=args.verbose,
            mode=args.mode)

#-- run main program
if __name__ == '__main__':
    main()
