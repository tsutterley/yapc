#!/usr/bin/env python
u"""
append_YAPC_ICESat2_ATL03.py
Written by Aimee Gibbons and Tyler Sutterley (06/2022)
Reads ICESat-2 ATL03 data files and appends photon classification flags
    from YAPC (Yet Another Photon Classifier) to granules

CALLING SEQUENCE:
    python append_YAPC_ICESat2_ATL03.py ATL03_file

COMMAND LINE OPTIONS:
    --K X, -k X: number of values for KNN algorithm
        Use 0 for dynamic selection of neighbors
    --min-knn: Minimum value of K used in the KNN algorithm
    --min-ph X: minimum number of photons for a segment to be valid
    --min-x-spread X: minimum along-track spread of photon events
    --min-h-spread X: minimum window of heights for photon events
    --win_x X: along-track length of window
    --win_h X: height of window
    --aspect X: aspect ratio of x and h window
        Use 0 for pre-defined window dimensions
    --method X: algorithm for computing photon event weights
        `'ball_tree'`: use scikit.learn.BallTree with custom distance metric
        `'linear'`: use a brute-force approach with linear algebra
        `'brute'`: use a brute-force approach
    --metric X: metric for computing distances
        `'height'`: height differences
        `'manhattan'`: manhattan distances
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
    Updated 06/2022: can normalize weights over entire ATL03 granule
        added option for setting the minimum KNN value
    Updated 05/2022: use argparse descriptions within sphinx documentation
    Updated 04/2022: calculate weights for each ATL03 segment
    Updated 10/2021: using python logging for handling verbose output
        do not use possible TEP photons in YAPC calculation
        using new YAPC HDF5 variable names to match ASAS version
        added parsing for converting file lines to arguments
        check that given telemetry bands contain at least 1 DEM value
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

# PURPOSE: finds the vertical extent of the photons
def h_extent(h_ph):
  """Find the vertical extent of the photons
  """
  # Calculate bin edges needed for 1 meter bins
  n_bins = np.ceil(np.ptp(h_ph))
  if (n_bins <= 0):
      return 0
  #  construct a histogram
  hist,_ = np.histogram(h_ph, bins=int(n_bins))
  # Return the number of nonzero bins.
  return np.count_nonzero(hist)

# PURPOSE: reads ICESat-2 ATL03 HDF5 files
# computes photon classifications heights over 20m segments
def append_YAPC_ICESat2_ATL03(input_file, output='append', verbose=False,
    mode=0o775, **kwargs):
    # set default keyword arguments for photon classification
    kwargs.setdefault('K',0)
    kwargs.setdefault('min_knn', 5)
    kwargs.setdefault('min_ph', 3)
    kwargs.setdefault('min_xspread', 1.0)
    kwargs.setdefault('min_hspread', 0.01)
    kwargs.setdefault('win_x', 15.0)
    kwargs.setdefault('win_h', 6.0)
    kwargs.setdefault('aspect', 0.0)
    kwargs.setdefault('method', 'linear')
    kwargs.setdefault('metric', 'height')
    kwargs.setdefault('norm', 'segment')
    kwargs.setdefault('return_window', True)
    kwargs.setdefault('return_K', True)

    # create logger for verbosity level
    loglevel = logging.INFO if verbose else logging.CRITICAL
    logger = yapc.utilities.build_logger('yapc',level=loglevel)

    # compile regular expression operator for extracting data from ATL03 files
    rx = re.compile(r'(processed_)?(ATL\d+)_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})'
        r'(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    # read ICESat-2 ATL03 HDF5 files (extract base parameters)
    SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX=rx.findall(input_file).pop()

    # input/output directory
    directory = os.path.dirname(input_file)
    # Open the input HDF5 file for reading or appending
    assert output in ('append','copy','reduce')
    clobber = 'a' if (output == 'append') else 'r'
    f_in = h5py.File(input_file, mode=clobber)

    # output information for file
    logger.info('{0} -->'.format(input_file))

    # attributes for the output variables
    attrs = {}
    # width in meters of the selection window
    attrs['win_x'] = {}
    attrs['win_x']['units'] = "meters"
    attrs['win_x']['long_name'] = "Window width"
    attrs['win_x']['description'] = "Width in meters of the selection window"
    attrs['win_x']['source'] = "YAPC"
    attrs['win_x']['contentType'] = "referenceInformation"
    attrs['win_x']['coordinates'] = "delta_time"
    # height in meters of the selection window
    attrs['win_h'] = {}
    attrs['win_h']['units'] = "meters"
    attrs['win_h']['long_name'] = "Window height"
    attrs['win_h']['description'] = "Height in meters of the selection window"
    attrs['win_h']['source'] = "YAPC"
    attrs['win_h']['contentType'] = "referenceInformation"
    attrs['win_h']['coordinates'] = "delta_time"
    # height extent
    attrs['h_win_width'] = {}
    attrs['h_win_width']['units'] = "meters"
    attrs['h_win_width']['long_name'] = "Height extent"
    attrs['h_win_width']['description'] = "Vertical extent of the photons"
    attrs['h_win_width']['source'] = "YAPC"
    attrs['h_win_width']['contentType'] = "referenceInformation"
    attrs['h_win_width']['coordinates'] = "delta_time"
    # dynamically selected number of neighbors
    attrs['K'] = {}
    attrs['K']['units'] = "1"
    attrs['K']['long_name'] = "K"
    attrs['K']['description'] = "Dynamically selected number of neighbors"
    attrs['K']['source'] = "YAPC"
    attrs['K']['valid_min'] = kwargs['min_knn']
    attrs['K']['contentType'] = "referenceInformation"
    attrs['K']['coordinates'] = "delta_time"
    # normalization for photon event weights
    attrs['weight_ph_norm'] = {}
    attrs['weight_ph_norm']['units'] = 1
    attrs['weight_ph_norm']['long_name'] = "Maximum Weight"
    attrs['weight_ph_norm']['description'] = ("Maximum weight from the photon "
        "event classifier used as normalization for calculating the"
        "signal-to-noise ratio")
    attrs['weight_ph_norm']['source'] = "YAPC"
    attrs['weight_ph_norm']['normalization'] = kwargs['norm']
    attrs['weight_ph_norm']['contentType'] = "qualityInformation"
    attrs['weight_ph_norm']['coordinates'] = ("delta_time reference_photon_lat "
        "reference_photon_lon")
    # signal-to-noise ratio for each photon
    attrs['weight_ph'] = {}
    attrs['weight_ph']['units'] = 100
    attrs['weight_ph']['long_name'] = "Signal-to-Noise Ratio"
    attrs['weight_ph']['description'] = ("Signal-to-Noise ratio calculated using "
        "the photon event classifier, normalized using the maximum weight "
        "in an ATL03 segment")
    attrs['weight_ph']['source'] = "YAPC"
    attrs['weight_ph']['normalization'] = kwargs['norm']
    attrs['weight_ph']['contentType'] = "qualityInformation"
    attrs['weight_ph']['coordinates'] = "delta_time lat_ph lon_ph"
    # photon signal-to-noise confidence from photon classifier
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
    attrs['yapc_conf']['normalization'] = kwargs['norm']
    attrs['yapc_conf']['contentType'] = "qualityInformation"
    attrs['yapc_conf']['coordinates'] = "delta_time lat_ph lon_ph"

    # read each input beam within the file
    IS2_atl03_beams = []
    for gtx in [k for k in f_in.keys() if bool(re.match(r'gt\d[lr]',k))]:
        # check if subsetted beam contains data
        # check in both the geolocation and heights groups
        try:
            f_in[gtx]['geolocation']['segment_id']
            f_in[gtx]['heights']['delta_time']
        except KeyError:
            pass
        else:
            IS2_atl03_beams.append(gtx)

    # open output file
    if (output == 'append'):
        # copy filename and file object for output variables
        output_file = input_file
        f_out = f_in
    elif (output == 'copy'):
        # Copy input file to a new appended ATL03 file
        fargs=(SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
        file_format='{0}{1}_{2}{3}{4}{5}{6}{7}_{8}{9}{10}_{11}_{12}{13}_YAPC.h5'
        output_file = os.path.join(directory,file_format.format(*fargs))
        yapc.utilities.copy(input_file,output_file)
        # Open the output HDF5 files for appending
        f_out = h5py.File(output_file, 'a')
    elif (output == 'reduce'):
        # create a new reduced file with only YAPC parameters
        fargs=(SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
        file_format='{0}{1}_YAPC_{2}{3}{4}{5}{6}{7}_{8}{9}{10}_{11}_{12}{13}.h5'
        output_file = os.path.join(directory,file_format.format(*fargs))
        # Open the output HDF5 files for writing
        f_out = h5py.File(output_file, 'w')
        # copy file-level attributes from input to output
        for att_name,att_val in f_in.attrs.items():
            f_out.attrs[att_name] = att_val
        # for each valid beam group in the input file
        for gtx in IS2_atl03_beams:
            copy_ATL03_beam_group(f_in, f_out, gtx)

    # print output file if verbose
    logger.info('{0} ({1})'.format(output_file, output))

    # for each input beam within the file
    for gtx in sorted(IS2_atl03_beams):
        logger.info(gtx)
        # ATL03 Segment ID
        Segment_ID = f_in[gtx]['geolocation']['segment_id'][:]
        # number of ATL03 20 meter segments
        n_seg = len(Segment_ID)
        # number of photon events
        n_pe, = f_in[gtx]['heights']['delta_time'].shape

        # first photon in the segment (convert to 0-based indexing)
        Segment_Index_begin = f_in[gtx]['geolocation']['ph_index_beg'][:] - 1
        # number of photon events in the segment
        Segment_PE_count = f_in[gtx]['geolocation']['segment_ph_cnt'][:]
        # along-track distance for each ATL03 segment
        Segment_Distance = f_in[gtx]['geolocation']['segment_dist_x'][:]
        # along-track and across-track distance for photon events
        x_atc = f_in[gtx]['heights']['dist_ph_along'][:].copy()
        # photon event heights
        h_ph = f_in[gtx]['heights']['h_ph'][:].copy()
        # flag denoting photon events as possible TEP
        if (int(RL) < 4):
            isTEP = np.any((f_in[gtx]['heights']['signal_conf_ph'][:]==-2),axis=1)
        else:
            isTEP = (f_in[gtx]['heights']['quality_ph'][:] == 3)

        # photon event weights and normalization
        pe_weights = np.zeros((n_pe))
        weight_ph_norm = np.zeros((n_seg))
        # photon event variables
        heights = {}
        # photon signal-to-noise ratios from classifier
        heights['weight_ph'] = np.zeros((n_pe),dtype=np.uint8)
        # photon confidence levels from classifier
        heights['yapc_conf'] = np.zeros((n_pe),dtype=np.int8)
        # output YAPC segment variables
        yapc_window = {}
        # segment maximum range of heights
        yapc_window['h_win_width'] = np.zeros((n_seg),dtype=np.int64)
        # selection window sizes
        yapc_window['win_x'] = np.zeros((n_seg))
        yapc_window['win_h'] = np.zeros((n_seg))
        # dynamic number of values in KNN algorithm
        yapc_window['K'] = np.zeros((n_seg))
        # photon weight normalization scaled to byte
        yapc_window['weight_ph_norm'] = np.zeros((n_seg),dtype=np.uint8)
        # calculate weights for each 20m segment
        for j,_ in enumerate(Segment_ID):
            # index for 20m segment j
            idx = Segment_Index_begin[j]
            # skip segments with no photon events
            if (idx < 0):
                continue
            # number of photons in 20m segment
            cnt = Segment_PE_count[j]
            # buffer photons with previous and following segments
            cnt_m1 = 0 if (j == 0) else Segment_PE_count[j-1]
            cnt_p1 = 0 if (j == (n_seg-1)) else Segment_PE_count[j+1]
            # photon indices for segment (buffered by 1 on each side)
            i1 = np.arange(idx-cnt_m1, idx+cnt+cnt_p1, 1)
            # indices of non-TEP photons in central segment
            i2 = np.nonzero((i1 >= idx) & (i1 < (idx+cnt)) &
                np.logical_not(isTEP[i1]))
            # skip segments that are all TEP photons
            if not np.any(i2):
                continue
            # add segment distance to along-track coordinates
            distance_along_X = np.copy(x_atc[i1])
            distance_along_X += Segment_Distance[j]
            # calculate the height extent
            h_win_width = np.max([1., h_extent(h_ph[i1])])
            yapc_window['h_win_width'][j] = h_win_width.astype(np.int64)
            # photon event weights
            pe_weights[i1[i2]], win_x, win_h, K = yapc.classify_photons(
                distance_along_X, h_ph[i1], h_win_width, i2, **kwargs)
            # selection window sizes (can be dynamically calculated)
            yapc_window['win_x'][j] = np.copy(win_x)
            yapc_window['win_h'][j] = np.copy(win_h)
            # dynamically selected number of neighbors
            yapc_window['K'][j] = np.copy(K)
            # normalize photon weights for each segment
            if (kwargs['norm'] == 'segment'):
                weight_ph_norm[j] = np.max(pe_weights[i1[i2]])

        # normalize photon weights over complete granule
        if (kwargs['norm'] == 'granule'):
            weight_ph_norm[:] = np.max(pe_weights)

        # calculated scaled weights for each 20m segment
        for j,_ in enumerate(Segment_ID):
            # index for 20m segment j
            idx = Segment_Index_begin[j]
            # skip segments with no photon events
            if (idx < 0):
                continue
            # number of photons in 20m segment
            cnt = Segment_PE_count[j]
            # buffer photons with previous and following segments
            cnt_m1 = 0 if (j == 0) else Segment_PE_count[j-1]
            cnt_p1 = 0 if (j == (n_seg-1)) else Segment_PE_count[j+1]
            # photon indices for segment (buffered by 1 on each side)
            i1 = np.arange(idx-cnt_m1, idx+cnt+cnt_p1, 1)
            # indices of non-TEP photons in central segment
            i2 = np.nonzero((i1 >= idx) & (i1 < (idx+cnt)) &
                np.logical_not(isTEP[i1]))
            # skip segments that are all TEP photons
            if not np.any(i2):
                continue
            # skip segments where the maximum weight is zero
            if (weight_ph_norm[j] == 0):
                continue
            # scaled photon event weights from photon classifier
            segment_weights = 255.0*(pe_weights[i1[i2]]/weight_ph_norm[j])
            # verify segment weights and copy to output
            np.clip(segment_weights, 0, 255, out=segment_weights)
            heights['weight_ph'][i1[i2]] = segment_weights.astype(np.uint8)
            # calculate normalization for 20m segment
            yapc_window['weight_ph_norm'][j] = weight_ph_norm[j].astype(np.uint8)
            # photon event signal-to-noise ratio from photon classifier
            if (weight_ph_norm[j] > 0):
                # calculate PE signal-to-noise ratio
                scaled_SNR = (100.0*segment_weights)/weight_ph_norm[j]
                # verify PE SNR values and add to output array
                np.clip(scaled_SNR, 0, 100, out=scaled_SNR)
                # calculate confidence levels from photon classifier
                segment_class = np.zeros_like(scaled_SNR, dtype=np.int8)
                segment_class[scaled_SNR >= L_CONF] = 2
                segment_class[scaled_SNR >= M_CONF] = 3
                segment_class[scaled_SNR >= H_CONF] = 4
                # copy segment classification to output heights variable
                heights['yapc_conf'][i1[i2]] = segment_class.copy()

        # add YAPC attributes to geolocation group
        f_out[gtx]['geolocation'].attrs['yapc_version'] = \
            yapc.version.full_version
        for att_name in ['min_ph','min_xspread','min_hspread','metric']:
            f_out[gtx]['geolocation'].attrs[att_name] = kwargs[att_name]

        # segment variables from photon classifier
        for key in ['h_win_width','win_x','win_h','K','weight_ph_norm']:
            val = '{0}/{1}/{2}'.format(gtx,'geolocation',key)
            logger.info(val)
            h5 = f_out.create_dataset(val, np.shape(yapc_window[key]),
                data=yapc_window[key], dtype=yapc_window[key].dtype,
                compression='gzip')
            # attach dimensions
            for i,dim in enumerate(['delta_time']):
                h5.dims[i].attach_scale(f_out[gtx]['geolocation'][dim])
            # add HDF5 variable attributes
            for att_name,att_val in attrs[key].items():
                h5.attrs[att_name] = att_val

        # photon signal-to-noise ratio from photon classifier
        for key in ['weight_ph','yapc_conf']:
            val = '{0}/{1}/{2}'.format(gtx,'heights',key)
            logger.info(val)
            h5 = f_out.create_dataset(val, np.shape(heights[key]),
                data=heights[key], dtype=heights[key].dtype,
                compression='gzip')
            # attach dimensions
            for i,dim in enumerate(['delta_time']):
                h5.dims[i].attach_scale(f_out[gtx]['heights'][dim])
            # add HDF5 variable attributes
            for att_name,att_val in attrs[key].items():
                h5.attrs[att_name] = att_val

    # close the HDF5 files
    f_in.close()
    f_out.close()
    # change the permissions mode
    os.chmod(output_file, mode=mode)

# PURPOSE: copy ATL03 variables and attributes to output file
def copy_ATL03_beam_group(f_in, f_out, gtx):
    # group variables to copy for each beam group
    groups = {}
    groups['geolocation'] = ['delta_time','segment_id',
        'reference_photon_lat','reference_photon_lon']
    groups['heights'] = ['delta_time','lat_ph','lon_ph']
    # attributes to not copy from input HDF5 file
    invalid_attributes = ['CLASS','DIMENSION_LIST','NAME','REFERENCE_LIST']
    # create the beam group
    f_out.create_group(gtx)
    # copy group attributes from input to output
    for att_name,att_val in f_in[gtx].attrs.items():
        f_out[gtx].attrs[att_name] = att_val
    # create data groups and copy necessary variables
    for group,keys in groups.items():
        f_out[gtx].create_group(group)
        # for each variable to copy from the input file
        for key in keys:
            val = '{0}/{1}/{2}'.format(gtx,group,key)
            h5 = f_out.create_dataset(val, data=f_in[val],
                dtype=f_in[val].dtype, compression='gzip')
            # create or attach dimensions
            if (key == 'delta_time'):
                # make dimension
                h5.make_scale('delta_time')
            else:
                # attach dimensions
                for i,dim in enumerate(['delta_time']):
                    h5.dims[i].attach_scale(f_out[gtx][group][dim])
            # copy group attributes from input to output
            for att_name,att_val in f_in[val].attrs.items():
                if att_name not in invalid_attributes:
                    f_out[val].attrs[att_name] = att_val

# PURPOSE: create argument parser
def arguments():
    # Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Reads ICESat-2 ATL03 data files and appends
            photon classification flags from YAPC
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = yapc.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL03 file to run')
    # number of values for KNN algorithm
    parser.add_argument('--K','-k',
        type=int, default=0,
        help='Number of values for KNN algorithm')
    # minimum value of K for KNN algorithm
    parser.add_argument('--min-knn',
        type=int, default=5,
        help='Minimum value of K used in the KNN algorithm')
    # minimum number of photons for a segment
    parser.add_argument('--min-ph',
        type=int, default=3,
        help='Minimum number of photons for a segment to be valid')
    # minimum along-track spread of photon events
    parser.add_argument('--min-x-spread',
        type=float, default=1.0,
        help='Minimum along-track spread of photon events')
    # minimum window of heights for photon events
    parser.add_argument('--min-h-spread',
        type=float, default=0.01,
        help='Minimum window of heights for photon events')
    # x and h window
    parser.add_argument('--win-x',
        type=float, default=15.0,
        help='Along-track length of window')
    parser.add_argument('--win-h',
        type=float, default=6.0,
        help='Height of window')
    # aspect ratio of x and h window
    parser.add_argument('--aspect',
        type=float, default=0.0,
        help='Aspect ratio of x and h window')
    # algorithm for computing photon event weights
    choices = ('ball_tree','linear','brute')
    parser.add_argument('--method',
        type=str.lower, default='linear', choices=choices,
        help='Algorithm for computing photon event weights')
    # metric for computing distances
    choices = ('height','manhattan')
    parser.add_argument('--metric',
        type=str.lower, default='height', choices=choices,
        help='Metric for computing distances')
    # normalization scheme for weights (segment level or granule)
    choices = ('segment','granule')
    parser.add_argument('--norm',
        type=str.lower, default='segment', choices=choices,
        help='Normalization scheme for weights')
    # output file type (appended, copied or reduced)
    choices = ('append','copy','reduce')
    parser.add_argument('--output','-O',
        type=str.lower, default='copy', choices=choices,
        help='Output file type')
    # verbosity settings
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    # permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permissions mode of output files')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args = parser.parse_args()

    # run the program for the ATL03 file
    for ATL03_file in args.infile:
        append_YAPC_ICESat2_ATL03(ATL03_file,
            K=args.K,
            min_knn=args.min_knn,
            min_ph=args.min_ph,
            min_xspread=args.min_x_spread,
            min_hspread=args.min_h_spread,
            win_x=args.win_x,
            win_h=args.win_h,
            aspect=args.aspect,
            method=args.method,
            metric=args.metric,
            norm=args.norm,
            output=args.output,
            verbose=args.verbose,
            mode=args.mode)

# run main program
if __name__ == '__main__':
    main()
