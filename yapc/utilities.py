#!/usr/bin/env python
u"""
utilities.py
Written by Tyler Sutterley (10/2021)
Management utilities for Yet Another Photon Classifier (YAPC)

UPDATE HISTORY:
    Written 10/2021
"""
import os
import re
import shutil
import logging

def build_logger(name, **kwargs):
    """
    Builds a logging instance with the specified name

    Arguments
    ---------
    name: name of the logger

    Keyword arguments
    -----------------
    format: event description message format
    level: lowest-severity log message logger will handle
    propagate: events logged will be passed to higher level handlers
    stream: specified stream to initialize StreamHandler
    """
    # set default arguments
    kwargs.setdefault('format', '%(levelname)s:%(name)s:%(message)s')
    kwargs.setdefault('level', logging.CRITICAL)
    kwargs.setdefault('propagate',False)
    kwargs.setdefault('stream',None)
    # build logger
    logger = logging.getLogger(name)
    logger.setLevel(kwargs['level'])
    logger.propagate = kwargs['propagate']
    # create and add handlers to logger
    if not logger.handlers:
        # create handler for logger
        handler = logging.StreamHandler(stream=kwargs['stream'])
        formatter = logging.Formatter(kwargs['format'])
        handler.setFormatter(formatter)
        # add handler to logger
        logger.addHandler(handler)
    return logger

# PURPOSE: convert file lines to arguments
def convert_arg_line_to_args(arg_line):
    """
    Convert file lines to arguments

    Arguments
    ---------
    arg_line: line string containing a single argument and/or comments
    """
    # remove commented lines and after argument comments
    for arg in re.sub(r'\#(.*?)$',r'',arg_line).split():
        if not arg.strip():
            continue
        yield arg

# PURPOSE: make a copy of a file with all system information
def copy(source, destination, move=False):
    """
    Copy or move a file with all system information

    Arguments
    ---------
    source: source file
    destination: copied destination file

    Keyword arguments
    -----------------
    move: remove the source file
    """
    source = os.path.abspath(os.path.expanduser(source))
    destination = os.path.abspath(os.path.expanduser(destination))
    shutil.copyfile(source, destination)
    shutil.copystat(source, destination)
    if move:
        os.remove(source)
