#!/usr/bin/env python
u"""
utilities.py
Written by Tyler Sutterley (10/2021)
Management utilities for Yet Another Photon Classifier (YAPC)

UPDATE HISTORY:
    Written 10/2021
"""
import logging

def build_logger(name, **kwargs):
    """
    Builds a logging instance with the specified name

    Arguments
    ---------
    name: name of the logger

    Keyword arguments
    -----------------
    level: lowest-severity log message logger will handle
    format: event description message format
    """
    # set default arguments
    kwargs.setdefault('level', logging.CRITICAL)
    kwargs.setdefault('format', '%(levelname)s:%(name)s:%(message)s')
    # build logger
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(kwargs['format'])
    # add handler to logger
    logger.setLevel(kwargs['level'])
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger