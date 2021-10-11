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