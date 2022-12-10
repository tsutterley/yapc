#!/usr/bin/env python
u"""
utilities.py
Written by Tyler Sutterley (10/2021)
Management utilities for Yet Another Photon Classifier (YAPC)

UPDATE HISTORY:
    Updated 12/2022: functions for managing and maintaining git repositories
    Written 10/2021
"""
import os
import re
import shutil
import inspect
import logging
import warnings
import subprocess

def build_logger(name, **kwargs):
    """
    Builds a logging instance with the specified name

    Parameters
    ----------
    name: str
        name of the logger
    format: str
        event description message format
    level: int or obj, default logging.CRITICAL
        lowest-severity log message logger will handle
    propagate: bool, default False
        events logged will be passed to higher level handlers
    stream: obj or NoneType, default None
        specified stream to initialize StreamHandler
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

# PURPOSE: get the git hash value
def get_git_revision_hash(refname='HEAD', short=False):
    """
    Get the git hash value for a particular reference

    Parameters
    ----------
    refname: str, default HEAD
        Symbolic reference name
    short: bool, default False
        Return the shorted hash value
    """
    # get path to .git directory from current file path
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    basepath = os.path.dirname(os.path.dirname(os.path.abspath(filename)))
    gitpath = os.path.join(basepath,'.git')
    # build command
    cmd = ['git', f'--git-dir={gitpath}', 'rev-parse']
    cmd.append('--short') if short else None
    cmd.append(refname)
    # get output
    with warnings.catch_warnings():
        return str(subprocess.check_output(cmd), encoding='utf8').strip()

# PURPOSE: get the current git status
def get_git_status():
    """Get the status of a git repository as a boolean value
    """
    # get path to .git directory from current file path
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    basepath = os.path.dirname(os.path.dirname(os.path.abspath(filename)))
    gitpath = os.path.join(basepath,'.git')
    # build command
    cmd = ['git', f'--git-dir={gitpath}', 'status', '--porcelain']
    with warnings.catch_warnings():
        return bool(subprocess.check_output(cmd))

# PURPOSE: convert file lines to arguments
def convert_arg_line_to_args(arg_line):
    """
    Convert file lines to arguments

    Parameters
    ----------
    arg_line: str
        line string containing a single argument and/or comments
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

    Parameters
    ----------
    source: str
        source file
    destination: str
        copied destination file
    move: bool, default False
        remove the source file
    """
    source = os.path.abspath(os.path.expanduser(source))
    destination = os.path.abspath(os.path.expanduser(destination))
    shutil.copyfile(source, destination)
    shutil.copystat(source, destination)
    if move:
        os.remove(source)
