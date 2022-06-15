#!/usr/bin/env python
u"""
version.py (09/2021)
"""
from pkg_resources import get_distribution
# get version for distribution
version = get_distribution("pyYAPC").version
# append "v" before the version
full_version = "v{0}".format(version)
