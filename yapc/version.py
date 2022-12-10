#!/usr/bin/env python
u"""
version.py (09/2021)
"""
from pkg_resources import get_distribution
# get version for distribution
version = get_distribution("pyYAPC").version
# append "v" before the version
full_version = f"v{version}"
# get project name
project_name = get_distribution("pyYAPC").project_name
