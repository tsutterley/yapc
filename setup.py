import os
import sys
import logging
import subprocess
from setuptools import setup, Extension, find_packages

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger()

# package description and keywords
description = ('Python version of the NASA GSFC YAPC '
    '("Yet Another Photon Classifier") Program')
keywords = 'ICESat-2 laser altimetry, ATLAS, photon classification, kNN filtering'
# get long_description from README.rst
with open("README.rst", mode="r", encoding="utf8") as fh:
    long_description = fh.read()
long_description_content_type = "text/x-rst"

# get install requirements
with open('requirements.txt', mode="r", encoding="utf8") as fh:
    install_requires = [line.split().pop(0) for line in fh.read().splitlines()]

# get version
with open('version.txt', mode="r", encoding="utf8") as fh:
    version = fh.read()

# list of all scripts to be included with package
scripts=[os.path.join('scripts',f) for f in os.listdir('scripts') if f.endswith('.py')]


# run cmd from the command line
def check_output(cmd):
    return subprocess.check_output(cmd).decode('utf')

# check if HDF5 is installed
hdf5_output = [None] * 2
try:
    for i, cmd in enumerate((["h5cc","-showconfig"], ["h5dump","--version"])):
        hdf5_output[i] = check_output(cmd).strip()
    # parse HDF5 version from h5dump
    hdf5_version = hdf5_output[1].split().pop(2)
except Exception as e:
    log.warning('Failed to get HDF5 options')
else:
    log.info(f"HDF5 version from via h5dump: {hdf5_version}")
# if the HDF5 version not found
if not any(hdf5_output) and ('h5py' in install_requires):
    hdf5_index = install_requires.index('h5py')
    install_requires.pop(hdf5_index)

# Setuptools 18.0 properly handles Cython extensions.
setup_requires=[
    'setuptools>=18.0',
    'cython',
]
# cythonize extensions
ext_modules=[
    Extension('yapc._dist_metrics', sources=['yapc/_dist_metrics.pyx'])
]

setup(
    name='pyYAPC',
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url='https://github.com/tsutterley/yapc',
    author='Tyler Sutterley',
    author_email='tsutterl@uw.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords=keywords,
    packages=find_packages(),
    install_requires=install_requires,
    setup_requires=setup_requires,
    scripts=scripts,
    include_package_data=True,
    ext_modules=ext_modules,
)
