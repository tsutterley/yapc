import os
import sys
import logging
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
