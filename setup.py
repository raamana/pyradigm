#!/usr/bin/env python
import os

from setuptools import setup, find_packages

setup(name='pyradigm',
      version='0.3.0.3',
      description='Python-based data structure to improve handling of datasets in machine learning workflows',
      long_description='Pyradigm: Python-based data structure to improve handling of datasets in machine learning workflows',
      keywords='machine learning, test dataset, python, workflow, provenance, data structure',
      author='Pradeep Reddy Raamana',
      author_email='raamana@gmail.com',
      url='https://github.com/raamana/pyradigm',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]), # packages=['pyradigm'],
      install_requires=['numpy', 'setuptools'],
     )

