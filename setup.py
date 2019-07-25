#!/usr/bin/env python

from setuptools import setup, find_packages
import versioneer 

setup(name='pyradigm',
      version=versioneer.get_version(),
      description='Python-based data structure to improve handling '
                  'of datasets in machine learning workflows',
      long_description='Pyradigm: Python-based data structure to improve '
                       'handling of datasets in machine learning workflows',
      keywords='machine learning, test dataset, python, workflow, '
               'provenance, data structure',
      author='Pradeep Reddy Raamana',
      author_email='raamana@gmail.com',
      url='https://github.com/raamana/pyradigm',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      install_requires=['numpy', 'setuptools'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'Programming Language :: Python :: 3.6',
      ],
      entry_points={
          "console_scripts": [
              "pyradigm=pyradigm.__main__:main",
          ]
      },
      cmdclass=versioneer.get_cmdclass()
      )
