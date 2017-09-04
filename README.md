# Pyradigm: PYthon based data structure to improve Dataset's InteGrity in Machine learning workflows

[![status](http://joss.theoj.org/papers/c5c231486d699bca982ca7ebd9cf32d2/status.svg)](http://joss.theoj.org/papers/c5c231486d699bca982ca7ebd9cf32d2)
[![travis](https://travis-ci.org/raamana/pyradigm.svg?branch=master)](https://travis-ci.org/raamana/pyradigm.svg?branch=master)
[![Code Health](https://landscape.io/github/raamana/pyradigm/master/landscape.svg?style=flat)](https://landscape.io/github/raamana/pyradigm/master)
[![PyPI version](https://badge.fury.io/py/pyradigm.svg)](https://badge.fury.io/py/pyradigm)
[![codecov](https://codecov.io/gh/raamana/pyradigm/branch/master/graph/badge.svg)](https://codecov.io/gh/raamana/pyradigm)

A common problem for machine learning developers is keeping track of the source of the features extracted, and to ensure integrity of the dataset (e.g. not getting data mixed up from different subjects and/or classes). This is incredibly hard as the number of projects grow, or personnel changes are frequent. These aspects can break the chain of hyper-local info about various datasets, such as where did the original data come from, how was it processed or quality controlled, how was it put together, by who and what does some columns in the table mean etc. This package provides a Python data structure to encapsulate a machine learning dataset with key info greatly suited for neuroimaging applications (or any other domain), where each sample needs to be uniquely identified with a subject ID (or something similar). Key-level correspondence across data, labels (e.g. 1 or 2), classnames (e.g. 'healthy', 'disease') and the related helps maintain data integrity, in addition to offering a way to easily trace back to the sources from where the features have been originally derived.

For users of Panadas, some of the elements in `pyradigm`'s API/interface may look familiar. However, the aim of this data structure is not to offer an alternative to pandas, but to ease the machine learning workflow for neuroscientists by 1) offering several well-knit methods and useful attributes specifically geared towards neuroscience research, 2) aiming to offer utilities that combines multiple or advanced patterns of routine dataset handling and 3) using a more accessible language (compared to hard to read pandas docs aimed at econometric audience) to better cater to neuroscience developers (esp. the novice).

Thanks for checking out. Your feedback will be appreciated.

## Installation

`pip install pyradigm`

## Usage

This [Pyradigm Example](PyradigmExample.ipynb) notebook illustrates the usage.

## Requirements

 * Packages: `numpy`
 * Python versions: I plan to support all the popular versions soon. Only 2.7 is tested for support at the moment.

## Support on Beerpay
Hey dude! Help me out for a couple of :beers:!

[![Beerpay](https://beerpay.io/raamana/pyradigm/badge.svg?style=beer-square)](https://beerpay.io/raamana/pyradigm)  [![Beerpay](https://beerpay.io/raamana/pyradigm/make-wish.svg?style=flat-square)](https://beerpay.io/raamana/pyradigm?focus=wish)
