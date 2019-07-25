__all__ = ['MLDataset', 'cli_run', 'check_compatibility']

import argparse
import copy
import logging
import numpy as np
import os
import pickle
import random
import sys
from sys import version_info
import traceback
import warnings
from collections import Counter, OrderedDict, Sequence
from itertools import islice
from os.path import basename, dirname, exists as pexists, isfile, join as pjoin, realpath


from pyradigm.base import BaseMLDataset


class MLDataset(BaseMLDataset):
    """The main class for user-facing MLDataset, defaulted for classification.
    """



def cli_run():
    """
    Command line interface

    This interface saves you coding effort to:

        - display basic info (classes, sizes etc) about datasets
        - display meta data (class membership) for samples
        - perform basic arithmetic (add multiple classes or feature sets)


    """

    path_list, meta_requested, summary_requested, add_path_list, out_path = parse_args()

    # printing info if requested
    if path_list:
        for ds_path in path_list:
            ds = MLDataset(ds_path)
            if summary_requested:
                print_info(ds, ds_path)
            if meta_requested:
                print_meta(ds, ds_path)

    # combining datasets
    if add_path_list:
        combine_and_save(add_path_list, out_path)

    return


def print_info(ds, ds_path=None):
    "Prints basic summary of a given dataset."

    if ds_path is None:
        bname = ''
    else:
        bname = basename(ds_path)

    dashes = '-' * len(bname)
    print('\n{}\n{}\n{:full}'.format(dashes, bname, ds))

    return


def print_meta(ds, ds_path=None):
    "Prints meta data for subjects in given dataset."

    print('\n#' + ds_path)
    for sub, cls in ds.classes.items():
        print('{},{}'.format(sub, cls))

    return


def combine_and_save(add_path_list, out_path):
    """
    Combines whatever datasets that can be combined,
    and save the bigger dataset to a given location.
    """

    add_path_list = list(add_path_list)
    # first one!
    first_ds_path = add_path_list[0]
    print('Starting with {}'.format(first_ds_path))
    combined = MLDataset(first_ds_path)
    for ds_path in add_path_list[1:]:
        try:
            combined = combined + MLDataset(ds_path)
        except:
            print('      Failed to add {}'.format(ds_path))
            traceback.print_exc()
        else:
            print('Successfully added {}'.format(ds_path))

    combined.save(out_path)

    return


def get_parser():
    """Argument specifier.

    """

    parser = argparse.ArgumentParser(prog='pyradigm')

    parser.add_argument('path_list', nargs='*', action='store',
                        default=None, help='List of paths to display info about.')

    parser.add_argument('-m', '--meta', action='store_true', dest='meta_requested',
                        required=False,
                        default=False, help='Prints the meta data (subject_id,class).')

    parser.add_argument('-i', '--info', action='store_true', dest='summary_requested',
                        required=False,
                        default=False,
                        help='Prints summary info (classes, #samples, #features).')

    arithmetic_group = parser.add_argument_group('Options for multiple datasets')
    arithmetic_group.add_argument('-a', '--add', nargs='+', action='store',
                                  dest='add_path_list', required=False,
                                  default=None,
                                  help='List of MLDatasets to combine')

    arithmetic_group.add_argument('-o', '--out_path', action='store', dest='out_path',
                                  required=False,
                                  default=None,
                                  help='Output path to save the resulting dataset.')

    return parser


def parse_args():
    """Arg parser.

    """

    parser = get_parser()

    if len(sys.argv) < 2:
        parser.print_help()
        logging.warning('Too few arguments!')
        parser.exit(1)

    # parsing
    try:
        params = parser.parse_args()
    except Exception as exc:
        print(exc)
        raise ValueError('Unable to parse command-line arguments.')

    path_list = list()
    if params.path_list is not None:
        for dpath in params.path_list:
            if pexists(dpath):
                path_list.append(realpath(dpath))
            else:
                print('Below dataset does not exist. Ignoring it.\n{}'.format(dpath))

    add_path_list = list()
    out_path = None
    if params.add_path_list is not None:
        for dpath in params.add_path_list:
            if pexists(dpath):
                add_path_list.append(realpath(dpath))
            else:
                print('Below dataset does not exist. Ignoring it.\n{}'.format(dpath))

        if params.out_path is None:
            raise ValueError(
                'Output path must be specified to save the combined dataset to')

        out_path = realpath(params.out_path)
        parent_dir = dirname(out_path)
        if not pexists(parent_dir):
            os.mkdir(parent_dir)

        if len(add_path_list) < 2:
            raise ValueError('Need a minimum of datasets to combine!!')

    # removing duplicates (from regex etc)
    path_list = set(path_list)
    add_path_list = set(add_path_list)

    return path_list, params.meta_requested, params.summary_requested, \
           add_path_list, out_path


if __name__ == '__main__':
    cli_run()
