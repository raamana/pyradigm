import random
import warnings
from collections import Iterable
from copy import copy
from operator import itemgetter
from sys import version_info

import numpy as np

if version_info.major == 2 and version_info.minor == 7:
    from pyradigm import MLDataset
elif version_info.major > 2:
    from pyradigm.pyradigm import MLDataset
else:
    raise NotImplementedError('pyradigm supports only 2.7 or 3+. '
                              'Upgrade to Python 3+ is recommended.')


class MultiDataset(object):
    """
    Container data structure to hold and manage multiple MLDataset instances.

    Key uses:
        - Uniform processing individual MLDatasets e.g. querying same set of IDs
        - ensuring correspondence across multiple datasets in CV

    """


    def __init__(self,
                 dataset_spec=None,
                 name='MultiDataset'):
        """
        Constructor.

        Parameters
        ----------
        dataset_spec : Iterable or None
            List of MLDatasets, or absolute paths to serialized MLDatasets.

        """

        self._list = list()
        self._is_init = False

        # number of modalities for each sample id
        self._modality_count = 0

        self._ids = set()
        self._classes = dict()
        self._modalities = dict()
        self._labels = dict()

        self._num_features = list()

        # TODO more efficient internal repr is possible as ids/classes do not need be
        # stored redundantly for each dataset
        # perhaps as different attributes/modalities/feat-sets (of .data) for example?

        if dataset_spec is not None:
            if not isinstance(dataset_spec, Iterable) or len(dataset_spec) < 1:
                raise ValueError('Input must be a list of atleast two datasets.')

            self._load(dataset_spec)

        self._name = name


    def _load(self, dataset_spec):
        """Actual loading of datasets"""

        for idx, ds in enumerate(dataset_spec):
            self.append(ds, idx)


    def _get_id(self):
        """Returns an ID for a new dataset that's different from existing ones."""

        self._modality_count += 1

        return self._modality_count


    def append(self, dataset, identifier):
        """
        Adds a dataset, if compatible with the existing ones.

        Parameters
        ----------

        dataset : MLDataset or compatible

        identifier : hashable
            String or integer or another hashable to uniquely identify this dataset

        """

        dataset = dataset if isinstance(dataset, MLDataset) else MLDataset(dataset)

        if not self._is_init:
            self._ids = set(dataset.keys)
            self._classes = dataset.classes
            self._class_sizes = dataset.class_sizes

            self._num_samples = len(self._ids)
            self._modalities[identifier] = dataset.data
            self._num_features.append(dataset.num_features)

            # maintaining a no-data MLDataset internally for reuse its methods
            self._dataset = copy(dataset)
            # replacing its data with zeros
            self._dataset.data = {id_: np.zeros(1) for id_ in self._ids}
        else:
            # this also checks for the size (num_samples)
            if set(dataset.keys) != self._ids:
                raise ValueError('Differing set of IDs in two datasets.'
                                 'Unable to add this dataset to the MultiDataset.')

            if dataset.classes != self._classes:
                raise ValueError('Classes for IDs differ in the two datasets.')

            if identifier not in self._modalities:
                self._modalities[identifier] = dataset.data
                self._num_features.append(dataset.num_features)
            else:
                raise KeyError('{} already exists in MultiDataset'.format(identifier))

        # each addition should be counted, if successful
        self._modality_count += 1


    def __str__(self):
        """human readable repr"""

        string = "{}: {} samples, " \
                 "{} modalities, " \
                 "dims: {}\nclass sizes: ".format(self._name, self._num_samples,
                                                 self._modality_count, self._num_features)

        string += ', '.join(['{}: {}'.format(c, n) for c, n in self._class_sizes.items()])

        return string


