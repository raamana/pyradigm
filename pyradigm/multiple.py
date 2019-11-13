import random
from warnings import warn
from collections import Iterable
from copy import copy
from operator import itemgetter
from sys import version_info
from abc import abstractmethod
import numpy as np

if version_info.major > 2:
    from pyradigm.base import BaseDataset, CompatibilityException
    from pyradigm import MLDataset, ClassificationDataset as ClfDataset
else:
    raise NotImplementedError('pyradigm supports only python 3 or higher! '
                              'Upgrade to Python 3+ is recommended.')


class BaseMultiDataset(object):
    """
    Container data structure to hold and manage multiple MLDataset instances.

    Key uses:
        - Uniform processing individual MLDatasets e.g. querying same set of IDs
        - ensuring correspondence across multiple datasets in CV

    """


    def __init__(self,
                 dataset_class=ClfDataset,
                 dataset_spec=None,
                 name='MultiDataset'):
        """
        Constructor.

        Parameters
        ----------
        dataset_spec : Iterable or None
            List of MLDatasets, or absolute paths to serialized MLDatasets.

        """

        if issubclass(dataset_class, BaseDataset):
            self._dataset_class = dataset_class
        else:
            raise TypeError('Input class type is not recognized!'
                            ' Must be a child class of pyradigm.BaseDataset')

        self._list = list()
        self._is_init = False

        # number of modalities for each sample id
        self._modality_count = 0

        self._ids = set()
        self._targets = dict()
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

        if not isinstance(dataset, BaseDataset):
            dataset = self._dataset_class(dataset_path=dataset)

        if not self._is_init:
            self._ids = set(dataset.samplet_ids)
            self._targets = dataset.targets
            self._target_sizes = dataset.target_sizes

            self._num_samples = len(self._ids)
            self._modalities[identifier] = dataset.data
            self._num_features.append(dataset.num_features)

            # maintaining a no-data MLDataset internally for reuse its methods
            self._dataset = copy(dataset)
            # replacing its data with zeros
            self._dataset.data = {id_: np.zeros(1) for id_ in self._ids}

            self._is_init = True
        else:
            # this also checks for the size (num_samplets)
            if set(dataset.samplet_ids) != self._ids:
                raise CompatibilityException(
                        'Differing set of IDs in two datasets.'
                        ' Unable to add this dataset to the MultiDataset.')

            if dataset.targets != self._targets:
                raise CompatibilityException(
                        'Targets for some IDs differ in the two datasets.'
                        ' Unable to add this dataset to the MultiDataset.')

            if identifier not in self._modalities:
                self._modalities[identifier] = dataset.data
                self._num_features.append(dataset.num_features)
            else:
                raise KeyError('{} already exists in MultiDataset'.format(identifier))

        # each addition should be counted, if successful
        self._modality_count += 1


    @abstractmethod
    def __str__(self):
        """human readable repr"""

    @abstractmethod
    def holdout(self,
                train_perc=0.7,
                num_rep=50,
                return_ids_only=False,
                format='MLDataset'):
        """
        Builds a generator for train and test sets for cross-validation.
        """


    def _get_data(self, id_list, format='MLDataset'):
        """Returns the data, from all modalities, for a given list of IDs"""

        format = format.lower()

        features = list()  # returning a dict would be better if AutoMKL() can handle it
        for modality, data in self._modalities.items():
            if format in ('ndarray', 'data_matrix'):
                # turning dict of arrays into a data matrix
                # this is arguably worse, as labels are difficult to pass
                subset = np.array(itemgetter(*id_list)(data))
            elif format in ('mldataset', 'pyradigm'):
                # getting container with fake data
                subset = self._dataset.get_subset(id_list)
                # injecting actual features
                subset.data = { id_: data[id_] for id_ in id_list }
            else:
                raise ValueError('Invalid output format - choose only one of '
                                 'pyradigm or data_matrix')

            features.append(subset)

        return features


def compute_training_sizes(train_perc, target_sizes, stratified=True):
    """Computes the maximum training size that the smallest class can provide """

    size_per_class = np.int64(np.around(train_perc * target_sizes))

    if stratified:
        print("Different classes in training set are stratified to match smallest class!")

        # per-class
        size_per_class = np.minimum(np.min(size_per_class), size_per_class)

        # single number
        reduced_sizes = np.unique(size_per_class)
        if len(reduced_sizes) != 1:  # they must all be the same
            raise ValueError("Error in stratification of training set based on "
                             "smallest class!")

    total_test_samples = np.int64(np.sum(target_sizes) - sum(size_per_class))

    return size_per_class, total_test_samples
