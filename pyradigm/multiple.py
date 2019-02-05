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


class MultiOutputMLDataset(MLDataset):
    """
    New class allowing the labels for a sample to be a vector.

    Recommended way to construct the dataset is via add_sample method, one sample
    at a time, as it allows for unambiguous identification of each row in data matrix.

    This constructor can be used in 3 ways:
        - As a copy constructor to make a copy of the given in_dataset
        - Or by specifying the tuple of dictionaries for data, labels and classes.
            In this usage, you can provide additional inputs such as description
            and feature_names.
        - Or by specifying a file path which contains previously saved
        MultiOutputMLDataset.

    Parameters
    ----------
    filepath : str
        path to saved MLDataset on disk, to directly load it.

    in_dataset : MLDataset
        MLDataset to be copied to create a new one.

    data : dict
        dict of features (keys are treated to be sample ids)

    labels : dict
        dict of labels
        (keys must match with data/classes, are treated to be sample ids)

    classes : dict
        dict of class names
        (keys must match with data/labels, are treated to be sample ids)

    description : str
        Arbitrary string to describe the current dataset.

    feature_names : list, ndarray
        List of names for each feature in the dataset.

    encode_nonnumeric : bool
        Flag to specify whether to encode non-numeric features (categorical,
        nominal or string) features to numeric values.
        Currently used only when importing ARFF files.
        It is usually better to encode your data at the source,
        and them import them to Use with caution!

    Raises
    ------
    ValueError
        If in_dataset is not of type MLDataset or is empty, or
        An invalid combination of input args is given.
    IOError
        If filepath provided does not exist.

    """

    _multi_output = True


    def __init__(self,
                 num_outputs=None,
                 filepath=None,
                 in_dataset=None,
                 data=None,
                 labels=None,
                 classes=None,
                 description='',
                 feature_names=None,
                 encode_nonnumeric=False):
        super().__init__(filepath=filepath,
                         in_dataset=in_dataset,
                         data=data, labels=labels, classes=classes,
                         description=description,
                         feature_names=feature_names,
                         encode_nonnumeric=encode_nonnumeric)

        self._num_outputs = num_outputs


    def _check_labels(self, label_array):
        """Label check for multi-output datasets: label for a subject can be a vector!"""

        if any([self._is_label_invalid(lbl) for lbl in label_array]):
            raise ValueError('One or more of the labels is not valid!')

        return np.array(label_array)


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

            self._is_init = True
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
                                                  self._modality_count,
                                                  self._num_features)

        string += ', '.join(['{}: {}'.format(c, n) for c, n in self._class_sizes.items()])

        return string


    def holdout(self,
                train_perc=0.7,
                num_rep=50,
                stratified=True,
                return_ids_only=False,
                format='MLDataset'):
        """
        Builds a generator for train and test sets for cross-validation.

        """

        ids_in_class = {cid: self._dataset.sample_ids_in_class(cid)
                        for cid in self._class_sizes.keys()}

        sizes_numeric = np.array([len(ids_in_class[cid]) for cid in ids_in_class.keys()])
        size_per_class, total_test_count = compute_training_sizes(
                train_perc, sizes_numeric, stratified=stratified)

        if len(self._class_sizes) != len(size_per_class):
            raise ValueError('size spec differs in num elements with class sizes!')

        for rep in range(num_rep):
            print('rep {}'.format(rep))

            train_set = list()
            for index, (cls_id, class_size) in enumerate(self._class_sizes.items()):
                # shuffling the IDs each time
                random.shuffle(ids_in_class[cls_id])

                subset_size = max(0, min(class_size, size_per_class[index]))
                if subset_size < 1 or class_size < 1:
                    warnings.warn('No subjects from class {} were selected.'
                                  ''.format(cls_id))
                else:
                    subsets_this_class = ids_in_class[cls_id][0:size_per_class[index]]
                    train_set.extend(subsets_this_class)

            # this ensures both are mutually exclusive!
            test_set = list(self._ids - set(train_set))

            if return_ids_only:
                # when only IDs are required, without associated features
                # returning tuples to prevent accidental changes
                yield tuple(train_set), tuple(test_set)
            else:
                yield self._get_data(train_set, format), self._get_data(test_set, format)


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
                subset.data = {id_: data[id_] for id_ in id_list}
            else:
                raise ValueError('Invalid output format - choose only one of '
                                 'MLDataset or data_matrix')

            features.append(subset)

        return features


def compute_training_sizes(train_perc, class_sizes, stratified=True):
    """Computes the maximum training size that the smallest class can provide """

    size_per_class = np.int64(np.around(train_perc * class_sizes))

    if stratified:
        print("Different classes in training set are stratified to match smallest class!")

        # per-class
        size_per_class = np.minimum(np.min(size_per_class), size_per_class)

        # single number
        reduced_sizes = np.unique(size_per_class)
        if len(reduced_sizes) != 1:  # they must all be the same
            raise ValueError("Error in stratification of training set based on "
                             "smallest class!")

    total_test_samples = np.int64(np.sum(class_sizes) - sum(size_per_class))

    return size_per_class, total_test_samples
