__all__ = [ 'MLDataset', 'cli_run' ]

import copy
import os
import sys
import pickle
import random
import warnings
import argparse
import traceback
import logging
from os.path import join as pjoin, exists as pexists, realpath, basename, dirname
from collections import Counter, OrderedDict, Sequence
from itertools import islice

import numpy as np


# TODO profile the class for different scales of samples and features
class MLDataset(object):
    """An ML dataset to ease workflow and maintain integrity."""

    def __init__(self, filepath=None, in_dataset=None,
                 data=None, labels=None, classes=None,
                 description='', feature_names=None):
        """
        Default constructor.
        Suggested way to construct the dataset is via add_sample method, one sample at a time.

        This constructor can be used in 3 ways:
            - As a copy constructor to make a copy of the given in_dataset
            - Or by specifying the tuple of data, labels and classes.
                In this usage, you can provide additional inputs such as description and feature_names.
            - Or by specifying a file path which contains previously saved MLDataset.

        Parameters
        ----------
        filepath : str
            path to saved MLDataset on disk, to directly load it.
        in_dataset : MLDataset
            MLDataset to be copied to create a new one.
        data : dict
            dict of features (keys are treated to be sample ids)
        labels : dict
            dict of labels (keys must match with data/classes, are treated to be sample ids)
        classes : dict
            dict of class names (keys must match with data/labels, are treated to be sample ids)
        description : str
            Arbitray string to describe the current dataset.
        feature_names : list, ndarray
            List of names for each feature in the dataset.

        Raises
        ------
        ValueError
            If in_dataset is not of type MLDataset or is empty, or
            An invalid combination of input args is given.
        IOError
            If filepath provided does not exist.

        """

        if filepath is not None:
            if os.path.isfile(filepath):
                # print('Loading the dataset from: {}'.format(filepath))
                self.__load(filepath)
            else:
                raise IOError('Specified file could not be read.')
        elif in_dataset is not None:
            if not isinstance(in_dataset, MLDataset):
                raise ValueError('Invalid class input: MLDataset expected!')
            if in_dataset.num_samples <= 0:
                raise ValueError('Dataset to copy is empty.')
            self.__copy(in_dataset)
        elif data is None and labels is None and classes is None:
            # TODO refactor the code to use only basic dict, as it allows for better equality comparisons
            self.__data = OrderedDict()
            self.__labels = OrderedDict()
            self.__classes = OrderedDict()
            self.__num_features = 0
            self.__dtype = None
            self.__description = ''
            self.__feature_names = None
        elif data is not None and labels is not None and classes is not None:
            # ensuring the inputs really correspond to each other
            # but only in data, labels and classes, not feature names
            self.__validate(data, labels, classes)

            # OrderedDict to ensure the order is maintained when data/labels are returned in a matrix/array form
            self.__data = OrderedDict(data)
            self.__labels = OrderedDict(labels)
            self.__classes = OrderedDict(classes)
            self.__description = description

            sample_ids = list(data)
            self.__num_features = len(data[sample_ids[0]])
            self.__dtype = type(data[sample_ids[0]])

            # assigning default names for each feature
            if feature_names is None:
                self.__feature_names = self.__str_names(self.num_features)
            else:
                self.__feature_names = feature_names

        else:
            raise ValueError('Incorrect way to construct the dataset.')

    @property
    def data(self):
        """data in its original dict form."""
        return self.__data

    def data_and_labels(self):
        """
        Dataset features and labels in a matrix form for learning.

        Also returns sample_ids in the same order.

        Returns
        -------
        data_matrix : ndarray
            2D array of shape [num_samples, num_features] with features  corresponding row-wise to sample_ids
        labels : ndarray
            Array of numeric labels for each sample corresponding row-wise to sample_ids
        sample_ids : list
            List of sample ids

        """

        sample_ids = np.array(self.keys)
        label_dict = self.labels
        matrix = np.full([self.num_samples, self.num_features], np.nan)
        labels = np.full([self.num_samples, 1], np.nan)
        for ix, sample in enumerate(sample_ids):
            matrix[ix, :] = self.__data[sample]
            labels[ix] = label_dict[sample]

        return matrix, np.ravel(labels), sample_ids

    @data.setter
    def data(self, values):
        """
        Populates this dataset with the provided data.
        Usage of this method is discourage (unless you know what you are doing).

        Parameters
        ----------
        values : dict
            dict of features keyed in by sample ids.

        Raises
        ------
        ValueError
            If number of samples does not match the size of existing set, or
            If atleast one sample is not provided.

        """
        if isinstance(values, dict):
            if self.__labels is not None and len(self.__labels) != len(values):
                raise ValueError('number of samples do not match the previously assigned labels')
            elif len(values) < 1:
                raise ValueError('There must be at least 1 sample in the dataset!')
            else:
                self.__data = values
        else:
            raise ValueError('data input must be a dictionary!')

    @property
    def labels(self):
        """Returns the array of labels for all the samples."""
        # TODO numeric label need to be removed, as this can be made up on the fly as needed from str to num encoders.
        return self.__labels

    @labels.setter
    def labels(self, values):
        """Class labels (such as 1, 2, -1, 'A', 'B' etc.) for each sample in the dataset."""
        if isinstance(values, dict):
            if self.__data is not None and len(self.__data) != len(values):
                raise ValueError('number of samples do not match the previously assigned data')
            elif set(self.keys) != set(list(values)):
                raise ValueError('sample ids do not match the previously assigned ids.')
            else:
                self.__labels = values
        else:
            raise ValueError('labels input must be a dictionary!')

    @property
    def classes(self):
        """Identifiers (sample IDs, or sample names etc) forming the basis of dict-type MLDataset."""
        return self.__classes

    @classes.setter
    def classes(self, values):
        """Classes setter."""
        if isinstance(values, dict):
            if self.__data is not None and len(self.__data) != len(values):
                raise ValueError('number of samples do not match the previously assigned data')
            elif set(self.keys) != set(list(values)):
                raise ValueError('sample ids do not match the previously assigned ids.')
            else:
                self.__classes = values
        else:
            raise ValueError('classes input must be a dictionary!')

    @property
    def feature_names(self):
        "Returns the feature names as an numpy array of strings."

        return self.__feature_names

    @feature_names.setter
    def feature_names(self, names):
        "Stores the text labels for features"

        if len(names) != self.num_features:
            raise ValueError("Number of names do not match the number of features!")
        if not isinstance(names, (Sequence, np.ndarray, np.generic)):
            raise ValueError("Input is not a sequence. Ensure names are in the same order and length as features.")

        self.__feature_names = np.array(names)

    @property
    def class_sizes(self):
        """Returns the sizes of different objects in a Counter object."""
        return Counter(self.classes.values())

    @staticmethod
    def __take(nitems, iterable):
        """Return first n items of the iterable as a list"""
        return dict(islice(iterable, int(nitems)))

    @staticmethod
    def __str_names(num):

        return np.array(['f{}'.format(x) for x in range(num)])

    def glance(self, nitems=5):
        """Quick and partial glance of the data matrix.

        Parameters
        ----------
        nitems : int
            Number of items to glance from the dataset.
            Default : 5

        Returns
        -------
        dict

        """
        nitems = max([1, min([nitems, self.num_samples - 1])])
        return self.__take(nitems, iter(self.__data.items()))

    def summarize_classes(self):
        """
        Summary of classes: names, numeric labels and sizes

        Returns
        -------
        tuple : class_set, label_set, class_sizes

        class_set : list
            List of names of all the classes
        label_set : list
            Label for each class in class_set
        class_sizes : list
            Size of each class (number of samples)

        """

        class_sizes = np.zeros(len(self.class_set))
        for idx, cls in enumerate(self.class_set):
            class_sizes[idx] = self.class_sizes[cls]

        # TODO consider returning numeric label set e.g. for use in scikit-learn
        return self.class_set, self.label_set, class_sizes

    # TODO try implementing based on pandas
    def add_sample(self, sample_id, features, label, class_id=None, feature_names=None):
        """
        Adds a new sample to the dataset with its features, label and class ID.

        This is the preferred way to construct the dataset.

        Parameters
        ----------
        sample_id : str, int
            The identifier that uniquely identifies this sample.
        features : list, ndarray
            The features for this sample
        label : int, str
            The label for this sample
        class_id : int, str
            The class for this sample.
            If not provided, label converted to a string becomes its ID.
        feature_names : list
            The names for each feature. Assumed to be in the same order as `features`

        Raises
        ------
        ValueError
            If `sample_id` is already in the MLDataset, or
            If dimensionality of the current sample does not match the current, or
            If `feature_names` do not match existing names
        TypeError
            If sample to be added is of different data type compared to existing samples.

        """

        if sample_id not in self.__data:
            # ensuring there is always a class name, even when not provided by the user.
            # this is needed, in order for __str__ method to work.
            # TODO consider enforcing label to be numeric and class_id to be string
            #  so portability with other packages is more uniform e.g. for use in scikit-learn
            if class_id is None:
                class_id = str(label)

            if self.num_samples <= 0:
                self.__data[sample_id] = features
                self.__labels[sample_id] = label
                self.__classes[sample_id] = class_id
                self.__dtype = type(features)
                self.__num_features = len(features)
                if feature_names is None:
                    self.__feature_names = self.__str_names(self.num_features)
            else:
                if self.__num_features != len(features):
                    raise ValueError('dimensionality of this sample ({}) does not match existing samples ({})'.format(
                        len(features), self.__num_features))
                if not isinstance(features, self.__dtype):
                    raise TypeError("Mismatched dtype. Provide {}".format(self.__dtype))

                self.__data[sample_id] = features
                self.__labels[sample_id] = label
                self.__classes[sample_id] = class_id
                if feature_names is not None:
                    # if it was never set, allow it
                    # class gets here when adding the first sample, after dataset was initialized with empty constructor
                    if self.__feature_names is None:
                        self.__feature_names = np.array(feature_names)
                    else:  # if set already, ensure a match
                        print('')
                        if not np.array_equal(self.feature_names, np.array(feature_names)):
                            raise ValueError("supplied feature names do not match the existing names!")
        else:
            raise ValueError('{} already exists in this dataset!'.format(sample_id))

    def del_sample(self, sample_id):
        """
        Method to remove a sample from the dataset.

        Parameters
        ----------
        sample_id : str
            sample id to be removed.

        Raises
        ------
        UserWarning
            If sample id to delete was not found in the dataset.

        """
        if sample_id not in self.__data:
            warnings.warn('Sample to delete not found in the dataset - nothing to do.')
        else:
            self.__data.pop(sample_id)
            self.__classes.pop(sample_id)
            self.__labels.pop(sample_id)
            print('{} removed.'.format(sample_id))

    def get_feature_subset(self, subset_idx):
        """
        Returns the subset of features indexed numerically.

        Parameters
        ----------
        subset_idx : list, ndarray
            List of indices to features to be returned

        Returns
        -------
        MLDataset : MLDataset
            with subset of features requested.

        Raises
        ------
        UnboundLocalError
            If input indices are out of bounds for the dataset.

        """

        subset_idx = np.asarray(subset_idx)
        if not (max(subset_idx) < self.__num_features) and (min(subset_idx) >= 0):
            raise UnboundLocalError('indices out of range for the dataset. '
                                    'Max index: {} Min index : 0'.format(self.__num_features))

        sub_data = {sample: features[subset_idx] for sample, features in self.__data.items()}
        new_descr = 'Subset features derived from: \n ' + self.__description
        subdataset = MLDataset(data=sub_data,
                               labels=self.__labels, classes=self.__classes,
                               description=new_descr,
                               feature_names=self.__feature_names[subset_idx])

        return subdataset

    @staticmethod
    def keys_with_value(dictionary, value):
        "Returns a subset of keys from the dict with the value supplied."

        subset = [key for key in dictionary if dictionary[key] == value]

        return subset

    def get_class(self, class_id):
        """
        Returns a smaller dataset belonging to the requested classes.

        Parameters
        ----------
        class_id : str
            identifier of the class to be returned.

        Returns
        -------
        MLDataset
            With subset of samples belonging to the given class.

        Raises
        ------
        ValueError
            If one or more of the requested classes do not exist in this dataset.
            If the specified id is empty or None

        """
        if class_id in [None, '']:
            raise ValueError("class id can not be empty or None.")

        if isinstance(class_id, str):
            class_ids = [class_id, ]
        else:
            class_ids = class_id

        non_existent = set(self.class_set).intersection(set(class_ids))
        if len(non_existent) < 1:
            raise ValueError('These classes {} do not exist in this dataset.'.format(non_existent))

        subsets = list()
        for class_id in class_ids:
            # subsets_this_class = [sample for sample in self.__classes if self.__classes[sample] == class_id]
            subsets_this_class = self.keys_with_value(self.__classes, class_id)
            subsets.extend(subsets_this_class)

        return self.get_subset(subsets)

    def train_test_split_ids(self, train_perc=None, count_per_class=None):
        """
        Returns two disjoint sets of sample ids for use in cross-validation.

        Offers two ways to specify the sizes: fraction or count.
        Only one access method can be used at a time.

        Parameters
        ----------
        train_perc : float
            fraction of samples from each class to build the training subset.

        count_per_class : int
            exact count of samples from each class to build the training subset.

        Returns
        -------
        train_set : list
            List of ids in the training set.
        test_set : list
            List of ids in the test set.

        Raises
        ------
        ValueError
            If the fraction is outside open interval (0, 1), or
            If counts are outside larger than the smallest class, or
            If unrecongized format is provided for input args, or
            If the selection results in empty subsets for either train or test sets.

        """

        _ignore1, _ignore2, class_sizes = self.summarize_classes()
        smallest_class_size = np.min(class_sizes)

        if count_per_class is None and (0.0 < train_perc < 1.0):
            if train_perc < 1.0 / smallest_class_size:
                raise ValueError('Training percentage selected too low '
                                 'to return even one sample from the smallest class!')
            train_set = self.random_subset_ids(perc_per_class=train_perc)
        elif train_perc is None and count_per_class > 0:
            if count_per_class >= smallest_class_size:
                raise ValueError('Selections would exclude the smallest class from test set. '
                                 'Reduce sample count per class for the training set!')
            train_set = self.random_subset_ids_by_count(count_per_class=count_per_class)
        else:
            raise ValueError('Invalid or out of range selection: '
                             'only one of count or percentage can be used to select subset.')

        test_set = list(set(self.keys) - set(train_set))

        if len(train_set) < 1 or len(test_set) < 1:
            raise ValueError('Selection resulted in empty training or test set - check your selections or dataset!')

        return train_set, test_set

    def random_subset_ids_by_count(self, count_per_class=1):
        """
        Returns a random subset of sample ids of specified size by count,
            within each class.

        Parameters
        ----------
        count_per_class : int
            Exact number of samples per each class.

        Returns
        -------
        subset : list
            Combined list of sample ids from all classes.

        """

        class_sizes = self.class_sizes
        subsets = list()

        if count_per_class < 1:
            warnings.warn('Atleast one sample must be selected from each class')
            return list()
        elif count_per_class >= self.num_samples:
            warnings.warn('All samples requested - returning a copy!')
            return self.keys

        # seeding the random number generator
        # random.seed(random_seed)

        for class_id, class_size in class_sizes.items():
            # samples belonging to the class
            # this_class = [sample for sample in self.classes if self.classes[sample] == class_id]
            this_class = self.keys_with_value(self.classes, class_id)
            # shuffling the sample order; shuffling works in-place!
            random.shuffle(this_class)

            # clipping the range to [0, class_size]
            subset_size_this_class = max(0, min(class_size, count_per_class))
            if subset_size_this_class < 1 or this_class is None:
                # warning if none were selected
                warnings.warn('No subjects from class {} were selected.'.format(class_id))
            else:
                subsets_this_class = this_class[0:count_per_class]
                subsets.extend(subsets_this_class)

        if len(subsets) > 0:
            return subsets
        else:
            warnings.warn('Zero samples were selected. Returning an empty list!')
            return list()

    def random_subset_ids(self, perc_per_class=0.5):
        """
        Returns a random subset of sample ids (of specified size by percentage) within each class.

        Parameters
        ----------
        perc_per_class : float
            Fraction of samples per class

        Returns
        -------
        subset : list
            Combined list of sample ids from all classes.

        Raises
        ------
        ValueError
            If no subjects from one or more classes were selected.
        UserWarning
            If an empty or full dataset is requested.

        """

        class_sizes = self.class_sizes
        subsets = list()

        if perc_per_class <= 0.0:
            warnings.warn('Zero percentage requested - returning an empty dataset!')
            return list()
        elif perc_per_class >= 1.0:
            warnings.warn('Full or a larger dataset requested - returning a copy!')
            return self.keys

        # seeding the random number generator
        # random.seed(random_seed)

        for class_id, class_size in class_sizes.items():
            # samples belonging to the class
            # this_class = [sample for sample in self.classes if self.classes[sample] == class_id]
            this_class = self.keys_with_value(self.classes, class_id)
            # shuffling the sample order; shuffling works in-place!
            random.shuffle(this_class)
            # calculating the requested number of samples
            subset_size_this_class = np.int64(np.floor(class_size * perc_per_class))
            # clipping the range to [1, n]
            subset_size_this_class = max(1, min(class_size, subset_size_this_class))
            if subset_size_this_class < 1 or len(this_class) < 1 or this_class is None:
                # warning if none were selected
                raise ValueError('No subjects from class {} were selected.'.format(class_id))
            else:
                subsets_this_class = this_class[0:subset_size_this_class]
                subsets.extend(subsets_this_class)

        if len(subsets) > 0:
            return subsets
        else:
            warnings.warn('Zero samples were selected. Returning an empty list!')
            return list()

    def random_subset(self, perc_in_class=0.5):
        """
        Returns a random sub-dataset (of specified size by percentage) within each class.

        Parameters
        ----------
        perc_in_class : float
            Fraction of samples to be taken from each class.

        Returns
        -------
        subdataset : MLDataset
            random sub-dataset of specified size.

        """

        subsets = self.random_subset_ids(perc_in_class)
        if len(subsets) > 0:
            return self.get_subset(subsets)
        else:
            warnings.warn('Zero samples were selected. Returning an empty dataset!')
            return MLDataset()

    def sample_ids_in_class(self, class_id):
        """
        Returns a list of sample ids belonging to a given class.

        Parameters
        ----------
        class_id : str
            class id to query.

        Returns
        -------
        subset_ids : list
            List of sample ids belonging to a given class.

        """

        # subset_ids = [sid for sid in self.keys if self.classes[sid] == class_id]
        subset_ids = self.keys_with_value(self.classes, class_id)
        return subset_ids

    def get_subset(self, subset_ids):
        """
        Returns a smaller dataset identified by their keys/sample IDs.

        Parameters
        ----------
        subset_ids : list
            List od sample IDs to extracted from the dataset.

        Returns
        -------
        sub-dataset : MLDataset
            sub-dataset containing only requested sample IDs.

        """

        num_existing_keys = sum([1 for key in subset_ids if key in self.__data])
        if subset_ids is not None and num_existing_keys > 0:
            # need to ensure data are added to data, labels etc in the same order of sample IDs
            # TODO come up with a way to do this even when not using OrderedDict()
            # putting the access of data, labels and classes in the same loop would ensure there is correspondence
            # across the three attributes of the class
            data = self.__get_subset_from_dict(self.__data, subset_ids)
            labels = self.__get_subset_from_dict(self.__labels, subset_ids)
            if self.__classes is not None:
                classes = self.__get_subset_from_dict(self.__classes, subset_ids)
            else:
                classes = None
            subdataset = MLDataset(data=data, labels=labels, classes=classes)
            # Appending the history
            subdataset.description += '\n Subset derived from: ' + self.description
            subdataset.feature_names = self.__feature_names
            subdataset.__dtype = self.dtype
            return subdataset
        else:
            warnings.warn('subset of IDs requested do not exist in the dataset!')
            return MLDataset()

    def __contains__(self, item):
        "Boolean test of membership of a sample in the dataset."
        if item in self.keys:
            return True
        else:
            return False

    @staticmethod
    def __get_subset_from_dict(input_dict, subset):
        # Using OrderedDict helps ensure data are added to data, labels etc in the same order of sample IDs
        return OrderedDict((sid, value) for sid, value in input_dict.items() if sid in subset)

    @property
    def keys(self):
        """Sample identifiers (strings) forming the basis of MLDataset (same as sample_ids)"""
        return list(self.__data)

    @property
    def sample_ids(self):
        """Sample identifiers (strings) forming the basis of MLDataset (same as keys)."""
        return self.keys

    @property
    def description(self):
        """Text description (header) that can be set by user."""
        return self.__description

    @description.setter
    def description(self, str_val):
        """Text description that can be set by user."""
        if not str_val: raise ValueError('description can not be empty')
        self.__description = str_val

    @property
    def num_features(self):
        """number of features in each sample."""
        return np.int64(self.__num_features)

    @num_features.setter
    def num_features(self, int_val):
        "Method that should not exist!"
        raise AttributeError("num_features property can't be set, only retrieved!")
        # assert isinstance(int_val, int) and (0 < int_val < np.Inf), UnboundLocalError('Invalid number of features.')
        # self.__num_features = int_val

    @property
    def dtype(self):
        """number of features in each sample."""
        return self.__dtype

    @dtype.setter
    def dtype(self, type_val):
        if self.__dtype is None:
            if not isinstance(type_val, type):
                raise TypeError('Invalid data type.')
            self.__dtype = type_val
        else:
            warnings.warn('Data type is already inferred. Can not be set!')

    @property
    def num_samples(self):
        """number of samples in the entire dataset."""
        if self.__data is not None:
            return len(self.__data)
        else:
            return 0

    @property
    def num_classes(self):
        """Total number of classes in the dataset."""
        return len(self.class_set)

    @property
    def class_set(self):
        """Set of unique classes in the dataset."""

        return list(set(self.__classes.values()))

    @property
    def label_set(self):
        """Set of labels in the dataset corresponding to class_set."""
        label_set = list()
        for class_ in self.class_set:
            samples_in_class = self.sample_ids_in_class(class_)
            label_set.append(self.labels[samples_in_class[0]])

        return label_set

    def add_classes(self, classes):
        """
        Helper to rename the classes, if provided by a dict keyed in by the orignal keys

        Parameters
        ----------
        classes : dict
            Dict of class named keyed in by sample IDs.

        Raises
        ------
        TypeError
            If classes is not a dict.
        ValueError
            If all samples in dataset are not present in input dict,
            or one of they samples in input is not recognized.

        """
        if not isinstance(classes, dict):
            raise TypeError('Input classes is not a dict!')
        if not len(classes) == self.num_samples:
            raise ValueError('Too few items - need {} keys'.format(self.num_samples))
        if not all([key in self.keys for key in classes]):
            raise ValueError('One or more unrecognized keys!')
        self.__classes = classes

    def __len__(self):
        return self.num_samples

    def __nonzero__(self):
        if self.num_samples < 1:
            return False
        else:
            return True

    def __str__(self):
        """Returns a concise and useful text summary of the dataset."""
        full_descr = list()
        full_descr.append(self.description)
        if bool(self):
            full_descr.append('{} samples, {} classes, {} features.'.format(
                self.num_samples, self.num_classes, self.num_features))
            class_ids = list(self.class_sizes)
            max_width = max([len(cls) for cls in class_ids])
            for cls in class_ids:
                full_descr.append('Class {:>{}} : {} samples.'.format(cls, max_width, self.class_sizes.get(cls)))
        else:
            full_descr.append('Empty dataset.')

        return '\n'.join(full_descr)

    def __format__(self, fmt_str='s'):
        if isinstance(fmt_str, str):
            return '{} samples x {} features with {} classes'.format(
                self.num_samples, self.num_features, self.num_classes)
        else:
            raise NotImplementedError('Requsted type of format not implemented.')

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def __dir__():
        """Returns the preferred list of attributes to be used with the dataset."""
        return ['add_sample',
                'glance',
                'summarize_classes',
                'sample_ids_in_class',
                'train_test_split_ids',
                'random_subset_ids',
                'random_subset_ids_by_count',
                'classes',
                'class_set',
                'class_sizes',
                'data_and_labels',
                'data',
                'del_sample',
                'description',
                'extend',
                'feature_names',
                'get_class',
                'get_subset',
                'random_subset',
                'get_feature_subset',
                'keys',
                'num_classes',
                'num_features',
                'num_samples',
                'sample_ids',
                'save',
                'add_classes']

    def __copy(self, other):
        """Copy constructor."""
        self.__data = copy.deepcopy(other.data)
        self.__classes = copy.deepcopy(other.classes)
        self.__labels = copy.deepcopy(other.labels)
        self.__dtype = copy.deepcopy(other.dtype)
        self.__description = copy.deepcopy(other.description)
        self.__feature_names = copy.deepcopy(other.feature_names)
        self.__num_features = copy.deepcopy(other.num_features)

        return self

    def __load(self, path):
        """Method to load the serialized dataset from disk."""
        try:
            path = os.path.abspath(path)
            with open(path, 'rb') as df:
                # loaded_dataset = pickle.load(df)
                self.__data, self.__classes, self.__labels, \
                self.__dtype, self.__description, \
                self.__num_features, self.__feature_names = pickle.load(df)

            # ensure the loaded dataset is valid
            self.__validate(self.__data, self.__classes, self.__labels)

        except IOError as ioe:
            raise IOError('Unable to read the dataset from file: {}', format(ioe))
        except:
            raise

    def save(self, file_path):
        """
        Method to save the dataset to disk.

        Parameters
        ----------
        file_path : str
            File path to save the current dataset to

        Raises
        ------
        IOError
            If saving to disk is not successful.

        """
        try:
            file_path = os.path.abspath(file_path)
            with open(file_path, 'wb') as df:
                # pickle.dump(self, df)
                pickle.dump((self.__data, self.__classes, self.__labels,
                             self.__dtype, self.__description, self.__num_features,
                             self.__feature_names),
                            df)
            return
        except IOError as ioe:
            raise IOError('Unable to save the dataset to file: {}', format(ioe))
        except:
            raise

    @staticmethod
    def __validate(data, classes, labels):
        "Validator of inputs."

        if not isinstance(data, dict):
            raise TypeError('data must be a dict! keys: sample ID or any unique identifier')
        if not isinstance(labels, dict):
            raise TypeError('labels must be a dict! keys: sample ID or any unique identifier')
        if classes is not None:
            if not isinstance(classes, dict):
                raise TypeError('labels must be a dict! keys: sample ID or any unique identifier')

        if not len(data) == len(labels) == len(classes):
            raise ValueError('Lengths of data, labels and classes do not match!')
        if not set(list(data)) == set(list(labels)) == set(list(classes)):
            raise ValueError('data, classes and labels dictionaries must have the same keys!')

        num_features_in_elements = np.unique([len(sample) for sample in data.values()])
        if len(num_features_in_elements) > 1:
            raise ValueError('different samples have different number of features - invalid!')

        return True

    def extend(self, other):
        """
        Method to extend the dataset vertically (add samples from  anotehr dataset).

        Parameters
        ----------
        other : MLDataset
            second dataset to be combined with the current (different samples, but same dimensionality)

        Raises
        ------
        TypeError
            if input is not an MLDataset.
        """

        if not isinstance(other, MLDataset):
            raise TypeError('Incorrect type of dataset provided!')
        # assert self.__dtype==other.dtype, TypeError('Incorrect data type of features!')
        for sample in other.keys:
            self.add_sample(sample, other.data[sample], other.labels[sample], other.classes[sample])

    def __add__(self, other):
        "Method to combine to MLDatasets, sample-wise or feature-wise."

        if not isinstance(other, MLDataset):
            raise TypeError('Incorrect type of dataset provided!')

        if set(self.keys) == set(other.keys):
            print('Identical keys found. Trying to horizontally concatenate features for each sample.')
            if not self.__classes == other.classes:
                raise ValueError('Class identifiers per sample differ in the two datasets!')
            if other.num_features < 1:
                raise ValueError('No features to concatenate.')
            # making an empty dataset
            combined = MLDataset()
            # populating it with the concatenated feature set
            for sample in self.keys:
                comb_data = np.concatenate([self.__data[sample], other.data[sample]])
                combined.add_sample(sample, comb_data, self.__labels[sample], self.__classes[sample])

            return combined

        elif len(set(self.keys).intersection(other.keys)) < 1 and self.__num_features == other.num_features:
            # making a copy of self first
            combined = MLDataset(in_dataset=self)
            # adding the new dataset
            combined.extend(other)
            return combined
        else:
            raise ArithmeticError('Two datasets could not be combined.')

    def __sub__(self, other):
        """Removing one dataset from another."""
        if not isinstance(other, type(self)):
            raise TypeError('Incorrect type of dataset provided!')

        num_existing_keys = len(set(self.keys).intersection(other.keys))
        if num_existing_keys < 1:
            warnings.warn('None of the sample ids to be removed found in this dataset - nothing to do.')
        if len(self.keys) == num_existing_keys:
            warnings.warn('Requested removal of all the samples - output dataset would be empty.')

        removed = copy.deepcopy(self)
        for sample in other.keys:
            removed.del_sample(sample)

        return removed

    def __iadd__(self, other):
        """Augmented assignment for add."""
        return self.__add__(other)

    def __isub__(self, other):
        """Augmented assignment for sample."""
        return self.__sub__(other)

    def __eq__(self, other):
        """Equality of two datasets in samples and their values."""
        if set(self.keys) != set(other.keys):
            print('differing sample ids.')
            return False
        elif dict(self.__classes) != dict(other.classes):
            print('differing classes for the sample ids.')
            return False
        elif id(self.__data) != id(other.data):
            for key in self.keys:
                if not np.all(self.data[key] == other.data[key]):
                    print('differing data for the sample ids.')
                    return False
            return True
        else:
            return True


def cli_run():
    """Command line interface

    This is the command line interface

        - to display basic info about datasets without having to code
        - to perform basic arithmetic (add multiple classes or feature sets)

    """

    path_list, add_path_list, out_path = parse_args()

    # printing info if requested
    if path_list:
        for ds_path in path_list:
            ds = MLDataset(ds_path)
            print_info(ds, ds_path)

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
    print(bname)
    print(dashes)
    print(ds)
    print(dashes)

    return


def combine_and_save(add_path_list, out_path):
    "Combines whatever datasets that can be, and save the bigger dataset to a given location."

    combined = None
    first_path = None
    for ds_path in add_path_list:
        if combined is None:
            combined = MLDataset(ds_path)
        else:
            try:
                print('Combining <1> with <2>, where \n<1> : {}\n<2> : {}'.format(ds_path, first_path))
                combined = combined + MLDataset(ds_path)
                first_path = ds_path
            except:
                print('Failed - skipping <1>')
                traceback.print_exc()

    combined.save(out_path)

    return


def get_parser():
    "Arg constructor"

    parser = argparse.ArgumentParser(prog='pyradigm')
    parser.add_argument('-i', '--info', nargs='+', action='store', dest='path_list', required=False,
                        default=None, help='List of paths to display info about.')

    parser.add_argument('-a', '--add', nargs='+', action='store', dest='add_path_list', required=False,
                        default=None, help='List of paths to MLDatasets to combine into a larger dataset.')

    parser.add_argument('-o', '--out_path', action='store', dest='out_path', required=False,
                        default=None, help='Output path to save the resulting dataset.')

    return parser


def parse_args():
    "Arg parser."

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
            raise ValueError('Output path must be specified to save the combined dataset to')

        out_path = realpath(params.out_path)
        parent_dir = dirname(out_path)
        if not pexists(parent_dir):
            os.mkdir(parent_dir)

        if len(add_path_list) < 2:
            raise ValueError('Need a minimum of datasets to combine!!')

    # removing duplicates (from regex etc)
    path_list = set(path_list)
    add_path_list = set(add_path_list)

    return path_list, add_path_list, out_path


if __name__ == '__main__':
    cli_run()