__all__ = ['RegressionDataset', ]

import random
from collections import Counter, OrderedDict
from os.path import isfile, realpath
from warnings import warn

import numpy as np
from pyradigm.base import BaseDataset


class RegressionDataset(BaseDataset):
    """
    RegressionDataset where target values are numeric (such as float or int)
    """


    def __init__(self,
                 dataset_path=None,
                 in_dataset=None,
                 data=None,
                 targets=None,
                 description='',
                 feature_names=None,
                 dtype=np.float_,
                 allow_nan_inf=False,
                 ):
        """
        Default constructor.
        Recommended way to construct the dataset is via add_samplet method,
        one samplet at a time, as it allows for unambiguous identification of
        each row in data matrix.

        This constructor can be used in 3 ways:
            - As a copy constructor to make a copy of the given in_dataset
            - Or by specifying the tuple of data, targets and classes.
                In this usage, you can provide additional inputs such as description
                and feature_names.
            - Or by specifying a file path which contains previously saved Dataset.

        Parameters
        ----------
        dataset_path : str
            path to saved Dataset on disk, to directly load it.

        in_dataset : Dataset
            Dataset to be copied to create a new one.

        data : dict
            dict of features (samplet_ids are treated to be samplet ids)

        targets : dict
            dict of targets
            (samplet_ids must match with data/classes, are treated to be samplet ids)

        description : str
            Arbitrary string to describe the current dataset.

        feature_names : list, ndarray
            List of names for each feature in the dataset.

        dtype : np.dtype
            Data type of the features to be stored

        allow_nan_inf : bool or str
            Flag to indicate whether raise an error if NaN or Infinity values are
            found. If False, adding samplets with NaN or Inf features raises an error
            If True, neither NaN nor Inf raises an error. You can pass 'NaN' or
            'Inf' to specify which value to allow depending on your needs.

        Raises
        ------
        ValueError
            If in_dataset is not of type Dataset or is empty, or
            An invalid combination of input args is given.
        IOError
            If dataset_path provided does not exist.

        """

        super().__init__(target_type=np.float_,
                         dtype=dtype,
                         allow_nan_inf=allow_nan_inf,
                         )

        if dataset_path is not None:
            if isfile(realpath(dataset_path)):
                # print('Loading the dataset from: {}'.format(dataset_path))
                self._load(dataset_path)
            else:
                raise IOError('Specified file could not be read.')
        elif in_dataset is not None:
            if not isinstance(in_dataset, self.__class__):
                raise TypeError('Invalid Class input: {} expected!'
                                ''.format(self.__class__))
            if in_dataset.num_samplets <= 0:
                raise ValueError('Dataset to copy is empty.')
            self._copy(in_dataset)
        elif data is None and targets is None:
            self._data = OrderedDict()
            self._targets = OrderedDict()
            self._num_features = 0
            self._description = ''
            self._feature_names = None
        elif data is not None and targets is not None:
            # ensuring the inputs really correspond to each other
            # but only in data and targets, not feature names
            self._validate(data, targets)

            # OrderedDict to ensure the order is maintained when
            # data/targets are returned in a matrix/array form
            self._data = OrderedDict(data)
            self._targets = OrderedDict(targets)
            self._description = description

            sample_ids = list(data)
            features0 = data[sample_ids[0]]
            self._num_features = features0.size \
                if isinstance(features0, np.ndarray) \
                else len(features0)

            # assigning default names for each feature
            if feature_names is None:
                self._feature_names = self._str_names(self.num_features)
            else:
                self._feature_names = feature_names

        else:
            raise ValueError('Incorrect way to construct the dataset.')


    @property
    def target_set(self):
        """Set of unique classes in the dataset."""

        return list(set(self._targets.values()))


    @property
    def target_sizes(self):
        """Returns the sizes of different objects in a Counter object."""
        return Counter(self._targets.values())


    @property
    def num_targets(self):
        """Total number of unique classes in the dataset."""
        return len(self.target_set)


    def samplet_ids_with_target(self, target_id):
        """
        Returns a list of samplet ids with a given target_id

        Parameters
        ----------
        target_id : float
            Value of the target to query

        Returns
        -------
        subset_ids : list
            List of samplet ids belonging to a given target_id.

        """

        subset_ids = self._keys_with_value(self.targets, target_id)
        return subset_ids


    def train_test_split_ids(self, train_perc=None, count=None):
        """
        Returns two disjoint sets of samplet ids for use in cross-validation.

        Offers two ways to specify the sizes: fraction or count.
        Only one access method can be used at a time.

        Parameters
        ----------
        train_perc : float
            fraction of samplets to build the training subset, between 0 and 1

        count : int
            exact count of samplets to build the training subset, between 1 and N-1

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
            If unrecognized format is provided for input args, or
            If the selection results in empty subsets for either train or test sets.

        """

        _ignore1, target_sizes = self.summarize()
        smallest_class_size = np.min(target_sizes)

        if count is None and (0.0 < train_perc < 1.0):
            if train_perc < 1.0 / smallest_class_size:
                raise ValueError('Training percentage selected too low '
                                 'to return even one samplet from the smallest '
                                 'class!')
            train_set = self.random_subset_ids(train_perc)
        elif train_perc is None and count > 0:
            if count >= smallest_class_size:
                raise ValueError(
                        'Selections would exclude the smallest class from test set. '
                        'Reduce samplet count per class for the training set!')
            train_set = self.random_subset_ids_by_count(count)
        else:
            raise ValueError('Invalid, or out of range selection: '
                             'only one of count or percentage '
                             'can be used to select subset at a given time.')

        test_set = list(set(self.samplet_ids) - set(train_set))

        if len(train_set) < 1:
            raise ValueError('Empty training set! Selection perc or count too small!'
                             'Change selections or check your dataset.')

        if len(test_set) < 1:
            raise ValueError('Empty test set! Selection perc or count too small!'
                             'Change selections or check your dataset.')

        return train_set, test_set


    def random_subset(self, perc=0.5):
        """
        Returns a random sub-dataset (of specified size by percentage) within each
        class.

        Parameters
        ----------
        perc : float
            Fraction of samples to be taken from the dataset

        Returns
        -------
        subdataset : RegressionDataset
            random sub-dataset of specified size.

        """

        id_subset = self.random_subset_ids(perc)
        if len(id_subset) > 0:
            return self.get_subset(id_subset)
        else:
            warn('Zero samples were selected. Returning an empty dataset!')
            return self.__class__()


    def random_subset_ids(self, perc=0.5):
        """
        Returns a random subset of sample ids, of size percentage*num_samplets

        Parameters
        ----------
        perc : float
            Fraction of the dataset

        Returns
        -------
        subset : list
            List of samplet ids

        Raises
        ------
        ValueError
            If no subjects from one or more classes were selected.
        UserWarning
            If an empty or full dataset is requested.

        """

        if perc <= 0.0:
            warn('Zero percentage requested - returning an empty dataset!')
            return list()
        elif perc >= 1.0:
            warn('Full or a larger dataset requested - returning a copy!')
            return self.samplet_ids

        # calculating the requested number of samples
        subset_size = np.int64(np.floor(self.num_samplets * perc))

        return self.random_subset_ids_by_count(subset_size)


    def random_subset_ids_by_count(self, count=5):
        """
        Returns a random subset of sample ids, of size percentage*num_samplets

        Parameters
        ----------
        count : float
            Fraction of dataset

        Returns
        -------
        subset : list
            List of samplet ids

        Raises
        ------
        ValueError
            If no subjects from one or more classes were selected.
        UserWarning
            If an empty or full dataset is requested.

        """

        if count < 1:
            warn('Zero samplets requested - returning an empty dataset!')
            return list()
        elif count >= self.num_samplets:
            warn('Full or a larger dataset requested - returning a copy!')
            return self.samplet_ids

        # clipping the range to [1, n]
        subset_size = max(1, min(self.num_samplets, count))

        # making a local copy
        id_list = list(self.samplet_ids)
        random.shuffle(id_list)
        id_subset = id_list[:subset_size]

        if len(id_subset) > 0:
            return id_subset
        else:
            warn('Zero samples were selected. Returning an empty list!')
            return list()


    def get_target(self, target_id):
        """
        Returns a smaller dataset of samplets with the requested target.

        Parameters
        ----------
        target_id : str or list
            identifier(s) of the class(es) to be returned.

        Returns
        -------
        RegressionDataset
            With subset of samples belonging to the given class(es).

        Raises
        ------
        ValueError
            If one or more of the requested classes do not exist in this dataset.
            If the specified id is empty or None

        """
        if target_id in [None, '']:
            raise ValueError("target id can not be empty or None.")

        if isinstance(target_id, self._target_type):
            target_ids = [target_id, ]
        else:
            target_ids = target_id

        non_existent = set(self.target_set).intersection(set(target_ids))
        if len(non_existent) < 1:
            raise ValueError('These classes {} do not exist in this dataset.'
                             ''.format(non_existent))

        subsets = list()
        for target_id in target_ids:
            subsets_this_class = self._keys_with_value(self._targets, target_id)
            subsets.extend(subsets_this_class)

        return self.get_subset(subsets)


    def summarize(self):
        """
        Summary of classes: names, numeric labels and sizes

        Returns
        -------
        target_set : list
            List of names of all the classes

        target_sizes : list
            Size of each class (number of samples)

        """

        target_sizes = np.zeros(len(self.target_set))
        for idx, target in enumerate(self.target_set):
            target_sizes[idx] = self.target_sizes[target]

        return self.target_set, target_sizes


    def __str__(self):
        """Returns a concise and useful text summary of the dataset."""
        full_descr = list()
        if self.description not in [None, '']:
            full_descr.append(self.description)
        if bool(self):
            full_descr.append('{} samplets, {} features'.format(
                    self.num_samplets, self.num_features))

            attr_descr = self._attr_repr()
            if len(attr_descr) > 0:
                full_descr.append(attr_descr)

            # class_ids = list(self.target_sizes)
            # max_width = max([len(cls) for cls in class_ids])
            # num_digit = max([len(str(val)) for val in self.target_sizes.values()])
            # for cls in class_ids:
            #     full_descr.append(
            #             'Class {cls:>{clswidth}} : '
            #             '{size:>{numwidth}} samplets'
            #             ''.format(cls=cls, clswidth=max_width,
            #                       size=self.target_sizes.get(cls),
            #                       numwidth=num_digit))
        else:
            full_descr.append('Empty dataset.')

        return '\n'.join(full_descr)


    def __format__(self, fmt_str='s'):
        if fmt_str.lower() in ['', 's', 'short']:
            descr = '{} samplets x {} features each'.format(
                    self.num_samplets, self.num_features)

            attr_descr = self._attr_repr()
            if len(attr_descr) > 0:
                descr += '\n {}'.format(attr_descr)

            return descr
        elif fmt_str.lower() in ['f', 'full']:
            return self.__str__()
        else:
            raise NotImplementedError("Requested type of format not implemented.\n"
                                      "It can only be 'short' (default) or 'full', "
                                      "or a shorthand: 's' or 'f' ")


    def __repr__(self):
        return self.__str__()


    def __dir__(self):
        """"""

        return ['add_attr',
                'add_dataset_attr',
                'add_samplet',
                'attr',
                'attr_summary',
                'compatible',
                'data',
                'data_and_targets',
                'dataset_attr',
                'del_samplet',
                'description',
                'dtype',
                'extend',
                'feature_names',
                'from_arff',
                'get',
                'get_data_matrix_in_order',
                'get_feature_subset',
                'get_subset',
                'get_target',
                'glance',
                'num_features',
                'num_samplets',
                'num_targets',
                'random_subset',
                'random_subset_ids',
                'random_subset_ids_by_count',
                'samplet_ids_with_target',
                'samplet_ids',
                'save',
                'shape',
                'summarize',
                'target_set',
                'target_sizes',
                'targets',
                'train_test_split_ids',
                'transform']
