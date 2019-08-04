"""

Base module to define the ABCs for all MLDatasets

"""

import copy
import os
import pickle
import random
import warnings
from warnings import warn
from collections import Counter, OrderedDict, Sequence
from itertools import islice
from os.path import isfile, realpath
from sys import version_info
import numpy as np
import types

from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """Abstract Base class for Dataset.

    samplet is a term reserved to referred to a single row in feature matrix X: N x p

    self.__class__() refers to the inherited child class instance at runtime!

    """

    def __init__(self,
                 target_type=float,
                 dtype=float,
                 allow_nan_inf=False,
                 encode_nonnumeric=False,
                 ):
        """Init for the ABC to define the type and properties of the Dataset

        Parameters
        -----------
        target_type : type, callable
            Data type of the target for the child class.
            Must be callable that takes in a datatype and converts to its own type.

        dtype : np.dtype
            Data type of the features to be stored

        allow_nan_inf : bool or str
            Flag to indicate whether raise an error if NaN or Infinity values are
            found. If False, adding samplets with NaN or Inf features raises an error
            If True, neither NaN nor Inf raises an error. You can pass 'NaN' or
            'Inf' to specify which value to allow depending on your needs.
        """

        if not callable(target_type):
            raise TypeError('target type must be callable, to allow for conversion!')
        else:
            self._target_type = target_type

        if np.issubdtype(dtype, np.generic):
            self._dtype = dtype

        if not isinstance(allow_nan_inf, (bool, str)):
            raise TypeError('allow_nan_inf flag can only be bool or str')
        else:
            self._allow_nan_inf = allow_nan_inf

        if not isinstance(encode_nonnumeric, bool):
            raise TypeError('encode_nonnumeric flag can only be bool')
        else:
            self._encode_nonnumeric = encode_nonnumeric


    @property
    def data(self):
        """data in its original dict form."""
        return self._data


    def data_and_targets(self):
        """
        Dataset features and targets in a matrix form for learning.

        Also returns samplet_ids in the same order.

        Returns
        -------
        data_matrix : ndarray
            2D array of shape [num_samplets, num_features]
            with features corresponding row-wise to samplet_ids
        targets : ndarray
            Array of numeric targets for each samplet corresponding row-wise to samplet_ids
        samplet_ids : list
            List of samplet ids

        """

        sample_ids = np.array(self.samplet_ids)
        label_dict = self.targets
        matrix = np.full([self.num_samplets, self.num_features], np.nan)
        targets = np.empty([self.num_samplets, 1], dtype=self._target_type)
        for ix, samplet in enumerate(sample_ids):
            matrix[ix, :] = self._data[samplet]
            targets[ix] = label_dict[samplet]

        return matrix, np.ravel(targets), sample_ids


    @data.setter
    def data(self, values, feature_names=None):
        """
        Populates this dataset with the provided data.
        Usage of this method is discourage (unless you know what you are doing).

        Parameters
        ----------
        values : dict
            dict of features keyed in by samplet ids.

        feature_names : list of str
            New feature names for the new features, if available.

        Raises
        ------
        ValueError
            If number of samplets does not match the size of existing set, or
            If atleast one samplet is not provided.

        """
        if isinstance(values, dict):
            if self._targets is not None and len(self._targets) != len(values):
                raise ValueError(
                    'number of samplets do not match the previously assigned targets')
            elif len(values) < 1:
                raise ValueError('There must be at least 1 samplet in the dataset!')
            else:
                self._data = values
                # update dimensionality
                # assuming all samplet_ids in dict have same len arrays
                self._num_features = len(values[self.samplet_ids[0]])

            if feature_names is None:
                self._feature_names = self._str_names(self.num_features)
            else:
                self.feature_names = feature_names
        else:
            raise ValueError('data input must be a dictionary!')


    @property
    def targets(self):
        """Returns the array of targets for all the samplets."""
        # TODO numeric label need to be removed,
        # as this can be made up on the fly as needed from str to num encoders.
        return self._targets


    @targets.setter
    def targets(self, values):
        """Class targets (such as 1, 2, -1, 'A', 'B' etc.) for each samplet in the dataset."""
        if isinstance(values, dict):
            if self._data is not None and len(self._data) != len(values):
                raise ValueError(
                    'number of samplets do not match the previously assigned data')
            elif set(self.samplet_ids) != set(list(values)):
                raise ValueError('samplet ids do not match the previously assigned ids.')
            else:
                self._targets = values
        else:
            raise ValueError('targets input must be a dictionary!')


    @property
    def feature_names(self):
        "Returns the feature names as an numpy array of strings."

        return self._feature_names


    @feature_names.setter
    def feature_names(self, names):
        "Stores the text targets for features"

        if len(names) != self.num_features:
            raise ValueError("Number of names do not match the number of features!")
        if not isinstance(names, (Sequence, np.ndarray, np.generic)):
            raise ValueError("Input is not a sequence. "
                             "Ensure names are in the same order "
                             "and length as features.")

        self._feature_names = np.array(names)

    @staticmethod
    def __take(nitems, iterable):
        """Return first n items of the iterable as a list"""
        return dict(islice(iterable, int(nitems)))


    @staticmethod
    def _str_names(num):

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
        nitems = max([1, min([nitems, self.num_samplets - 1])])
        return self.__take(nitems, iter(self._data.items()))


    @abstractmethod
    def summarize(self):
        """Method to summarize the sample inside in an appropriate way!"""


    def _check_target(self, target_value):
        """
        Method to ensure target to be added is not empty and vectorized.

        Parameters
        ----------
        target_value
            Value of the target

        Returns
        -------
        target_value
            Target value in the right data type

        Raises
        ------
        TypeError
            If input target is not of the right data type for this class.
        """

        if not isinstance(target_value, self._target_type):
            try:
                target_value = self._target_type(target_value)
            except:
                raise TypeError('Invalid type of target {} :'
                                ' must be of type {}, or be convertible to it'
                                ''.format(type(target_value), self._target_type))

        return target_value


    @abstractmethod
    def _check_features(self, features):
        """
        Method to ensure features to be added are valid (non-empty, vectorized etc)
        """


    def _check_id(self, samplet_id):
        """
        Method to validate the samplet ID
        """

        if not isinstance(samplet_id, str):
            return str(samplet_id)
        else:
            return samplet_id


    def add_samplet(self,
                    samplet_id,
                    features,
                    target,
                    overwrite=False,
                    feature_names=None):
        """Adds a new samplet to the dataset with its features, label and class ID.

        This is the preferred way to construct the dataset.

        Parameters
        ----------

        samplet_id : str
            An identifier uniquely identifies this samplet.
        features : list, ndarray
            The features for this samplet
        target : int, str
            The label for this samplet
        overwrite : bool
            If True, allows the overwite of features for an existing subject ID.
            Default : False.
        feature_names : list
            The names for each feature. Assumed to be in the same order as `features`

        Raises
        ------
        ValueError
            If `samplet_id` is already in the Dataset (and overwrite=False), or
            If dimensionality of the current samplet does not match the current, or
            If `feature_names` do not match existing names
        TypeError
            If samplet to be added is of different data type compared to existing
            samplets.

        """

        samplet_id = self._check_id(samplet_id)

        if samplet_id in self._data and not overwrite:
            raise ValueError('{} already exists in this dataset!'.format(samplet_id))

        features = self._check_features(features)
        target = self._check_target(target)

        if self.num_samplets <= 0:
            self._data[samplet_id] = features
            self._targets[samplet_id] = target
            self._num_features = features.size if isinstance(features,
                                                             np.ndarray) else len(
                features)
            if feature_names is None:
                self._feature_names = self._str_names(self.num_features)
        else:
            if self._num_features != features.size:
                raise ValueError('dimensionality of this samplet ({}) '
                                 'does not match existing samplets ({})'
                                 ''.format(features.size, self._num_features))

            self._data[samplet_id] = features
            self._targets[samplet_id] = target
            if feature_names is not None:
                # if it was never set, allow it
                # class gets here when adding the first samplet,
                #   after dataset was initialized with empty constructor
                if self._feature_names is None:
                    self._feature_names = np.array(feature_names)
                else:  # if set already, ensure a match
                    if not np.array_equal(self.feature_names, np.array(feature_names)):
                        raise ValueError(
                            "supplied feature names do not match the existing names!")


    def del_samplet(self, sample_id):
        """
        Method to remove a samplet from the dataset.

        Parameters
        ----------
        sample_id : str
            samplet id to be removed.

        Raises
        ------
        UserWarning
            If samplet id to delete was not found in the dataset.

        """
        if sample_id not in self._data:
            warn('Sample to delete not found in the dataset - nothing to do.')
        else:
            self._data.pop(sample_id)
            self._targets.pop(sample_id)
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
        Dataset : Dataset
            with subset of features requested.

        Raises
        ------
        UnboundLocalError
            If input indices are out of bounds for the dataset.

        """

        subset_idx = np.asarray(subset_idx)
        if not (max(subset_idx) < self._num_features) and (min(subset_idx) >= 0):
            raise UnboundLocalError('indices out of range for the dataset. '
                                    'Max index: {} Min index : 0'.format(
                self._num_features))

        sub_data = {samplet: features[subset_idx] for samplet, features in
                    self._data.items()}
        new_descr = 'Subset features derived from: \n ' + self._description
        subdataset = self.__class__(data=sub_data,
                                    targets=self._targets,
                                    description=new_descr,
                                    feature_names=self._feature_names[subset_idx])

        return subdataset


    @staticmethod
    def keys_with_value(dictionary, value):
        "Returns a subset of keys from the dict with the value supplied."

        subset = [key for key in dictionary if dictionary[key] == value]

        return subset



    def transform(self, func, func_description=None):
        """
        Applies a given a function to the features of each subject
            and returns a new dataset with other info unchanged.

        Parameters
        ----------
        func : callable
            A callable that takes in a single ndarray and returns a single ndarray.
            Ensure the transformed dimensionality must be the same for all subjects.

            If your function requires more than one argument,
            use `functools.partial` to freeze all the arguments
            except the features for the subject.

        func_description : str, optional
            Human readable description of the given function.

        Returns
        -------
        xfm_ds : Dataset
            with features obtained from subject-wise transform

        Raises
        ------
        TypeError
            If given func is not a callable
        ValueError
            If transformation of any of the subjects features raises an exception.

        Examples
        --------
        Simple:

        .. code-block:: python

            from pyradigm import Dataset

            thickness = Dataset(in_path='ADNI_thickness.csv')
            pcg_thickness = thickness.apply_xfm(func=get_pcg, description = 'applying ROI mask for PCG')
            pcg_median = pcg_thickness.apply_xfm(func=np.median, description='median per subject')


        Complex example with function taking more than one argument:

        .. code-block:: python

            from pyradigm import Dataset
            from functools import partial
            import hiwenet

            thickness = Dataset(in_path='ADNI_thickness.csv')
            roi_membership = read_roi_membership()
            hw = partial(hiwenet, groups = roi_membership)

            thickness_hiwenet = thickness.transform(func=hw, description = 'histogram weighted networks')
            median_thk_hiwenet = thickness_hiwenet.transform(func=np.median, description='median per subject')

        """

        if not callable(func):
            raise TypeError('Given function {} is not a callable'.format(func))

        xfm_ds = self.__class__()
        for samplet, data in self._data.items():
            try:
                xfm_data = func(data)
            except:
                print('Unable to transform features for {}. '
                      'Quitting.'.format(samplet))
                raise

            xfm_ds.add_samplet(samplet, xfm_data,
                               target=self._targets[samplet],
                               class_id=self._targets[samplet])

        xfm_ds.description = "{}\n{}".format(func_description, self._description)

        return xfm_ds


    def train_test_split_ids(self, train_perc=None, count_per_class=None):
        """
        Returns two disjoint sets of samplet ids for use in cross-validation.

        Offers two ways to specify the sizes: fraction or count.
        Only one access method can be used at a time.

        Parameters
        ----------
        train_perc : float
            fraction of samplets from each class to build the training subset.

        count_per_class : int
            exact count of samplets from each class to build the training subset.

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

        _ignore1, target_sizes = self.summarize()
        smallest_class_size = np.min(target_sizes)

        if count_per_class is None and (0.0 < train_perc < 1.0):
            if train_perc < 1.0 / smallest_class_size:
                raise ValueError('Training percentage selected too low '
                                 'to return even one samplet from the smallest class!')
            train_set = self.random_subset_ids(perc_per_class=train_perc)
        elif train_perc is None and count_per_class > 0:
            if count_per_class >= smallest_class_size:
                raise ValueError(
                    'Selections would exclude the smallest class from test set. '
                    'Reduce samplet count per class for the training set!')
            train_set = self.random_subset_ids_by_count(count_per_class=count_per_class)
        else:
            raise ValueError('Invalid, or out of range selection: '
                             'only one of count or percentage '
                             'can be used to select subset.')

        test_set = list(set(self.samplet_ids) - set(train_set))

        if len(train_set) < 1 or len(test_set) < 1:
            raise ValueError('Selection resulted in empty training or test set: '
                             'check your selections or dataset!')

        return train_set, test_set



    @abstractmethod
    def random_subset(self, perc=0.5):
        """
        Returns a random sub-dataset of specified size by percentage

        Parameters
        ----------
        perc : float
            Fraction of samplets to be taken
            The meaning of this varies based on the child class: for
            classification- oriented Dataset, this can be perc from each class.

        Returns
        -------
        subdataset : Dataset
            random sub-dataset of specified size.

        """



    def get_subset(self, subset_ids):
        """
        Returns a smaller dataset identified by their keys/samplet IDs.

        Parameters
        ----------
        subset_ids : list
            List od samplet IDs to extracted from the dataset.

        Returns
        -------
        sub-dataset : Dataset
            sub-dataset containing only requested samplet IDs.

        """

        num_existing_keys = sum([1 for key in subset_ids if key in self._data])
        if subset_ids is not None and num_existing_keys > 0:
            # ensure items are added to data, targets etc in the same order of samplet IDs
            # TODO come up with a way to do this even when not using OrderedDict()
            # putting the access of data, targets and classes in the same loop  would
            # ensure there is correspondence across the three attributes of the class
            data = self.__get_subset_from_dict(self._data, subset_ids)
            targets = self.__get_subset_from_dict(self._targets, subset_ids)
            subdataset = self.__class__(data=data, targets=targets)
            # Appending the history
            subdataset.description += '\n Subset derived from: ' + self.description
            subdataset.feature_names = self._feature_names
            subdataset._dtype = self.dtype
            return subdataset
        else:
            warn('subset of IDs requested do not exist in the dataset!')
            return self.__class__()


    def get_data_matrix_in_order(self, subset_ids):
        """
        Returns a numpy array of features, rows in the same order as subset_ids

        Parameters
        ----------
        subset_ids : list
            List od samplet IDs to extracted from the dataset.

        Returns
        -------
        matrix : ndarray
            Matrix of features, for each id in subset_ids, in order.
        """

        if len(subset_ids) < 1:
            warn('subset must have atleast one ID - returning empty matrix!')
            return np.empty((0, 0))

        if isinstance(subset_ids, set):
            raise TypeError('Input set is not ordered, hence can not guarantee order! '
                            'Must provide a list or tuple.')

        if isinstance(subset_ids, str):
            subset_ids = [subset_ids, ]

        num_existing_keys = sum([1 for key in subset_ids if key in self._data])
        if num_existing_keys < len(subset_ids):
            raise ValueError('One or more IDs from subset do not exist in the dataset!')

        matrix = np.full((num_existing_keys, self.num_features), np.nan)
        for idx, sid in enumerate(subset_ids):
            matrix[idx, :] = self._data[sid]

        return matrix


    def __contains__(self, item):
        "Boolean test of membership of a samplet in the dataset."
        if item in self.samplet_ids:
            return True
        else:
            return False


    def get(self, item, not_found_value=None):
        "Method like dict.get() which can return specified value if key not found"

        if item in self.samplet_ids:
            return self._data[item]
        else:
            return not_found_value


    def __getitem__(self, item):
        "Method to ease data retrieval i.e. turn dataset.data['id'] into dataset['id'] "

        if item in self.samplet_ids:
            return self._data[item]
        else:
            raise KeyError('{} not found in dataset.'.format(item))


    def __setitem__(self, item, features):
        """Method to replace features for existing samplet"""

        if item in self._data:
            features = self._check_features(features)
            if self._num_features != features.size:
                raise ValueError('dimensionality of supplied features ({}) '
                                 'does not match existing samplets ({})'
                                 ''.format(features.size, self._num_features))
            self._data[item] = features
        else:
            raise KeyError('{} not found in dataset.'
                           ' Can not replace features of a non-existing samplet.'
                           ' Add it first via .add_samplet()'.format(item))

    def __iter__(self):
        "Iterator over samplets"

        for samplet, features in self.data.items():
            yield samplet, features


    @staticmethod
    def __get_subset_from_dict(input_dict, subset):
        # Using OrderedDict helps ensure data are added to data, targets etc
        # in the same order of samplet IDs
        return OrderedDict(
                (sid, value) for sid, value in input_dict.items() if sid in subset)


    @property
    def samplet_ids(self):
        """Sample identifiers (strings) - the basis of Dataset (same as samplet_ids)"""
        return list(self._data)


    @property
    def samplet_ids(self):
        """Sample identifiers (strings) forming the basis of Dataset (same as keys)."""
        return list(self._data)


    @property
    def description(self):
        """Text description (header) that can be set by user."""
        return self._description


    @description.setter
    def description(self, str_val):
        """Text description that can be set by user."""
        if not str_val: raise ValueError('description can not be empty')
        self._description = str_val


    @property
    def num_features(self):
        """number of features in each samplet."""
        return np.int64(self._num_features)


    @num_features.setter
    def num_features(self, int_val):
        "Method that should not exist!"
        raise AttributeError("num_features property can't be set, only retrieved!")


    @property
    def dtype(self):
        """Returns the data type of the features in the Dataset"""

        return self._dtype

    @dtype.setter
    def dtype(self, type_val):
            raise SyntaxError('Data type can only be set during initialization!')

    @property
    def num_samplets(self):
        """number of samplets in the entire dataset."""
        if self._data is not None:
            return len(self._data)
        else:
            return 0


    @property
    def shape(self):
        """Returns the pythonic shape of the dataset: num_samplets x num_features.
        """

        return (self.num_samplets, self.num_features)


    def __len__(self):
        return self.num_samplets


    def __nonzero__(self):
        if self.num_samplets < 1:
            return False
        else:
            return True

    @abstractmethod
    def __str__(self):
        """Returns a concise and useful text summary of the dataset."""

    @abstractmethod
    def __format__(self, fmt_str='s'):
        """Returns variants of str repr to be used in .format() invocations"""

    @abstractmethod
    def __repr__(self):
        """Evaluatable repr"""

    def _copy(self, other):
        """Copy constructor."""
        self._data = copy.deepcopy(other.data)
        self._targets = copy.deepcopy(other.targets)
        self._dtype = copy.deepcopy(other.dtype)
        self._description = copy.deepcopy(other.description)
        self._feature_names = copy.deepcopy(other.feature_names)
        self._num_features = copy.deepcopy(other.num_features)

        return self


    @classmethod
    def from_arff(cls, arff_path, encode_nonnumeric=False):
        """Loads a given dataset saved in Weka's ARFF format.

        Parameters
        ----------

        arff_path : str
            Path to a dataset saved in Weka's ARFF file format.

        encode_nonnumeric : bool
            Flag to specify whether to encode non-numeric features (categorical,
            nominal or string) features to numeric values.
            Currently used only when importing ARFF files.
            It is usually better to encode your data at the source,
            and then import them. Use with caution!

        """
        try:
            from scipy.io.arff import loadarff
            arff_data, arff_meta = loadarff(arff_path)
        except:
            raise ValueError('Error loading the ARFF dataset!')

        attr_names = arff_meta.names()[:-1]  # last column is class
        attr_types = arff_meta.types()[:-1]
        if not encode_nonnumeric:
            # ensure all the attributes are numeric
            uniq_types = set(attr_types)
            if 'numeric' not in uniq_types:
                raise ValueError(
                    'Currently only numeric attributes in ARFF are supported!')

            non_numeric = uniq_types.difference({'numeric'})
            if len(non_numeric) > 0:
                raise ValueError('Non-numeric features provided ({}), '
                                 'without requesting encoding to numeric. '
                                 'Try setting encode_nonnumeric=True '
                                 'or encode features to numeric!'.format(non_numeric))
        else:
            raise NotImplementedError(
                'encoding non-numeric features to numeric is not implemented yet! '
                'Encode features beforing to ARFF.')

        dataset = cls()
        dataset._description = arff_meta.name

        # initializing the key containers, before calling self.add_samplet
        dataset._data = OrderedDict()
        dataset._targets = OrderedDict()
        dataset._targets = OrderedDict()

        num_samples = len(arff_data)
        num_digits = len(str(num_samples))
        make_id = lambda index: 'row{index:0{nd}d}'.format(index=index, nd=num_digits)
        sample_classes = [cls.decode('utf-8') for cls in arff_data['class']]
        class_set = set(sample_classes)
        label_dict = dict()
        # encoding class names to targets 1 to n
        for ix, cls in enumerate(class_set):
            label_dict[cls] = ix + 1

        for index in range(num_samples):
            samplet = arff_data.take([index])[0].tolist()
            sample_attrs = samplet[:-1]
            sample_class = samplet[-1].decode('utf-8')
            dataset.add_samplet(samplet_id=make_id(index),  # ARFF rows do not have an ID
                                features=sample_attrs,
                                target=sample_class)
            # not necessary to set feature_names=attr_names for each samplet,
            # as we do it globally after loop

        dataset._feature_names = attr_names

        return dataset


    def _load(self, path):
        """Method to load the serialized dataset from disk."""
        try:
            path = os.path.abspath(path)
            with open(path, 'rb') as df:
                # loaded_dataset = pickle.load(df)
                self._data, self._targets, \
                self._dtype, self._description, \
                self._num_features, self._feature_names = pickle.load(df)

            # ensure the loaded dataset is valid
            self._validate(self._data, self._targets)

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

        # TODO need a file format that is flexible and efficient to allow the following:
        #   1) being able to read just meta info without having to load the ENTIRE dataset
        #       i.e. use case: compatibility check with #subjects, ids and their classes
        #   2) random access layout: being able to read features for a single subject!

        try:
            file_path = os.path.abspath(file_path)
            with open(file_path, 'wb') as df:
                # pickle.dump(self, df)
                pickle.dump((self._data, self._targets,
                             self._dtype, self._description,
                             self._num_features, self._feature_names),
                            df)
            return
        except IOError as ioe:
            raise IOError('Unable to save the dataset to file: {}', format(ioe))
        except:
            raise


    @staticmethod
    def _validate(data, targets):
        "Validator of inputs."

        if not isinstance(data, dict):
            raise TypeError(
                'data must be a dict! keys: samplet ID or any unique identifier')
        if not isinstance(targets, dict):
            raise TypeError(
                'targets must be a dict! keys: samplet ID or any unique identifier')

        if not len(data) == len(targets):
            raise ValueError('Lengths of data, targets and classes do not match!')
        if not set(list(data)) == set(list(targets)):
            raise ValueError("data and targets dict's must have the same keys!")

        num_features_in_elements = [samplet.size for samplet in data.values()]
        if len(np.unique(num_features_in_elements)) > 1:
            raise ValueError('Different samplets have different number of features - '
                              'invalid!')

        return True


    def extend(self, other):
        """
        Method to extend the dataset vertically (add samplets from  anotehr dataset).

        Parameters
        ----------
        other : Dataset
            second dataset to be combined with the current
            (different samplets, but same dimensionality)

        Raises
        ------
        TypeError
            if input is not an Dataset.
        """

        if not isinstance(other, self.__class__):
            raise TypeError('Incorrect type of dataset provided!')
        # assert self.__dtype==other.dtype, TypeError('Incorrect data type of features!')
        for samplet in other.samplet_ids:
            self.add_samplet(samplet, other.data[samplet], other.targets[samplet])

        # TODO need a mechanism add one feature at a time, and
        #   consequently update feature names for any subset of features


    def __add__(self, other):
        "Method to combine to MLDatasets, samplet-wise or feature-wise."

        if not isinstance(other, self.__class__):
            raise TypeError('Incorrect type of dataset provided!')

        if set(self.samplet_ids) == set(other.samplet_ids):
            print('Identical keys found. '
                  'Trying to horizontally concatenate features for each samplet.')

            if other.num_features < 1:
                raise ValueError('No features to concatenate.')
            # making an empty dataset
            combined = self.__class__()
            # populating it with the concatenated feature set
            for samplet in self.samplet_ids:
                comb_data = np.concatenate([self._data[samplet], other.data[samplet]])
                combined.add_samplet(samplet, comb_data, self._targets[samplet])

            comb_names = np.concatenate([self._feature_names, other.feature_names])
            combined.feature_names = comb_names

            return combined

        elif len(set(self.samplet_ids).intersection(
                other.samplet_ids)) < 1 and self._num_features == other.num_features:
            # making a copy of self first
            combined = self.__class__(in_dataset=self)
            # adding the new dataset
            combined.extend(other)
            return combined
        else:
            raise ArithmeticError('Two datasets could not be combined.')


    def __sub__(self, other):
        """Removing one dataset from another."""
        if not isinstance(other, type(self)):
            raise TypeError('Incorrect type of dataset provided!')

        num_existing_keys = len(set(self.samplet_ids).intersection(other.samplet_ids))
        if num_existing_keys < 1:
            warn('None of the samplet ids to be removed found in this dataset '
                          '- nothing to do.')
        if len(self.samplet_ids) == num_existing_keys:
            warn('Requested removal of all the samplets - '
                          'output dataset would be empty.')

        removed = copy.deepcopy(self)
        for samplet in other.samplet_ids:
            removed.del_samplet(samplet)

        return removed


    def __iadd__(self, other):
        """Augmented assignment for add."""
        return self.__add__(other)


    def __isub__(self, other):
        """Augmented assignment for samplet."""
        return self.__sub__(other)


    def __eq__(self, other):
        """Equality of two datasets in samplets and their values."""
        if set(self.samplet_ids) != set(other.samplet_ids):
            print('differing samplet ids.')
            return False
        elif id(self._data) != id(other.data):
            for key in self.samplet_ids:
                if not np.all(self.data[key] == other.data[key]):
                    print('differing data for the samplet ids.')
                    return False
            return True
        else:
            return True


    @abstractmethod
    def compatible(self, another):
        """
        Checks whether the input dataset is compatible with the current instance:
        i.e. with same set of subjects, each beloning to the same class.

        Parameters
        ----------
        dataset : MLdataset or similar

        Returns
        -------
        compatible : bool
            Boolean flag indicating whether two datasets are compatible or not
        """


def check_compatibility_BaseDataset(datasets, reqd_num_features=None):
    """
    Checks whether the given MLdataset instances are compatible

    i.e. with same set of subjects, each beloning to the same class in all instances.

    Checks the first dataset in the list against the rest, and returns a boolean array.

    Parameters
    ----------
    datasets : Iterable
        A list of n datasets

    reqd_num_features : int
        The required number of features in each dataset.
        Helpful to ensure test sets are compatible with training set,
            as well as within themselves.

    Returns
    -------
    all_are_compatible : bool
        Boolean flag indicating whether all datasets are compatible or not

    compatibility : list
        List indicating whether first dataset is compatible with the rest individually.
        This could be useful to select a subset of mutually compatible datasets.
        Length : n-1

    dim_mismatch : bool
        Boolean flag indicating mismatch in dimensionality from that specified

    size_descriptor : tuple
        A tuple with values for (num_samplets, reqd_num_features)
        - num_samplets must be common for all datasets that are evaluated for compatibility
        - reqd_num_features is None (when no check on dimensionality is perfomed), or
            list of corresponding dimensionalities for each input dataset

    """

    from collections import Iterable
    if not isinstance(datasets, Iterable):
        raise TypeError('Input must be an iterable '
                        'i.e. (list/tuple) of MLdataset/similar instances')

    datasets = list(datasets)  # to make it indexable if coming from a set
    num_datasets = len(datasets)

    check_dimensionality = False
    dim_mismatch = False
    if reqd_num_features is not None:
        if isinstance(reqd_num_features, Iterable):
            if len(reqd_num_features) != num_datasets:
                raise ValueError('Specify dimensionality for exactly {} datasets.'
                                 ' Given for a different number {}'
                                 ''.format(num_datasets, len(reqd_num_features)))
            reqd_num_features = list(map(int, reqd_num_features))
        else:  # same dimensionality for all
            reqd_num_features = [int(reqd_num_features)] * num_datasets

        check_dimensionality = True
    else:
        # to enable iteration
        reqd_num_features = [None,] * num_datasets

    pivot = datasets[0]
    if not isinstance(pivot, Dataset):
        pivot = Dataset(pivot)

    if check_dimensionality and pivot.num_features != reqd_num_features[0]:
        warn('Dimensionality mismatch! Expected {} whereas current {}.'
                      ''.format(reqd_num_features[0], pivot.num_features))
        dim_mismatch = True

    compatible = list()
    for ds, reqd_dim in zip(datasets[1:], reqd_num_features[1:]):
        if not isinstance(ds, Dataset):
            ds = Dataset(ds)

        is_compatible = True
        # compound bool will short-circuit, not optim required
        if pivot.num_samplets != ds.num_samplets \
                or pivot.samplet_ids != ds.samplet_ids:
            is_compatible = False

        if check_dimensionality and reqd_dim != ds.num_features:
            warn('Dimensionality mismatch! Expected {} whereas current {}.'
                          ''.format(reqd_dim, ds.num_features))
            dim_mismatch = True

        compatible.append(is_compatible)

    return all(compatible), compatible, dim_mismatch, \
           (pivot.num_samplets, reqd_num_features)
