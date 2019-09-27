"""

Base module to define the ABCs for all MLDatasets

"""

import copy
import os
import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from collections import OrderedDict
from itertools import islice
from warnings import warn

import numpy as np


class PyradigmException(Exception):
    """Custom exception to highlight pyradigm-specific issues."""
    pass


class EmptyFeatureSetException(PyradigmException):
    """Custom exception to catch empty feature set"""


class ConstantValuesException(PyradigmException):
    """Custom exception to indicate the exception of catching all constant values
    for a given samplet, for a specific feature across the samplets"""


class InfiniteOrNaNValuesException(PyradigmException):
    """Custom exception to catch NaN or Inf values."""


def is_iterable_but_not_str(value):
    """Boolean check for iterables that are not strings"""

    return not isinstance(value, str) and isinstance(value, Iterable)


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

        # samplet-wise attributes
        self._attr = dict()
        self._attr_dtype = dict()
        # dataset-wise attributes, common to all samplets
        self._dataset_attr = dict()


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
        matrix = np.full([self.num_samplets, self.num_features], np.nan)
        # dtype=object allows for variable length strings!
        targets = np.empty([self.num_samplets, 1], dtype=object)
        for ix, samplet in enumerate(sample_ids):
            matrix[ix, :] = self._data[samplet]
            targets[ix] = self.targets[samplet]

        return matrix, np.ravel(targets).astype(self._target_type), sample_ids


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
        if not is_iterable_but_not_str(names):
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


    def _check_features(self, features):
        """
        Method to ensure features to be added are not empty and vectorized.

        Parameters
        ----------
        features : iterable
            Any data that can be converted to a numpy array.

        Returns
        -------
        features : numpy array
            Flattened non-empty numpy array.

        Raises
        ------
        ValueError
            If input data is empty.
        """

        if not isinstance(features, np.ndarray):
            features = np.asarray(features)

        try:
            features = features.astype(self.dtype)
        except:
            raise TypeError("Input features (of dtype {}) can not be converted to "
                            "Dataset's data type {}"
                            "".format(features.dtype, self.dtype))

        if features.size <= 0:
            raise EmptyFeatureSetException('Provided features are empty.')

        if not self._allow_nan_inf:
            if np.isnan(features).any() or np.isinf(features).any():
                raise InfiniteOrNaNValuesException('NaN or Inf values found!'
                                 ' They are not allowed and disabled by default.'
                                 ' Use allow_nan_inf=True if you need to use them.')

        if features.ndim > 1:
            features = np.ravel(features)

        return features


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
                    feature_names=None,
                    attr_names=None,
                    attr_values=None):
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

        attr_names : str or list of str
            Name of the attribute to be added for this samplet

        attr_values : generic or list of generic
            Value of the attribute. Any data type allowed as long as they are
            compatible across all the samplets in this dataset.

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

            if attr_names is not None:
                if is_iterable_but_not_str(attr_names):
                    if len(attr_names) != len(attr_values) or \
                            (not is_iterable_but_not_str(attr_values)):
                        raise ValueError('When you supply a list for attr_names, '
                                         'attr_values also must be a list of same '
                                         'length')
                    for name, value in zip(attr_names, attr_values):
                        self.add_attr(name, samplet_id, value)
                else:
                    self.add_attr(attr_names, samplet_id, attr_values)


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


    def add_dataset_attr(self, attr_name, attr_value):
        """
        Adds dataset-wide attributes (common to all samplets).

        This is a great way to add meta data (such as version, processing details,
        anything else common to all samplets). This is better than encoding the info
        into the description field, as this allow programmatic retrieval.

        Parameters
        ----------
        attr_name : str
            Identifier for the attribute

        attr_value : object
            Value of the attribute (any datatype)

        """

        self._dataset_attr[attr_name] = attr_value


    def add_attr(self, attr_name, samplet_id, attr_value):
        """
        Method to add samplet-wise attributes to the dataset.

        Note: attribute values get overwritten by default, if they already exist
        for a given samplet

        Parameters
        ----------
        attr_name : str
            Name of the attribute to be added.

        samplet_id : str or list of str
            Identifier(s) for the samplet

        attr_value : generic or list of generic
            Value(s) of the attribute. Any data type allowed, although it is
            strongly recommended to keep the data type the same, or compatible,
            across all the samplets in this dataset.

        """

        if attr_name is not None:

            if attr_name not in self._attr:
                self._attr[attr_name] = dict()
                self._attr_dtype[attr_name] = None

            if is_iterable_but_not_str(samplet_id):
                if not isinstance(attr_value, (Sequence, np.ndarray, np.generic)):
                    raise TypeError('When samplet_id is a list, attr_value must '
                                    'also be a list')
                if len(samplet_id) != len(attr_value):
                    raise ValueError('Number of attribute values provided do not '
                                     'match the number of samplet IDs')

                for sid, val in zip(samplet_id, attr_value):
                    self.__add_single_attr(attr_name, sid, val)

            else:
                if is_iterable_but_not_str(attr_value):
                    raise TypeError('When samplet_id is not a list, attr_value also '
                                    'must not be a list')

                self.__add_single_attr(attr_name, samplet_id, attr_value)

        else:
            raise ValueError('Attribute name can not be None!')


    def __add_single_attr(self, attr_name, samplet_id, attr_value):
        """Actual attr adder."""

        if samplet_id not in self._data:
            raise KeyError('Samplet {} does not exist in this Dataset.'
                           'Add it first via .add_samplet() method.')

        if self._attr_dtype[attr_name] is not None:
            if not np.issubdtype(type(attr_value), self._attr_dtype[attr_name]):
                raise TypeError('Datatype of attribute {} is expected to be {}. '
                                'Value provided is of type: {}'
                                ''.format(attr_name, self._attr_dtype[attr_name],
                                          type(attr_value)))
        else:
            self._attr_dtype[attr_name] = type(attr_value)

        self._attr[attr_name][samplet_id] = attr_value


    @property
    def attr(self):
        """Returns attributes dictionary, keyed-in by [attr_name][samplet_id]"""

        return self._attr


    @attr.setter
    def attr(self, values):
        """Batch setter of attributes via dict of dicts"""

        if isinstance(values, dict):
            for attr_name in values.keys():
                this_attr = values[attr_name]
                if not isinstance(this_attr, dict):
                    raise TypeError('Value of attr {} must be a dict keyed in by '
                                    'samplet ids.'.format(attr_name))
                if len(this_attr) < 1:
                    warn('Attribute {} is empty. Ignoring it'.format(attr_name))
                    continue

                existing_ids = set(self.samplet_ids).intersection(set(list(this_attr)))
                if len(existing_ids) < 1:
                    raise ValueError('None of the samplets set for attr {} exist '
                                     'in dataset!'.format(attr_name))

                self._attr[attr_name] = self.__get_subset_from_dict(
                        this_attr, existing_ids)
        else:
            raise ValueError('attrs input must be a non-empty dict of dicts! '
                             'Top level key must be names of attributes. '
                             'Inner dicts must be keyed in by samplet ids.')


    def get_attr(self, attr_name, samplet_ids='all'):
        """
        Method to retrieve specified attribute for a list of samplet IDs

        Parameters
        ----------
        attr_name : str
            Name of the attribute

        samplet_ids : str or list
            One or more samplet IDs whose attribute is being queried.
            Default: 'all', all the existing samplet IDs will be used.

        Returns
        -------
        attr_values : ndarray
            Attribute values for the list of samplet IDs

        Raises
        -------
        KeyError
            If attr_name was never set for dataset, or any of the samplets requested

        """

        if attr_name not in self._attr:
            raise KeyError('Attribute {} is not set for this dataset'
                           ''.format(attr_name))

        if not is_iterable_but_not_str(samplet_ids):
            if samplet_ids.lower() == 'all':
                samplet_ids = self.samplet_ids
            else:
                samplet_ids = [samplet_ids, ]

        samplet_ids = np.array(samplet_ids)

        sid_not_exist = np.array([sid not in self._attr[attr_name]
                                  for sid in samplet_ids])
        if sid_not_exist.any():
            raise KeyError('Attr {} for {} samplets was not set:\n\t{}'
                           ''.format(attr_name, sid_not_exist.sum(),
                                     samplet_ids[sid_not_exist]))

        return np.array([self._attr[attr_name][sid] for sid in samplet_ids],
                        dtype=self._attr_dtype[attr_name])


    def del_attr(self, attr_name, samplet_ids='all'):
        """
        Method to retrieve specified attribute for a list of samplet IDs

        Parameters
        ----------
        attr_name : str
            Name of the attribute

        samplet_ids : str or list
            One or more samplet IDs whose attribute is being queried.
            Default: 'all', all the existing samplet IDs will be used.

        Returns
        -------
        None

        Raises
        -------
        Warning
            If attr_name was never set for this dataset
        """

        if attr_name not in self._attr:
            warn('Attribute {} is not set for this dataset'.format(attr_name),
                 UserWarning)
            return

        if not is_iterable_but_not_str(samplet_ids):
            if samplet_ids.lower() == 'all':
                samplet_ids = self.samplet_ids
            else:
                samplet_ids = [samplet_ids, ]

        for sid in samplet_ids:
            # None helps avoid error if sid does not exist
            self._attr[attr_name].pop(sid, None)


    def attr_summary(self):
        """Simple summary of attributes currently stored in the dataset"""

        print(self._attr_repr())


    @property
    def dataset_attr(self):
        """Returns dataset attributes"""

        return self._dataset_attr


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
    def _keys_with_value(dictionary, value):
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
                               target=self._targets[samplet])

        xfm_ds.description = "{}\n{}".format(func_description, self._description)

        return xfm_ds


    @abstractmethod
    def train_test_split_ids(self, train_perc=None, count=None):
        """
        Returns two disjoint sets of samplet ids for use in cross-validation.
        The behaviour of this method differs based on the whether the child class
        is a ClassificationDataset or RegressionDataset or something else.

        Offers two ways to specify the sizes: fraction or count.
        Only one access method can be used at a time.

        Parameters
        ----------
        train_perc : float
            fraction of samplets to build the training subset.

        count : int
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
            data = self.__get_subset_from_dict(self._data, subset_ids)
            targets = self.__get_subset_from_dict(self._targets, subset_ids)
            sub_ds = self.__class__(data=data, targets=targets)
            # Appending the history
            sub_ds.description += '\n Subset derived from: ' + self.description
            sub_ds.feature_names = self._feature_names
            sub_ds._dtype = self.dtype

            # propagating attributes
            attr_subset = dict()
            for attr in self._attr.keys():
                attr_subset[attr] = self.__get_subset_from_dict(self._attr[attr],
                                                         subset_ids)
            sub_ds.attr = attr_subset

            return sub_ds
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
        """Sample identifiers (strings) forming the basis of Dataset"""
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

    def _attr_repr(self):
        """Text summary of attributes, samplet- and dataset-wise, in the dataset."""

        # newline appended already if this is not empty
        attr_descr = self._dataset_attr_repr()

        if self._attr: # atleast one attribute exists!
            attr_counts = ('{} ({})'.format(attr_name, len(values))
                            for attr_name, values in self._attr.items())
            attr_descr += '{} samplet attributes: {}'.format(len(self._attr),
                                                             ', '.join(attr_counts))

        return attr_descr


    def _dataset_attr_repr(self):
        """Text summary of attributes in the dataset."""

        if self._dataset_attr: # atleast one attribute exists!
            attr_descr = '{} dataset attributes: {}\n' \
                         ''.format(len(self._dataset_attr),
                                   ', '.join(self._dataset_attr.keys()))
        else:
            attr_descr = ''

        return attr_descr


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
                self._data, self._targets, \
                self._dtype, self._target_type, self._description, \
                self._num_features, self._feature_names, \
                self._attr, self._attr_dtype, self._dataset_attr = pickle.load(df)

            # ensure the loaded dataset is valid
            self._validate(self._data, self._targets)

        except IOError as ioe:
            raise IOError('Unable to read the dataset from file: {}', format(ioe))
        except:
            raise


    def save(self, file_path,
             allow_constant_features=False,
             allow_constant_features_across_samplets=False):
        """
        Method to save the dataset to disk.

        Parameters
        ----------
        file_path : str
            File path to save the current dataset to

        allow_constant_features : bool
            Flag indicating whether to allow all the values for features for a
            samplet to be identical (e.g. all zeros). This flag (when False)
            intends to catch unusual, and likely incorrect, situations when all
            features for a  given samplet are all zeros or are all some other
            constant value. In normal, natural, and real-world scenarios,
            different features will have different values. So when they are 0s or
            some other constant value, it is indicative of a bug somewhere. When
            constant values is intended, pass True for this flag.

        allow_constant_features_across_samplets : bool
            While the previous flag allow_constant_features looks at one samplet
            at a time (across features; along rows in feature matrix X: N x p),
            this flag checks for constant values across all samplets for a given
            feature (along the columns). When similar values are expected across
            all samplets, pass True to this flag.

        Raises
        ------
        IOError
            If saving to disk is not successful.

        """

        # TODO need a file format that is flexible and efficient to allow the following:
        #   1) being able to read just meta info without having to load the ENTIRE dataset
        #       i.e. use case: compatibility check with #subjects, ids and their classes
        #   2) random access layout: being able to read features for a single subject!


        # sanity check #1 : per samplet
        if not allow_constant_features:
            if self._num_features > 1:
                self._check_for_constant_features_in_samplets(self._data)

        # sanity check # 2 : per feature across samplets
        if not allow_constant_features_across_samplets:
            if self.num_samplets > 1:
                data_matrix, targets, id_list = self.data_and_targets()
                self._check_for_constant_features_across_samplets(data_matrix)

        try:
            file_path = os.path.abspath(file_path)
            with open(file_path, 'wb') as df:
                pickle.dump((self._data, self._targets,
                             self._dtype, self._target_type, self._description,
                             self._num_features, self._feature_names,
                             self._attr, self._attr_dtype, self._dataset_attr),
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

    @staticmethod
    def _check_for_constant_features_in_samplets(data):
        """Helper to catch constant values in the samplets."""

        for samplet, features in data.items():
            uniq_values = np.unique(features)
            # when there is only one unique value, among n features
            if uniq_values.size < 2:
                raise ConstantValuesException(
                    'Constant values ({}) detected for {} '
                    '- double check the process, '
                    'or disable this check!'.format(uniq_values, samplet))


    def _check_for_constant_features_across_samplets(self, data_matrix):
        """Sanity check to identify identical values for all samplets for a given
        feature
         """

        # notice the transpose, which makes it a column
        for col_ix, col in enumerate(data_matrix.T):
            uniq_values = np.unique(col)
            if uniq_values.size < 2:
                raise ConstantValuesException(
                    'Constant values ({}) detected for feature {} (index {}) - '
                    'double check the process, '
                    'or disable this check (strongly discouraged)!'.format(
                        uniq_values, self._feature_names[col_ix], col_ix))


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




