import numpy as np
from collections import Counter, OrderedDict, Sequence
from itertools import ifilter, takewhile, islice
import random
import warnings
import os
import cPickle as pickle
import copy

# TODO profile the class for different scales of samples and features
class MLDataset(object):
    """Class defining a ML dataset that helps maintain integrity and ease of access."""

    def __init__(self, filepath=None, in_dataset=None,
                 data=None, labels=None, classes=None,
                 description='', feature_names = None):
        """Default constructor. Suggested way to construct the dataset is via add_sample method."""

        if filepath is not None:
            if os.path.isfile(filepath):
                # print 'Loading the dataset from: {}'.format(filepath)
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

            sample_ids = data.keys()
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
        """Populates this dataset with the provided data."""
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
            elif set(self.keys) != set(values.keys()):
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
            elif set(self.keys) != set(values.keys()):
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
        return dict(islice(iterable, nitems))

    @staticmethod
    def __str_names(num):

        return np.array(['f{}'.format(x) for x in range(num)])

    def glance(self, nitems=5):
        """Quick and partial glance of the data matrix."""
        nitems = max([1, min([nitems, self.num_samples])])
        return self.__take(nitems, self.__data.iteritems())

    def summarize_classes(self):
        "Summary of classes: names, numeric labels and sizes"

        class_sizes = np.zeros(len(self.class_set))
        for idx, cls in enumerate(self.class_set):
            class_sizes[idx] = self.class_sizes[cls]

        return self.class_set, self.label_set, class_sizes

    # TODO try implementing based on pandas
    def add_sample(self, sample_id, features, label, class_id=None, feature_names=None):
        """Adds a new sample to the dataset with its features, label and class ID.
        This is the preferred way to construct the dataset."""
        if sample_id not in self.__data:
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
                    else: # if set already, ensure a match
                        print ''
                        assert np.array_equal(self.feature_names, np.array(feature_names)), \
                            "supplied feature names do not match the existing names!"
        else:
            raise ValueError('{} already exists in this dataset!'.format(sample_id))

    def del_sample(self, sample_id):
        """Method remove a sample from the dataset."""
        if sample_id not in self.__data:
            warnings.warn('Sample to delete not found in the dataset - nothing to do.')
        else:
            self.__data.pop(sample_id)
            self.__classes.pop(sample_id)
            self.__labels.pop(sample_id)
            print '{} removed.'.format(sample_id)

    def get_feature_subset(self, subset_idx):
        """Returns the subset of features indexed numerically. """

        subset_idx = np.asarray(subset_idx)
        assert (max(subset_idx) < self.__num_features) and (min(subset_idx) >= 0), \
            UnboundLocalError('indices out of range for the dataset. Max index: {}'.format(self.__num_features))

        sub_data = {sample: features[subset_idx] for sample, features in self.__data.items()}
        new_descr = 'Subset features derived from: \n ' + self.__description
        subdataset = MLDataset(data=sub_data,
                               labels=self.__labels, classes=self.__classes,
                               description=new_descr,
                               feature_names= self.__feature_names[subset_idx])

        return subdataset

    @staticmethod
    def keys_with_value(dictionary, value):
        "Returns a subset of keys from the dict with the value supplied."

        subset = [key for key in dictionary if dictionary[key] == value]

        return subset

    def get_class(self, class_id):
        """Returns a smaller dataset belonging to the requested classes. """
        assert class_id not in [None, ''], "class id can not be empty or None."
        if isinstance(class_id, basestring):
            class_ids = [class_id, ]
        else:
            class_ids = class_id

        non_existent = set(self.class_set).intersection(set(class_ids))
        if len(non_existent)<1:
            raise ValueError('These classes {} do not exist in this dataset.'.format(non_existent))

        subsets = list()
        for class_id in class_ids:
            # subsets_this_class = [sample for sample in self.__classes if self.__classes[sample] == class_id]
            subsets_this_class = self.keys_with_value(self.__classes, class_id)
            subsets.extend(subsets_this_class)

        return self.get_subset(subsets)

    def train_test_split_ids(self, train_perc = None, count_per_class = None):
        "Returns two disjoint sets of sample ids for use in cross-validation."

        _, _, class_sizes = self.summarize_classes()
        smallest_class_size = np.min(class_sizes)

        if count_per_class is None and (train_perc>0.0 and train_perc<1.0):
            if train_perc < 1.0 / smallest_class_size:
                raise ValueError('Training percentage selected too low '
                                 'to return even one sample from the smallest class!')
            train_set = self.random_subset_ids(perc_per_class=train_perc)
        elif train_perc is None and count_per_class>0:
            if count_per_class >= smallest_class_size:
                raise ValueError('Selections would exclude the smallest class from test set. '
                                 'Reduce sample count per class for the training set!')
            train_set = self.random_subset_ids_by_count(count_per_class=count_per_class)
        else:
            raise ValueError('Invalid or out of range selection: '
                             'only one of count or percentage can be used to select subset.')

        test_set  = list(set(self.keys) - set(train_set))

        if len(train_set) < 1 or len(test_set) < 1:
            raise ValueError('Selection resulted in empty training or test set - check your selections or dataset!')

        return train_set, test_set

    def random_subset_ids_by_count(self, count_per_class=1):
        """Returns a random subset of sample ids (of specified size by percentage) within each class."""

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
        """Returns a random subset of sample ids (of specified size by percentage) within each class."""

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
        """Returns a random subset of dataset (of specified size by percentage) within each class."""

        subsets = self.random_subset_ids(perc_in_class)
        if len(subsets) > 0:
            return self.get_subset(subsets)
        else:
            warnings.warn('Zero samples were selected. Returning an empty dataset!')
            return MLDataset()

    def sample_ids_in_class(self, class_id):
        "Returns a list of sample ids belonging to a given class."

        # subset_ids = [sid for sid in self.keys if self.classes[sid] == class_id]
        subset_ids = self.keys_with_value(self.classes, class_id)
        return subset_ids

    def get_subset(self, subset_ids):
        """Returns a smaller dataset identified by their keys/sample IDs."""
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
        """Identifiers (sample IDs, or sample names etc) forming the basis of dict-type MLDataset."""
        return self.__data.keys()

    @property
    def sample_ids(self):
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
        "Method that should nor exist!"
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
            assert isinstance(type_val,type), TypeError('Invalid data type.')
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
        """Total numver of classes in the dataset."""
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
        """Helper to rename the classes, if provided by a dict keyed in by the orignal keys"""
        assert isinstance(classes, dict), TypeError('Input classes is not a dict!')
        assert len(classes) == self.num_samples, ValueError('Too few items - need {} keys'.format(self.num_samples))
        assert all([key in self.keys for key in classes]), ValueError('One or more unrecognized keys!')
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
            class_ids = self.class_sizes.keys()
            max_width = max([len(cls) for cls in class_ids])
            for cls in class_ids:
                full_descr.append('Class {:>{}} : {} samples.'.format(cls, max_width, self.class_sizes.get(cls)))
        else:
            full_descr.append('Empty dataset.')

        return '\n'.join(full_descr)

    def __format__(self, fmt_str='s'):
        if isinstance(fmt_str, basestring):
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

    def __load(self, path, **kwargs):
        """Method to load the serialized dataset from disk."""
        try:
            path = os.path.abspath(path)
            with open(path, 'rb') as df:
                # loaded_dataset = pickle.load(df)
                self.__data, self.__classes, self.__labels, \
                self.__dtype, self.__description, \
                self.__num_features, self.__feature_names = pickle.load(df, **kwargs)

            # ensure the loaded dataset is valid
            self.__validate(self.__data, self.__classes, self.__labels)

        except IOError as ioe:
            raise IOError('Unable to read the dataset from file: {}', format(ioe))
        except:
            raise

    def save(self, path):
        """Method to serialize the dataset to disk."""
        try:
            path = os.path.abspath(path)
            with open(path, 'wb') as df:
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

        assert isinstance(data, dict), 'data must be a dict! keys: sample ID or any unique identifier'
        assert isinstance(labels, dict), 'labels must be a dict! keys: sample ID or any unique identifier'
        if classes is not None:
            assert isinstance(classes, dict), 'labels must be a dict! keys: sample ID or any unique identifier'

        assert len(data) == len(labels) == len(classes), 'Lengths of data, labels and classes do not match!'
        assert set(data.keys()) == set(labels.keys()) == set(classes.keys()), 'data, classes and labels ' \
                                                                              'dictionaries must have the same keys!'
        num_features_in_elements = np.unique([len(sample) for sample in data.values()])
        assert len(num_features_in_elements) == 1, 'different samples have different number of features - invalid!'

        return True

    def extend(self, other):
        """Method to extend the dataset vertically (add samples from  anotehr dataset)."""
        assert isinstance(other, MLDataset), TypeError('Incorrect type of dataset provided!')
        # assert self.__dtype==other.dtype, TypeError('Incorrect data type of features!')
        for sample in other.keys:
            self.add_sample(sample, other.data[sample], other.labels[sample], other.classes[sample])

    def __add__(self, other):
        "Method to combine to MLDatasets, sample-wise or feature-wise."

        assert isinstance(other, MLDataset), TypeError('Incorrect type of dataset provided!')

        if set(self.keys) == set(other.keys):
            print 'Identical keys found. Trying to horizontally concatenate features for each sample.'
            assert self.__classes == other.classes, ValueError('Class identifiers per sample differ in the two '
                                                               'datasets!')
            assert other.num_features > 0, ValueError('No features to concatenate.')
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
        assert isinstance(other, type(self)), TypeError('Incorrect type of dataset provided!')

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
            print 'differing sample ids.'
            return False
        elif dict(self.__classes) != dict(other.classes):
            print 'differing classes for the sample ids.'
            return False
        elif id(self.__data) != id(other.data):
            for key in self.keys:
                if not np.all(self.data[key] == other.data[key]):
                    print 'differing data for the sample ids.'
                    return False
            return True
        else:
            return True
