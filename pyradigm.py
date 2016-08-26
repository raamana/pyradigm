import numpy as np
from collections import Counter, OrderedDict
from itertools import ifilter, takewhile, islice
import random
import warnings
import os
import cPickle as pickle
import copy

# TODO profile the class for different scales of samples and features
class MLDataset(object):
    """Class defining a ML dataset that helps maintain integrity and ease of access."""

    def __init__(self, filepath=None, in_dataset=None, data=None, labels=None, classes=None, description=''):
        """Default constructor. Suggested way to construct the dataset is via add_sample method."""

        if filepath is not None:
            if os.path.isfile(filepath):
                print 'Loading the dataset from: {}'.format(filepath)
                self.__load(filepath)
            else:
                raise IOError('Specified file could not be read.')
        elif in_dataset is not None and isinstance(in_dataset, MLDataset):
            assert in_dataset.num_samples>0, ValueError('Dataset to copy is empty.')
            self.__copy(in_dataset)
        elif data is None and labels is None and classes is None:
            # TODO refactor the code to use only basic dict, as it allows for better equality comparisons
            self.__data = OrderedDict()
            self.__labels = OrderedDict()
            self.__classes = OrderedDict()
            self.__num_features = 0
            self.__description = description
        elif data is not None and labels is not None and classes is not None:
            assert isinstance(data, dict), 'data must be a dict! keys: sample ID or any unique identifier'
            assert isinstance(labels, dict), 'labels must be a dict! keys: sample ID or any unique identifier'
            if classes is not None:
                assert isinstance(classes, dict), 'labels must be a dict! keys: sample ID or any unique identifier'

            assert len(data) == len(labels) == len(classes), 'Lengths of data, labels and classes do not match!'
            assert set(data.keys()) == set(labels.keys()) == set(classes.keys()), 'data, classes and labels ' \
                                                                                  'dictionaries must have the same keys!'
            num_features_in_elements = np.unique([len(sample) for sample in data.values()])
            assert len(num_features_in_elements) == 1, 'different samples have different number of features - invalid!'

            self.__num_features = num_features_in_elements[0]
            # OrderedDict to ensure the order is maintained when data/labels are returned in a matrix/array form
            self.__data = OrderedDict(data)
            self.__labels = OrderedDict(labels)
            self.__classes = OrderedDict(classes)
            self.__dtype = type(data)
            self.__description = description
        else:
            raise ValueError('Incorrect way to construct the dataset.')

    @property
    def data(self):
        """data in its original dict form."""
        return self.__data

    @property
    def data_matrix(self):
        """dataset features in a matrix form."""
        mat = np.zeros([self.num_samples, self.num_features])
        for ix, (sample, features) in enumerate(self.__data.items()):
            mat[ix, :] = features
        return mat

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
        #TODO numeric label need to be removed, as this can be made up on the fly as needed from str to num encoders.
        return self.__labels

    @property
    def target(self):
        """Returns the array of labels for all the samples."""
        return self.__labels.values()

    @labels.setter
    def labels(self, values):
        """Class labels (such as 1, 2, -1, 'A', 'B' etc.) for each sample in the dataset."""
        if isinstance(values, dict):
            if self.__data is not None and len(self.__data) != len(values):
                raise ValueError('number of samples do not match the previously assigned data')
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
    def class_sizes(self):
        """Returns the sizes of different objects in a Counter object."""
        return Counter(self.classes.values())

    @property
    def __label_set(self):
        return set(self.target)

    def __take(self, nitems, iterable):
        "Return first n items of the iterable as a list"
        return dict(islice(iterable, nitems))

    def glance(self, nitems=5):
        """Quick and partial glance of the data matrix."""
        nitems = max([1, min([nitems, self.num_samples]) ])
        return self.__take(nitems, self.__data.iteritems())


    # TODO try implementing based on pandas
    def add_sample(self, sample_id, features, label, class_id=None):
        """Adds a new sample to the dataset with its features, label and class ID.
        This is the preferred way to construct the dataset."""
        if sample_id not in self.__data:
            if self.num_samples <= 0:
                self.__data[sample_id] = features
                self.__labels[sample_id] = label
                self.__classes[sample_id] = class_id
                self.__dtype = type(features)
                self.__num_features = len(features)
            else:
                assert self.__num_features == len(features), \
                    ValueError('dimensionality of this sample ({}) does not match existing samples ({})'.format(
                    len(features),self.__num_features))
                assert isinstance(features,self.__dtype), TypeError("Mismatched dtype. Provide {}".format(self.__dtype))

                self.__data[sample_id] = features
                self.__labels[sample_id] = label
                self.__classes[sample_id] = class_id
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
        assert ( max(subset_idx) < self.__num_features) and (min(subset_idx)>=0), \
            UnboundLocalError('indices out of range for the dataset. Max index: {}'.format(self.__num_features))

        sub_data = {sample : features[subset_idx] for sample, features in self.__data.items()}
        new_descr = 'Subset features derived from: \n ' + self.__description
        subdataset = MLDataset(data=sub_data, labels=self.__labels, classes=self.__classes, description=new_descr)

        return subdataset


    def get_class(self, class_id):
        """Returns a smaller dataset belonging to the requested classes. """
        assert class_id not in [None, ''], "class id can not be empty or None."
        if isinstance(class_id,basestring):
            class_ids = [class_id, ]
        else:
            class_ids = class_id

        subsets = list()
        for class_id in class_ids:
            if class_id in self.class_set:
                subsets_this_class = [sample for sample in self.__classes if self.__classes[sample] in class_id]
                subsets.extend(subsets_this_class)
            else:
                warnings.warn('Requested class: {} does not exist in this dataset.'.format(class_id))

        if len(subsets) < 1:
            warnings.warn("Given class[es] do not belong the dataset")
            return None
        else:
            return self.get_subset(subsets)

    # TODO sampling of cross-validation splits?
    def random_subset(self, perc_per_class = 0.5, random_seed = None):
        """Returns a random subset of samples (of specified size by percentage) within each class."""

        class_sizes = self.class_sizes
        subsets = list()

        if perc_per_class <= 0:
            warnings.warn('Zero percentage requested - returning an empty dataset!')
            return MLDataset()
        elif perc_per_class >=1:
            warnings.warn('Full or a larger dataset requested - returning a copy!')
            return MLDataset(in_dataset=self)

        # seeding the random number generator
        # TODO make sure the default behaviour is to return different subset even without supplying the seed.
        if random_seed is not None:
            random.seed(random_seed)

        for class_id, class_size in class_sizes.items():
            # samples belonging to the class
            this_class = [sample for sample in self.classes if self.classes[sample] in class_id]
            # shuffling the sample order; shuffling works in-place!
            random.shuffle(this_class)
            # calculating the requested number of samples
            subset_size_this_class = int(round(class_size * perc_per_class))
            # clipping the range to [0, n]
            subset_size_this_class = max(0, min(class_size, subset_size_this_class))
            if subset_size_this_class < 1 or this_class is None:
                # warning if none were selected
                warnings.warn('No subjects from class {} were selected.'.format(class_id))
            else:
                subsets_this_class = this_class[0:subset_size_this_class]
                subsets.extend(subsets_this_class)

        if len(subsets) > 0:
            return self.get_subset(subsets)
        else:
            warnings.warn('Zero samples were selected. Returning an empty dataset!')
            return MLDataset()

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
        return self.__num_features

    @num_features.setter
    def num_features(self, int_val):
        assert isinstance(int_val, int) and (0 < int_val < np.Inf), UnboundLocalError('Invalid number of features.')
        self.__num_features = int_val

    @property
    def dtype(self):
        """number of features in each sample."""
        return self.__dtype

    @dtype.setter
    def dtype(self, type_val):
        #TODO assert isinstance(type_val,type), TypeError('Invalid data type.')
        self.__dtype = type_val

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
        return len(self.__label_set)

    @property
    def class_set(self):
        """Set of unique classes in the dataset."""
        return set(self.__classes.values())

    def add_classes(self, classes):
        """Helper to rename the classes, if provided by a dict keyed in by the orignal keys"""
        assert isinstance(classes,dict), TypeError('Input classes is not a dict!')
        assert len(classes) == self.num_samples, ValueError('Too few items - need {} keys'.format(self.num_samples))
        assert all([ key in self.keys for key in classes ]), ValueError('One or more unrecognized keys!')
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
            full_descr.append('{} samples and {} features.'.format(self.num_samples, self.num_features))
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

    def __dir__(self):
        """Returns the preferred list of attributes to be used with the dataset."""
        return ['add_sample',
                'glance',
                'class_set',
                'class_sizes',
                'classes',
                'data',
                'data_matrix',
                'del_sample',
                'description',
                'extend',
                'get_class',
                'get_subset',
                'random_subset',
                'get_feature_subset',
                'keys',
                'target',
                'num_classes',
                'num_features',
                'num_samples',
                'sample_ids',
                'save',
                'add_classes' ]

    def __copy(self, other):
        """Copy constructor."""
        self.__data         = copy.deepcopy(other.data)
        self.__classes      = copy.deepcopy(other.classes)
        self.__labels       = copy.deepcopy(other.labels)
        self.__dtype        = copy.deepcopy(other.dtype)
        self.__description  = copy.deepcopy(other.description)
        self.__num_features = copy.deepcopy(other.num_features)

        return self

    def __load(self, path):
        """Method to load the serialized dataset from disk."""
        try:
            path = os.path.abspath(path)
            with open(path, 'rb') as df:
                # loaded_dataset = pickle.load(df)
                self.__data, self.__classes, self.__labels, \
                    self.__dtype, self.__description, self.__num_features = pickle.load(df)
        except IOError as ioe:
            raise IOError('Unable to read the dataset from file: {}',format(ioe))
        except:
            raise

    def save(self, path):
        """Method to serialize the dataset to disk."""
        try:
            path = os.path.abspath(path)
            with open(path, 'wb') as df:
                # pickle.dump(self, df)
                pickle.dump((self.__data, self.__classes, self.__labels,
                             self.__dtype, self.__description, self.__num_features),
                            df)
            return
        except IOError as ioe:
            raise IOError('Unable to read the dataset from file: {}',format(ioe))
        except:
            raise

    def extend(self, other):
        """Method to extend the dataset vertically (add samples from  anotehr dataset)."""
        assert isinstance(other, MLDataset), TypeError('Incorrect type of dataset provided!')
        # assert self.__dtype==other.dtype, TypeError('Incorrect data type of features!')
        for sample in other.keys:
            self.add_sample(sample, other.data[sample], other.labels[sample], other.classes[sample])

    def __add__(self, other):
        assert isinstance(other,MLDataset), TypeError('Incorrect type of dataset provided!')

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
            warnings.warn('None of the keys in the dataset to be removed found in this dataset - nothing to do.')
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
