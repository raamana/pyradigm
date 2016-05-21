import numpy as np
from collections import Counter, OrderedDict
from itertools import ifilter, takewhile, islice
import warnings
import os
import cPickle as pickle

class MLDataset(object):
    """Class defining a ML dataset that helps maintain integrity and ease of access."""

    def __init__(self, data=None, labels=None, classes=None, description=''):
        """Default constructor. Suggested way to construct the dataset is via add_sample method."""

        if data is None or labels is None:
            self.__data = OrderedDict()
            self.__labels = OrderedDict()
            self.__classes = OrderedDict()
            self.__num_features = 0
        else:
            assert isinstance(data, dict), 'data must be a dict! keys: subject ID or any unique identifier'
            assert isinstance(labels, dict), 'labels must be a dict! keys: subject ID or any unique identifier'
            if classes is not None:
                assert isinstance(classes, dict), 'labels must be a dict! keys: subject ID or any unique identifier'

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

    @property
    def data(self):
        """data in its original dict form."""
        return self.__data

    @property
    def data_matrix(self):
        """dataset features in a matrix form."""
        mat = np.zeros([self.num_samples, self.num_features])
        for ix, (sub, features) in enumerate(self.__data.items()):
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
        return self.__labels.values()

    @property
    def target(self):
        """Returns the array of labels for all the samples."""
        return self.labels

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
    def class_sizes(self):
        """Returns the sizes of different objects in a Counter object."""
        return Counter(self.classes)

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


    def add_sample(self, subject_id, features, label, class_id=None):
        """Adds a new sample to the dataset with its features, label and class ID.
        This is the preferred way to construct the dataset."""
        if subject_id not in self.__data:
            if self.num_samples <= 0:
                self.__data[subject_id] = features
                self.__labels[subject_id] = label
                self.__classes[subject_id] = class_id
                self.__dtype = type(features)
                self.__num_features = len(features)
            else:
                assert self.__num_features == len(features), \
                    ValueError('dimensionality of this sample ({}) does not match existing samples ({})'.format(
                    len(features),self.__num_features))
                assert isinstance(features,self.__dtype), TypeError("Mismatched dtype. Provide {}".format(self.__dtype))

                self.__data[subject_id] = features
                self.__labels[subject_id] = label
                self.__classes[subject_id] = class_id
        else:
            raise ValueError('{} already exists in this dataset!'.format(subject_id))

    def get_feature_subset(self, subset_idx):
        """Returns the subset of features indexed numerically. """

        subset_idx = np.asarray(subset_idx)
        assert ( max(subset_idx) < self.__num_features) and (min(subset_idx)>=0), \
            UnboundLocalError('indices out of range for the dataset. Max index: {}'.format(self.__num_features))

        sub_data = {sub : features[subset_idx] for sub, features in self.__data.items()}
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
                subsets_this_class = [sub_id for sub_id in self.__classes if self.__classes[sub_id] == class_id]
                subsets.extend(subsets_this_class)
            else:
                warnings.warn('Requested class: {} does not exist in this dataset.'.format(class_id))

        if len(subsets) < 1:
            warnings.warn("All the classes do not belong the dataset")
            return None
        else:
            return self.get_subset(subsets)

    def get_subset(self, subset_ids):
        """Returns a smaller dataset identified by their keys/subject IDs."""
        num_existing_keys = sum([1 for key in subset_ids if key in self.__data])
        if subset_ids is not None and num_existing_keys > 0:
            # need to ensure data are added to data, labels etc in the same order of subject IDs
            data = self.__get_subset_from_dict(self.__data, subset_ids)
            labels = self.__get_subset_from_dict(self.__labels, subset_ids)
            if self.__classes is not None:
                classes = self.__get_subset_from_dict(self.__classes, subset_ids)
            else:
                classes = None
            subdataset = MLDataset(data, labels, classes)
            # Appending the history
            subdataset.description += '\n Subset derived from: ' + self.description
            return subdataset
        else:
            warnings.warn('subset of IDs requested do not exist in the dataset!')
            return MLDataset()

    def __get_subset_from_dict(self, input_dict, subset):
        # Using OrderedDict helps ensure data are added to data, labels etc in the same order of subject IDs
        return OrderedDict((sid, value) for sid, value in input_dict.items() if sid in subset)

        # # statement below doesn't work for some reason
        # return OrderedDict(ifilter( lambda key: key in subset, input_dict))

    @property
    def keys(self):
        """Identifiers (subject IDs, or sample names etc) forming the basis of dict-type MLDataset."""
        return self.__data.keys()

    @property
    def subject_ids(self):
        return self.keys

    @property
    def classes(self):
        """Identifiers (subject IDs, or sample names etc) forming the basis of dict-type MLDataset."""
        return self.__classes.values()

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

        return '\n'.join(full_descr)

    def __format__(self, fmt_str):
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
                'description',
                'get_class',
                'get_subset',
                'get_feature_subset',
                'keys',
                'labels',
                'num_classes',
                'num_features',
                'num_samples',
                'subject_ids',
                'add_classes' ]

    def load(self, path):
        raise NotImplementedError
        # try:
        #     path = os.path.abspath(path)
        #     with open(path, 'rb') as df:
        #         dataset = pickle.load(df)
        #         self.__dict__.update(dataset)
        #         return self
        # except IOError as ioe:
        #     raise IOError('Unable to read the dataset from file: {}',format(ioe))
        # finally:
        #     raise

    def save(self, path):
        raise NotImplementedError
        # try:
        #     path = os.path.abspath(path)
        #     with open(path, 'wb') as df:
        #         save_state = dict(self.__dict__)
        #         pickle.dump(save_state, df)
        #         # pickle.dump((self.__data, self.__classes, self.__labels, self.__dtype, self.__description), df)
        #         return
        # except IOError as ioe:
        #     raise IOError('Unable to read the dataset from file: {}',format(ioe))
        # finally:
        #     raise

