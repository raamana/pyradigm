import numpy as np
from collections import Counter, OrderedDict
import warnings


class MLDataset(object):
    """Class defining a ML dataset that helps maintain integrity and ease of access."""

    def __init__(self, data=None, labels=None, classes=None):
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
            # TODO need a better way to ensure keys are identical
            assert data.keys() == labels.keys(), 'data and labels dictionaries must have the same keys!'

            num_features_in_elements = np.unique([len(sample) for sample in data.values()])
            assert len(num_features_in_elements) == 1, 'different samples have different number of features - invalid!'

            self.__num_features = num_features_in_elements[0]
            # OrderedDict to ensure the order is maintained when data/labels are returned in a matrix/array form
            self.__data = OrderedDict(data)
            self.__labels = OrderedDict(labels)
            self.__classes = classes

        self.__label_set = set(self.labels)
        self.class_sizes = Counter(self.labels)
        self.__description = ''

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

    def add_sample(self, subject_id, features, label, class_id=None):
        """Adds a new sample to the dataset with its features, label and class ID."""
        if subject_id not in self.__data:
            self.__data[subject_id] = features
            self.__labels[subject_id] = label
            self.__classes[subject_id] = class_id
        else:
            raise ValueError('{} already exists in this dataset!'.format(subject_id))

    def get_class(self, class_id):
        if class_id in self.class_set:
            subset_in_class = [sub_id for sub_id in self.__classes if self.__classes[sub_id] == class_id]
            return self.get_subset(subset_in_class)
        else:
            raise ValueError('Requested class: {} does not exist in this dataset.'.format(class_id))

    def get_subset(self, subset_ids):

        num_existing_keys = sum([1 for key in subset_ids if key in self.__data])
        if subset_ids is not None and num_existing_keys > 0:
            data = self.__get_subset_from_dict(self.__data, subset_ids)
            labels = self.__get_subset_from_dict(self.__labels, subset_ids)
            if self.__classes is not None:
                classes = self.__get_subset_from_dict(self.__classes, subset_ids)
            else:
                classes = None
            subdataset = MLDataset(data, labels, classes)
            # Appending the history
            subdataset.description += 'Subset derived from ' + self.description
        else:
            warnings.warn('subset of IDs requested do not exist in the dataset!')
            return MLDataset()

    def __get_subset_from_dict(self, dict, subset):
        return OrderedDict((sid, dict[sid]) for sid in dict if sid in subset)

    @property
    def keys(self):
        """Identifiers (subject IDs, or sample names etc) forming the basis of dict-type MLDataset."""
        return self.__data.keys()

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
        return len(self.__label_set)

    @property
    def class_set(self):
        return set(self.__classes)

    def add_classes(self, classes):
        if len(classes) == self.num_classes:
            self.__label_set = classes

    def __len__(self):
        return self.num_samples

    def __nonzero__(self):
        if self.num_samples < 1:
            return False
        else:
            return True

    def __str__(self):
        full_descr = list()
        full_descr.append(self.description)
        full_descr.append('{} samples and {} features.'.format(self.num_samples, self.num_features))
        for cx in self.class_sizes:
            full_descr.append('Class {:3d} : {:5d} samples.'.format(cx, self.class_sizes.get(cx)))
        return '\n'.join(full_descr)

    def __format__(self, fmt_str):
        if isinstance(fmt_str, basestring):
            return '{:d} samples x {:d} features with {:d} classes'.format(self.num_samples, self.num_features,
                                                                           self.num_classes)

    def __repr__(self):
        return self.__str__()
