import random
from warnings import warn
from collections import Iterable, Counter
from copy import copy
from operator import itemgetter
from sys import version_info
from abc import abstractmethod
import numpy as np

if version_info.major > 2:
    from pyradigm.base import BaseDataset, CompatibilityException
    from pyradigm import MLDataset, ClassificationDataset as ClfDataset, \
        RegressionDataset as RegrDataset
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
                 dataset_class=BaseDataset,
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

        self.name = name
        self._list = list()
        self._is_init = False

        # number of modalities for each sample id
        self.modality_count = 0

        self._ids = set()
        self.targets = dict()
        self._modalities = dict()
        self._labels = dict()

        self.num_features = list()

        # TODO more efficient internal repr is possible as ids/classes do not need be
        # stored redundantly for each dataset
        # perhaps as different attributes/modalities/feat-sets (of .data) for example?

        if dataset_spec is not None:
            if not isinstance(dataset_spec, Iterable) or len(dataset_spec) < 1:
                raise ValueError('Input must be a list of atleast two datasets.')

            self._load(dataset_spec)


    def _load(self, dataset_spec):
        """Actual loading of datasets"""

        for idx, ds in enumerate(dataset_spec):
            self.append(ds, idx)


    def _get_id(self):
        """Returns an ID for a new dataset that's different from existing ones."""

        self.modality_count += 1

        return self.modality_count


    def append(self, dataset, identifier):
        """
        Adds a dataset, if compatible with the existing ones.

        Parameters
        ----------
        dataset : pyradigm dataset or compatible

        identifier : hashable
            String or integer or another hashable to uniquely identify this dataset
        """

        if isinstance(dataset, str):
            dataset = self._dataset_class(dataset_path=dataset)

        if not isinstance(dataset, self._dataset_class):
            raise CompatibilityException('Incompatible dataset. '
                                         'You can only add instances of '
                                         'type {}'.format(self._dataset_class))

        if len(dataset.description)>0:
            identifier = dataset.description

        if not self._is_init:
            self._ids = set(dataset.samplet_ids)
            self.targets = dataset.targets
            self._target_sizes = dataset.target_sizes

            self.num_samplets = len(self._ids)
            self._modalities[identifier] = dataset.data
            self.num_features.append(dataset.num_features)

            # maintaining a no-data pyradigm Dataset internally to reuse its methods
            self._dataset = copy(dataset)
            # replacing its data with zeros
            self._dataset.data = {id_: np.zeros(1) for id_ in self._ids}

            if hasattr(dataset, 'attr'):
                self._common_attr = dataset.attr
                self._common_attr_dtype = dataset.attr_dtype
            else:
                self._common_attr = dict()
                self._common_attr_dtype = dict()

            self._attr = dict()

            self._is_init = True
        else:
            # this also checks for the size (num_samplets)
            if set(dataset.samplet_ids) != self._ids:
                raise CompatibilityException(
                        'Differing set of IDs in two datasets.'
                        ' Unable to add this dataset to the MultiDataset.')

            if dataset.targets != self.targets:
                raise CompatibilityException(
                        'Targets for some IDs differ in the two datasets.'
                        ' Unable to add this dataset to the MultiDataset.')

            if identifier not in self._modalities:
                self._modalities[identifier] = dataset.data
                self.num_features.append(dataset.num_features)
            else:
                raise KeyError('{} already exists in MultiDataset'
                               ''.format(identifier))

            if hasattr(dataset, 'attr'):
                if len(self._common_attr) < 1:
                    # no attributes were set at all - simple copy sufficient
                    self._common_attr = dataset.attr.copy()
                    self._common_attr_dtype = dataset.attr_dtype.copy()
                else:
                    for a_name in dataset.attr:
                        if a_name not in self._common_attr:
                            self._common_attr[a_name] = dataset.attr[a_name]
                            self._common_attr_dtype[a_name] = \
                                dataset.attr_dtype[a_name]
                        elif self._common_attr[a_name] != dataset.attr[a_name]:
                            raise ValueError(
                                    'Values and/or IDs differ for attribute {}. '
                                    'Ensure all datasets have common attributes '
                                    'with the same values'.format(a_name))


        # each addition should be counted, if successful
        self.modality_count += 1


    @property
    def samplet_ids(self):
        """List of samplet IDs in the multi-dataset"""
        return list(self._ids)

    @property
    def modality_ids(self):
        """List of identifiers for all modalities"""
        return list(self._modalities.keys())

    @abstractmethod
    def __str__(self):
        """human readable repr"""

    def _common_str(self):
        """basic str() with common elements"""

        return "{}:\n\t{} samples, {} modalities, dims: {}" \
               "\n\tIdentifiers: {}" \
               "\n\tAttributes: {}" \
               "".format(self.name, self.num_samplets, self.modality_count,
                         self.num_features,
                         ', '.join([str(k) for k in self.modality_ids]),
                         ', '.join([str(k) for k in self._common_attr.keys()]))

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


    def __iter__(self):
        """Iterable mechanism"""

        for modality, data in self._modalities.items():
                yield modality, np.array([np.array(item) for item in data.values()])


    def get_subsets(self, subset_list):
        """
        Returns the requested subsets of data while iterating over modalities

        if subset_list were to contain two sets of IDs e.g. (train, test)

        return data would be this tuple:
            (modality, (train_data, train_targets), (test_data, test_targets))

        """

        for modality, data in self._modalities.items():
            yield modality, ( (np.array(itemgetter(*subset)(data)),
                               np.array(itemgetter(*subset)(self.targets)))
                              for subset in subset_list )


    @property
    def common_attr(self):
        """Attributes common to all subjects/datasets, such as covariates, in this
        MultiDataset"""

        return self._common_attr


    def set_attr(self, ds_id, attr_name, attr_value):
        """Method to set modality-/dataset-specific attributes"""

        if ds_id not in self._modalities:
            raise KeyError('Dataset {} not in this {} multi_dataset'
                           ''.format(ds_id, self._name))

        if ds_id not in self._attr:
            self._attr[ds_id] = dict()

        self._attr[ds_id][attr_name] = attr_value


    def get_attr(self, ds_id, attr_name, not_found_value='raise'):
        """Method to retrieve modality-/dataset-specific attributes"""

        if ds_id not in self._modalities:
            raise KeyError('Dataset {} not in this {} multi_dataset'
                           ''.format(ds_id, self._name))

        try:
            return self._attr[ds_id][attr_name]
        except KeyError:
            msg = 'attribute {} not set for dataset {}'.format(attr_name, ds_id)
            if not_found_value.lower() in ('raise', ):
                raise KeyError(msg)
            else:
                warn(msg)
                return not_found_value


class MultiDatasetClassify(BaseMultiDataset):
    """Container class to manage multimodal classification datasets."""


    def __init__(self,
                 dataset_spec=None,
                 name='MultiDatasetClassify',
                 subgroup=None):
        """
        Constructor.

        Parameters
        ----------
        dataset_spec : Iterable or None
            List of pyradigms, or absolute paths to serialized pyradigm Datasets.

        name : str
            human readable name for printing purposes

        """

        self._sub_groups = subgroup
        if subgroup is None:
            super().__init__(dataset_class=ClfDataset,
                             dataset_spec=dataset_spec,
                             name=name)
        else:
            super().__init__(dataset_class=ClfDataset, dataset_spec=None, name=name)
            for idx, ds in enumerate(dataset_spec):
                self.append_subgroup(ds, idx, subgroup)


    def append_subgroup(self, dataset, identifier, subgroup):
        """Custom add method"""

        if isinstance(dataset, str):
            dataset = self._dataset_class(dataset_path=dataset)

        target_set = set(dataset.target_set)
        subgroup = set(subgroup)
        if subgroup is None or subgroup == target_set:
            ds_out = dataset
        elif subgroup < target_set: # < on sets is an issubset operation
            ds_out = dataset.get_class(subgroup)
        else:
            raise ValueError('One or more classes in {} does not exist in\n{}'
                             ''.format(sub_group, fp))

        self.append(ds_out, identifier=identifier)


    @property
    def target_set(self):
        """Set of targets/classes in this multi-dataset"""

        return set(self.targets.values())

    @property
    def target_sizes(self):
        """
        Sizes of targets in this classification dataset.
        Useful for summary and to compute chance accuracy.
        """
        return Counter(self.targets.values())

    def __str__(self):
        """human readable repr"""

        string = "{}\n\tClasses n={}, sizes " \
                 "".format(self._common_str(), len(self._target_sizes))
        string += ', '.join(['{}: {}'.format(c, n)
                             for c, n in self._target_sizes.items()])

        return string


    def __repr__(self):

        return self.__str__()


    def __format__(self, format_spec):

        return self.__str__()


    def holdout(self,
                train_perc=0.7,
                num_rep=50,
                stratified=True,
                return_ids_only=False,
                format='MLDataset'):
        """
        Builds a holdout generator for train and test sets for cross-validation.
        Ensures all the classes are represented equally in the training set.

        """

        if train_perc <= 0.0 or train_perc >= 1.0:
            raise ValueError('Train perc > 0.0 and < 1.0')

        ids_in_class = {cid: self._dataset.sample_ids_in_class(cid)
                        for cid in self._target_sizes.keys()}

        sizes_numeric = np.array([len(ids_in_class[cid])
                                  for cid in ids_in_class.keys()])
        size_per_class, total_test_count = compute_training_sizes(
                train_perc, sizes_numeric, stratified=stratified)

        if len(self._target_sizes) != len(size_per_class):
            raise ValueError('size spec differs in num elements with class sizes!')

        for rep in range(num_rep):
            print('rep {}'.format(rep))

            train_set = list()
            for index, (cls_id, class_size) in enumerate(self._target_sizes.items()):
                # shuffling the IDs each time
                random.shuffle(ids_in_class[cls_id])

                subset_size = max(0, min(class_size, size_per_class[index]))
                if subset_size < 1 or class_size < 1:
                    warn('No subjects from class {} were selected.'
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
                yield self._get_data(train_set, format), \
                      self._get_data(test_set, format)


class MultiDatasetRegress(BaseMultiDataset):
    """Container class to manage multimodal classification datasets."""


    def __init__(self,
                 dataset_spec=None,
                 name='MultiDatasetRegress'):
        """
        Constructor.

        Parameters
        ----------
        dataset_spec : Iterable or None
            List of pyradigms, or absolute paths to serialized pyradigm Datasets.

        name : str
            human readable name for printing purposes

        """

        super().__init__(dataset_class=RegrDataset,
                         dataset_spec=dataset_spec,
                         name=name)


    def __str__(self):
        """human readable repr"""

        return self._common_str()


    def __repr__(self):

        return self.__str__()


    def __format__(self, format_spec):

        return self.__str__()


    def holdout(self,
                train_perc=0.7,
                num_rep=50,
                return_ids_only=False,
                format='MLDataset'):
        """
        Builds a holdout generator for train and test sets for cross-validation.

        """

        if train_perc <= 0.0 or train_perc >= 1.0:
            raise ValueError('Train perc > 0.0 and < 1.0')

        subset_size = np.int64(np.floor(self.num_samplets * train_perc))

        # clipping the range to [1, n]
        subset_size = max(1, min(self.num_samplets, subset_size))

        # making it indexible with a local copy
        id_list = list(self._ids)

        for rep in range(num_rep):

            random.shuffle(id_list)
            train_set = id_list[:subset_size]
            # this ensures both are mutually exclusive!
            test_set = list(self._ids - set(train_set))

            if return_ids_only:
                # when only IDs are required, without associated features
                # returning tuples to prevent accidental changes
                yield tuple(train_set), tuple(test_set)
            else:
                yield self._get_data(train_set, format), \
                      self._get_data(test_set, format)


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
