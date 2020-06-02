from collections.abc import Iterable

import numpy as np

from pyradigm import ClassificationDataset, RegressionDataset
from pyradigm.base import missing_value_indicator
from pyradigm.pyradigm import MLDataset
from pyradigm.base import is_iterable_but_not_str, BaseDataset
from warnings import warn

feat_generator = np.random.randn


def load_dataset(ds_path):
    """Convenience utility to quickly load any type of pyradigm dataset"""

    try:
        ds = ClassificationDataset(dataset_path=ds_path)
    except:
        try:
            ds = RegressionDataset(dataset_path=ds_path)
        except:
            try:
                warn('MLDtaset is deprecated. Switch to the latest pyradigm data '
                     'structures such as ClassificationDataset or '
                     'RegressionDataset as soon as possible.')
                ds = MLDataset(filepath=ds_path)
            except:
                raise TypeError('Dataset class @ path below not recognized!'
                                ' Must be a valid instance of one of '
                                'ClassificationDataset or '
                                'RegressionDataset or MLDataset.\n'
                                ' Ignoring {}'.format(ds_path))

    return ds


def load_arff_dataset(ds_path):
    """Convenience utility to quickly load ARFF files into pyradigm format"""

    try:
        ds = ClassificationDataset.from_arff(ds_path)
    except:
        try:
            ds = RegressionDataset.from_arff(ds_path)
        except:
            try:
                ds = MLDataset(arff_path=ds_path)
            except:
                raise TypeError('Error in loading the ARFF dataset @ path below!'
                                ' Ignoring {}'.format(ds_path))

    return ds


def check_compatibility(datasets,
                        class_type,
                        reqd_num_features=None,
                        ):
    """
    Checks whether the given Dataset instances are compatible

    i.e. with same set of subjects, each with the same target in all instances.

    Checks the first dataset in the list against the rest, and returns a boolean
    array.

    Parameters
    ----------
    datasets : Iterable
        A list of n datasets

    class_type : class
        Class of the datasets being compared e.g. ClassificationDataset or
        RegressionDataset or [the deprecated] MLDataset.
        All datasets being compared must be of the same class type.

    reqd_num_features : int
        The required number of features in each dataset.
        Helpful to ensure test sets are compatible with training set,
            as well as within themselves.

    Returns
    -------
    all_are_compatible : bool
        Boolean flag indicating whether all datasets are compatible or not

    compatibility : list
        List indicating whether first dataset is compatible with the rest
        individually.
        This could be useful to select a subset of mutually compatible datasets.
        Length : n-1

    dim_mismatch : bool
        Boolean flag indicating mismatch in dimensionality from that specified

    size_descriptor : tuple
        A tuple with values for (num_samplets, reqd_num_features)
        - num_samplets must be common for all datasets that are evaluated for
        compatibility
        - reqd_num_features is None (when no check on dimensionality is perfomed), or
            list of corresponding dimensionalities for each input dataset

    """

    from collections import Iterable
    if not is_iterable_but_not_str(datasets):
        raise TypeError('Input must be an iterable '
                        'i.e. (list/tuple) of MLdataset/similar instances')

    datasets = list(datasets)  # to make it indexable if coming from a set
    num_datasets = len(datasets)

    check_dimensionality = False
    dim_mismatch = False
    if reqd_num_features is not None:
        if isinstance(reqd_num_features, Iterable):
            if len(reqd_num_features) != num_datasets:
                raise ValueError(
                        'Specify dimensionality for exactly {} datasets.'
                        ' Given for a different number {}'
                        ''.format(num_datasets, len(reqd_num_features)))
            reqd_num_features = list(map(int, reqd_num_features))
        else:  # same dimensionality for all
            reqd_num_features = [int(reqd_num_features)] * num_datasets

        check_dimensionality = True
    else:
        # to enable iteration
        reqd_num_features = [None, ] * num_datasets

    pivot = datasets[0]
    if isinstance(pivot, str):
        pivot = class_type(dataset_path=pivot)
    elif not isinstance(pivot, BaseDataset):
        raise TypeError('All datasets in pyradigm must be subclasses of '
                        'BaseDataset')

    if check_dimensionality and pivot.num_features != reqd_num_features[0]:
        warn('Dimensionality mismatch! Expected {} whereas current {}.'
             ''.format(reqd_num_features[0], pivot.num_features))
        dim_mismatch = True

    compatible = list()
    for ds, reqd_dim in zip(datasets[1:], reqd_num_features[1:]):

        if isinstance(ds, str):
            ds = class_type(dataset_path=ds)
        elif not isinstance(ds, BaseDataset):
            raise TypeError('All datasets in pyradigm must be subclasses of '
                            'BaseDataset')

        is_compatible = True
        # compound bool will short-circuit, not optim required
        if pivot.num_samplets != ds.num_samplets \
                or pivot.samplet_ids != ds.samplet_ids \
                or pivot.targets != ds.targets:
            is_compatible = False

        if check_dimensionality and reqd_dim != ds.num_features:
            warn('Dimensionality mismatch! '
                 'Expected {} whereas current {}.'
                 ''.format(reqd_dim, ds.num_features))
            dim_mismatch = True

        compatible.append(is_compatible)

    return all(compatible), compatible, dim_mismatch, \
           (pivot.num_samplets, reqd_num_features)


def attr_generator(attr_type, count):
    """Generates distributions of a given type"""

    attr_type = attr_type.lower()
    if attr_type in ('int', 'age'):
        return np.random.randint(100, size=count)
    elif attr_type in ('float', 'weight'):
        return 100 * np.abs(np.random.rand(count))
    elif attr_type in ('sex', 'gender'):
        return np.random.choice(['male', 'female', 'other'], count, replace=True)
    elif attr_type in ('site',):
        return np.random.choice(['site{}'.format(ss) for ss in range(6)],
                                count, replace=True)
    elif isinstance(attr_type, Iterable):
        return np.random.choice(attr_type, count, replace=True)
    else:
        raise ValueError('Invalid type: must be int or float.'
                         ' Or an array of values to sample from.'
                         ' Type can also be age, sex, gender, weight, or site.')


def make_random_dataset(max_num_classes=20,
                        min_class_size=20,
                        max_class_size=50,
                        max_dim=100,
                        stratified=True,
                        with_missing_data=False,
                        class_type=ClassificationDataset,
                        min_num_classes=2,
                        attr_names=None,
                        attr_types=None):
    "Generates a random Dataset for use in testing."

    smallest = min(min_class_size, max_class_size)
    max_class_size = max(min_class_size, max_class_size)
    largest = max(50, max_class_size)
    largest = max(smallest + 3, largest)

    if min_num_classes < 2:
        min_num_classes = 2

    if max_num_classes <= min_num_classes:
        max_num_classes = min_num_classes

    num_classes = np.random.randint(min_num_classes, max_num_classes, 1)

    if type(num_classes) == np.ndarray:
        num_classes = num_classes[0]
    if not stratified:
        class_sizes = np.random.randint(smallest, largest + 1, num_classes)
    else:
        class_sizes = np.repeat(np.random.randint(smallest, largest), num_classes)

    num_samplets = class_sizes.sum()
    num_features = np.random.randint(min(3, max_dim), max(3, max_dim), 1)[0]
    # feat_names = [ str(x) for x in range(num_features)]

    class_ids = list()
    labels = list()
    for cl in range(num_classes):
        if issubclass(class_type, RegressionDataset):
            class_ids.append(cl)
        else:
            class_ids.append('class-{}'.format(cl))
        labels.append(int(cl))

    # attributes
    if attr_names is not None:
        if len(attr_names) != len(attr_types):
            raise ValueError('Differing number of names and types for attributes!')
        attrs = dict()
        for name, typ in zip(attr_names, attr_types):
            attrs[name] = attr_generator(typ, num_samplets)

    ds = class_type()
    s_index = 0
    for cc, class_ in enumerate(class_ids):
        subids = ['s{}-c{}'.format(ix, cc) for ix in range(class_sizes[cc])]
        for sid in subids:
            features = feat_generator(num_features)
            if with_missing_data:
                rand_loc = np.random.randint(num_features)
                features[rand_loc] = missing_value_indicator
            if isinstance(ds, MLDataset):
                ds.add_sample(sid, features, int(cc), class_)
            else:
                if attr_names is not None:
                    a_values = [attrs[a_name][s_index] for a_name in attr_names]
                    ds.add_samplet(sid, features, class_,
                                   attr_names=attr_names, attr_values=a_values)
                else:
                    ds.add_samplet(sid, features, class_)

            s_index += 1

    return ds


def make_random_ClfDataset(max_num_classes=20,
                           min_class_size=20,
                           max_class_size=50,
                           max_dim=100,
                           stratified=True,
                           min_num_classes=2,
                           attr_names=None,
                           attr_types=None
                           ):
    "Generates a random ClassificationDataset for use in testing."

    return make_random_dataset(max_num_classes=max_num_classes,
                               min_class_size=min_class_size,
                               max_class_size=max_class_size,
                               max_dim=max_dim,
                               stratified=stratified,
                               class_type=ClassificationDataset,
                               min_num_classes=min_num_classes,
                               attr_names=attr_names, attr_types=attr_types)


def make_random_RegrDataset(min_size=20,
                            max_size=50,
                            max_dim=100,
                            with_missing_data=False,
                            attr_names=None,
                            attr_types=None
                            ):
    "Generates a random ClassificationDataset for use in testing."

    smallest = min(min_size, max_size)
    max_size = max(min_size, max_size)
    largest = max(50, max_size)
    largest = max(smallest + 3, largest)

    sample_size = np.random.randint(smallest, largest + 1)

    num_features = np.random.randint(min(3, max_dim), max(3, max_dim), 1)[0]

    # attributes
    if attr_names is not None:
        if len(attr_names) != len(attr_types):
            raise ValueError('Differing number of names and types for attributes!')
        attrs = dict()
        for name, typ in zip(attr_names, attr_types):
            attrs[name] = attr_generator(typ, sample_size)

    ds = RegressionDataset()

    subids = ['s{}'.format(ix) for ix in range(sample_size)]
    for counter, sid in enumerate(subids):
        features = feat_generator(num_features)
        target = np.random.randint(sample_size)
        if with_missing_data:
            rand_loc = np.random.randint(num_features)
            features[rand_loc] = missing_value_indicator

        if attr_names is not None:
            a_values = [attrs[a_name][counter] for a_name in attr_names]
            ds.add_samplet(sid, features, target,
                           attr_names=attr_names, attr_values=a_values)
        else:
            ds.add_samplet(sid, features, target)

    return ds


def make_random_MLdataset(max_num_classes=20,
                          min_class_size=20,
                          max_class_size=50,
                          max_dim=100,
                          stratified=True):
    "Generates a random MLDataset for use in testing."

    return make_random_dataset(max_num_classes=max_num_classes,
                               min_class_size=min_class_size,
                               max_class_size=max_class_size,
                               max_dim=max_dim,
                               stratified=stratified,
                               class_type=MLDataset)


def dataset_with_new_features_same_everything_else(in_ds, max_feat_dim):
    """Helper utility for MultiDataset purposes."""

    feat_dim = np.random.randint(1, max_feat_dim)
    out_ds = in_ds.__class__()
    for id_ in in_ds.samplet_ids:
        out_ds.add_samplet(id_,
                           np.random.rand(feat_dim),
                           target=in_ds.targets[id_])
    # copying attr
    out_ds.attr = in_ds.attr
    out_ds.dataset_attr = in_ds.dataset_attr
    out_ds.attr_dtype = in_ds.attr_dtype

    return out_ds
