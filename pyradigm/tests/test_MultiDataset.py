import numpy as np
from os.path import join as pjoin, exists as pexists, realpath, dirname
from os import makedirs
from functools import partial
from pyradigm import (MultiDatasetClassify, MultiDatasetRegress,
                      ClassificationDataset as ClfDataset,
                      RegressionDataset as RegrDataset)
from pyradigm.utils import (make_random_ClfDataset, make_random_RegrDataset,
                            make_random_dataset)

test_dir = dirname(__file__)
out_dir = realpath(pjoin(test_dir, 'tmp'))
makedirs(out_dir, exist_ok=True)

min_num_modalities = 3
max_num_modalities = 10
max_feat_dim = 10

ds_class = RegrDataset


def make_fully_separable_classes(max_class_size=10, max_dim=22):
    from sklearn.datasets import make_blobs

    random_center = np.random.rand(max_dim)
    cluster_std = 1.5
    centers = [random_center, random_center + cluster_std * 6]
    blobs_X, blobs_y = make_blobs(n_samples=max_class_size, n_features=max_dim,
                                  centers=centers, cluster_std=cluster_std)

    unique_labels = np.unique(blobs_y)
    class_ids = {lbl: str(lbl) for lbl in unique_labels}

    new_ds = ClfDataset()
    for index, row in enumerate(blobs_X):
        new_ds.add_samplet('sub{}'.format(index),
                           row, class_ids[blobs_y[index]])

    return new_ds


def new_dataset_with_same_ids_targets(in_ds):
    feat_dim = np.random.randint(1, max_feat_dim)
    out_ds = in_ds.__class__()
    for id_ in in_ds.samplet_ids:
        out_ds.add_samplet(id_,
                           np.random.rand(feat_dim),
                           target=in_ds.targets[id_])
    # copying attr
    out_ds.attr = in_ds.attr
    out_ds.dataset_attr = in_ds.dataset_attr

    return out_ds


num_modalities = np.random.randint(min_num_modalities, max_num_modalities)


def test_holdout():
    """"""

    for multi_class, ds_class in zip((MultiDatasetClassify, MultiDatasetRegress),
                                     (ClfDataset, RegrDataset)):

        # ds = make_fully_separable_classes()
        ds = make_random_dataset(5, 20, 50, 10, stratified=False,
                                 class_type=ds_class)
        multi = multi_class()

        for ii in range(num_modalities):
            multi.append(new_dataset_with_same_ids_targets(ds), identifier=ii)

        # for trn, tst in multi.holdout(num_rep=5, return_ids_only=True):
        #     print('train: {}\ntest: {}\n'.format(trn, tst))

        print(multi)

        return_ids_only = False
        for trn, tst in multi.holdout(num_rep=5, train_perc=0.51,
                                      return_ids_only=return_ids_only):
            if return_ids_only:
                print('train: {}\ttest: {}\n'.format(len(trn), len(tst)))
            else:
                for aa, bb in zip(trn, tst):
                    if aa.num_features != bb.num_features:
                        raise ValueError(
                            'train and test dimensionality do not match!')

                    print('train: {}\ntest : {}\n'.format(aa.shape, bb.shape))

        print()


def test_init_list_of_paths():
    """main use case for neuropredict"""

    for multi_class, ds_class in zip((MultiDatasetClassify, MultiDatasetRegress),
                                     (ClfDataset, RegrDataset)):

        ds = make_random_dataset(5, 20, 50, 10, stratified=False,
                                 class_type=ds_class)

        paths = list()
        for ii in range(num_modalities):
            new_ds = new_dataset_with_same_ids_targets(ds)
            path = pjoin(out_dir, 'ds{}.pkl'.format(ii))
            new_ds.save(path)
            paths.append(path)

        try:
            multi = multi_class(dataset_spec=paths)
        except:
            raise ValueError('MultiDataset constructor via list of paths does not '
                             'work!')


def test_attributes():
    """basic tests to ensure attrs are handled properly in MultiDataset"""

    random_clf_ds = partial(make_random_ClfDataset, 5, 20, 50, 10, stratified=False)
    random_regr_ds = partial(make_random_RegrDataset, 20, 100, 50)
    for multi_cls, ds_gen in zip((MultiDatasetClassify, MultiDatasetRegress),
                                 (random_clf_ds, random_regr_ds)):
        ds = ds_gen(attr_names=('age', 'gender'),
                    attr_types=('age', 'gender'))
        multi_ds = multi_cls()
        multi_ds.append(ds, 0)
        for ii in range(num_modalities - 1):
            new_ds = new_dataset_with_same_ids_targets(ds)
            multi_ds.append(new_ds, ii + 1)

        if multi_ds.common_attr != ds.attr:
            raise ValueError('Attributes in indiv Dataset and MultiDataset differ!')

        print('!! --- write tests for ds.dataset_attr and mutli_ds.meta')
