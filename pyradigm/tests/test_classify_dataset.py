import os
import sys
from os.path import dirname, join as pjoin, realpath

import numpy as np

sys.dont_write_bytecode = True

from pytest import raises, warns

from pyradigm.classify import ClassificationDataset as ClfDataset

out_dir = '.'

num_targets = np.random.randint(2, 50)
target_sizes = np.random.randint(10, 100, num_targets)
num_features = np.random.randint(10, 100)
num_samples = sum(target_sizes)

target_set = np.array([f'C{x:05d}' for x in range(num_targets)])
feat_names = np.array([str(x) for x in range(num_features)])

test_dataset = ClfDataset()
for target_index, target_id in enumerate(target_set):
    for sub_ix in range(target_sizes[target_index]):
        subj_id = f'{target_set[target_index]}_S{sub_ix:05d}'
        feat = np.random.random(num_features)
        test_dataset.add_samplet(subj_id, feat, target_id,
                                 feature_names=feat_names)

out_file = os.path.join(out_dir, 'random_example_dataset.pkl')
test_dataset.save(out_file)

# same IDs, new features
same_ids_new_feat = ClfDataset()
for sub_id in test_dataset.samplet_ids:
    feat = np.random.random(num_features)
    same_ids_new_feat.add_samplet(sub_id, feat,
                                  test_dataset.targets[sub_id])

same_ids_new_feat.feature_names = np.array([f'new_f{x}' for x in range(
        num_features)])

test_dataset.description = 'test dataset'
print(test_dataset)
print('default format:\n {}'.format(test_dataset))
print('full repr     :\n {:full}'.format(test_dataset))
print('string/short  :\n {:s}'.format(test_dataset))

target_set, target_sizes = test_dataset.summarize()

reloaded_dataset = ClfDataset(dataset_path=out_file, 
                              description='reloaded test_dataset')

copy_dataset = ClfDataset(in_dataset=test_dataset)

rand_index = np.random.randint(0, len(target_set), 1)[0]
random_target_name = target_set[rand_index]
random_target_ds = test_dataset.get_class(random_target_name)

other_targets_ds = test_dataset - random_target_ds

other_target_set = set(target_set) - set([random_target_name])
other_targets_get_with_list = test_dataset.get_class(other_target_set)

recombined = other_targets_ds + random_target_ds

empty_dataset = ClfDataset()

test2 = ClfDataset()
test3 = ClfDataset()


def test_empty():
    assert not empty_dataset


def test_target_type():

    rand_id = test_dataset.samplet_ids[np.random.randint(2, num_samples)]
    if not isinstance(test_dataset.targets[rand_id],
                     test_dataset._target_type):
        raise TypeError(f'invalid target type for samplet id {rand_id}')


def test_num_targets():
    assert test_dataset.num_targets == num_targets


def test_num_features():
    assert test_dataset.num_features == num_features


def test_shape():
    assert test_dataset.shape == (num_samples, num_features)


def test_num_features_setter():
    with raises(AttributeError):
        test_dataset.num_features = 0


def test_num_samples():
    assert test_dataset.num_samplets == sum(target_sizes)


def test_subtract():
    assert other_targets_ds.num_samplets == sum(target_sizes) - target_sizes[rand_index]


def test_get_target_list():
    assert other_targets_ds == other_targets_get_with_list


def test_add():
    a = other_targets_ds + random_target_ds
    n = a.num_samplets
    n1 = other_targets_ds.num_samplets
    n2 = random_target_ds.num_samplets
    assert n1 + n2 == n

    assert set(a.samplet_ids) == set(
            other_targets_ds.samplet_ids + random_target_ds.samplet_ids)
    assert a.num_features == other_targets_ds.num_features == \
           random_target_ds.num_features
    assert all(a.feature_names == other_targets_ds.feature_names)

    comb_ds = test_dataset + same_ids_new_feat
    comb_names = np.concatenate([test_dataset.feature_names,
                                 same_ids_new_feat.feature_names])
    if not all(comb_ds.feature_names == comb_names):
        raise ValueError('feature names were not carried forward in combining two '
                         'datasets with same IDs and different feature names!')


def test_set_existing_sample():
    sid = test_dataset.samplet_ids[0]
    new_feat = np.random.random(num_features)

    with raises(KeyError):
        test_dataset[sid + 'nonexisting'] = new_feat

    with raises(ValueError):
        test_dataset[sid] = new_feat[:-2]  # diff dimensionality

    test_dataset[sid] = new_feat
    if not np.all(test_dataset[sid] == new_feat):
        raise ValueError('Bug in replacing features for an existing sample!'
                         'Retrieved features do not match previously set features.')


def test_data_type():

    for in_dtype in [np.float64, int, np.bool_]:
        cds = ClfDataset(dtype=in_dtype)
        cds.add_samplet('a', [1, 2.0, -434], 'class')
        if cds.dtype != in_dtype or cds['a'].dtype != in_dtype:
            raise TypeError('Dataset not maintaining the features in the requested'
                            'dtype {}. They are in {}'.format(in_dtype, cds.dtype))


def test_cant_read_nonexisting_file():
    with raises(IOError):
        a = ClfDataset('/nonexistentrandomdir/disofddlsfj/arbitrary.noname.pkl')


def test_cant_write_to_nonexisting_dir():
    with raises(IOError):
        test_dataset.save('/nonexistentrandomdir/jdknvoindvi93/arbitrary.noname.pkl')


def test_invalid_constructor():
    with raises(TypeError):
        a = ClfDataset(
            in_dataset='/nonexistentrandomdir/disofddlsfj/arbitrary.noname.pkl')

    with raises(ValueError):
        # data simply should not be a dict
        b = ClfDataset(dataset_path=None, in_dataset=None, data=list())

    with raises(ValueError):
        c = ClfDataset(dataset_path=None,
                       in_dataset=None,
                       data=None,
                       targets='invalid_value')


def test_return_data_labels():
    matrix, vec_labels, sub_ids = test_dataset.data_and_labels()
    assert len(vec_labels) == len(sub_ids)
    assert len(vec_labels) == matrix.shape[0]


def test_init_with_dict():
    new_ds = ClfDataset(data=test_dataset.data,
                        targets=test_dataset.targets)
    assert new_ds == test_dataset


# def test_labels_setter():
#     fewer_labels = test_dataset.labels
#     label_keys = list(fewer_labels.samplet_ids())
#     fewer_labels.pop(label_keys[0])
# 
#     with raises(ValueError):
#         test_dataset.labels = fewer_labels
# 
#     same_len_diff_key = fewer_labels
#     same_len_diff_key[u'sldiursvdkvjs'] = 1
#     with raises(ValueError):
#         test_dataset.labels = same_len_diff_key
# 
#     # must be dict
#     with raises(ValueError):
#         test_dataset.labels = None


def test_targets_setter():
    fewer_targets = test_dataset.targets
    targets_keys = list(fewer_targets.keys())
    fewer_targets.pop(targets_keys[0])

    with raises(ValueError):
        test_dataset.targets = fewer_targets

    same_len_diff_key = fewer_targets
    same_len_diff_key['sldiursvdkvjs'] = 'lfjd'
    with raises(ValueError):
        test_dataset.targets = same_len_diff_key


def test_feat_names_setter():
    # fewer
    with raises(ValueError):
        test_dataset.feature_names = feat_names[0:test_dataset.num_features - 2]

    # too many
    with raises(ValueError):
        test_dataset.feature_names = np.append(feat_names, 'blahblah')


def test_add_existing_id():
    sid = test_dataset.samplet_ids[0]
    with raises(ValueError):
        test_dataset.add_samplet(sid, None, None)


def test_add_new_id_diff_dim():
    new_id = 'dsfdkfslj38748937439kdshfkjhf38'
    sid = test_dataset.samplet_ids[0]
    data_diff_dim = np.random.rand(test_dataset.num_features + 1, 1)
    with raises(ValueError):
        test_dataset.add_samplet(new_id, data_diff_dim, None, None)


def test_del_nonexisting_id():
    nonexisting_id = u'dsfdkfslj38748937439kdshfkjhf38'
    with warns(UserWarning):
        test_dataset.del_samplet(nonexisting_id)


def test_get_nonexisting_class():
    nonexisting_id = u'dsfdkfslj38748937439kdshfkjhf38'
    with raises(ValueError):
        test_dataset.get_class(nonexisting_id)


def test_rand_feat_subset():
    nf = copy_dataset.num_features
    subset_len = np.random.randint(1, nf)
    subset = np.random.randint(1, nf, size=subset_len)
    subds = copy_dataset.get_feature_subset(subset)
    assert subds.num_features == subset_len


def test_eq_self():
    assert test_dataset == test_dataset


def test_eq_copy():
    new_copy = ClfDataset(in_dataset=copy_dataset)
    assert new_copy == copy_dataset


def test_unpickling():
    out_file = os.path.join(out_dir, 'random_pickled_dataset.pkl')
    copy_dataset.save(out_file)
    reloaded_dataset = ClfDataset(dataset_path=out_file,
                                  description='reloaded test_dataset')
    assert copy_dataset == reloaded_dataset


def test_subset_class():
    assert random_target_ds.num_samplets == target_sizes[rand_index]


def test_get_subset():
    assert random_target_ds == reloaded_dataset.get_class(random_target_name)

    nonexisting_id = u'dsfdkfslj38748937439kdshfkjhf38'
    with warns(UserWarning):
        test_dataset.get_subset(nonexisting_id)


def test_membership():
    rand_idx = np.random.randint(0, test_dataset.num_samplets)
    member = test_dataset.samplet_ids[rand_idx]
    not_member = u'sdfdkshfdsk34823058wdkfhd83hifnalwe8fh8t'
    assert member in test_dataset
    assert not_member not in test_dataset


def rand_ints_range(n, k):
    return np.random.randint(1, n, size=min(n, k))


def test_glance():
    for k in np.random.randint(1, test_dataset.num_samplets - 1, 10):
        glanced_subset = test_dataset.glance(k)
        assert len(glanced_subset) == k


def test_random_subset():
    for perc in np.arange(0.1, 1, 0.2):
        subset = copy_dataset.random_subset(perc_in_class=perc)
        # separating the calculation by class to mimic the implementation in the 
        # class
        expected_size = sum([np.int64(np.floor(n_in_class * perc)) for n_in_class in
                             target_sizes])
        assert subset.num_samplets == expected_size


def test_random_subset_by_count():
    smallest_size = min(target_sizes)
    for count in np.random.randint(1, smallest_size, 7):
        subset = copy_dataset.random_subset_ids_by_count(count_per_class=count)
        assert len(subset) == num_targets * count


def test_train_test_split_ids_count():
    smallest_size = min(target_sizes)
    for count in np.random.randint(1, smallest_size, 7):
        subset_train, subset_test = copy_dataset.train_test_split_ids(
            count_per_class=count)
        assert len(subset_train) == num_targets * count
        assert len(subset_test) == copy_dataset.num_samplets - num_targets * count
        assert len(set(subset_train).intersection(subset_test)) == 0

    with raises(ValueError):
        copy_dataset.train_test_split_ids(count_per_class=-1)

    with raises(ValueError):
        copy_dataset.train_test_split_ids(
            count_per_class=copy_dataset.num_samplets + 1.0)

    with raises(ValueError):
        # both cant be specified at the same time
        copy_dataset.train_test_split_ids(count_per_class=2, train_perc=0.5)


def test_train_test_split_ids_perc():
    for perc in np.arange(0.25, 1.0, 0.1):
        subset_train, subset_test = copy_dataset.train_test_split_ids(
            train_perc=perc)
        expected_train_size = sum(np.floor(target_sizes * perc))
        assert len(subset_train) == expected_train_size
        assert len(subset_test) == copy_dataset.num_samplets - expected_train_size
        assert len(set(subset_train).intersection(subset_test)) == 0

    with raises(ValueError):
        subset_train, subset_test = copy_dataset.train_test_split_ids(
            train_perc=0.00001)

    with raises(ValueError):
        copy_dataset.train_test_split_ids(train_perc=1.1)

    with raises(ValueError):
        copy_dataset.train_test_split_ids(train_perc=-1)


# ------------------------------------------------
# different file formats
# ------------------------------------------------

def test_load_arff():
    arff_path = realpath(pjoin(dirname(__file__),
                               '..', '..', 'example_datasets', 'iris.arff'))
    mld = ClfDataset.from_arff(arff_path=arff_path)

    if mld.num_samplets != 150:
        raise ValueError('number of samples mismatch')

    if mld.num_features != 4:
        raise ValueError('number of features mismatch')

    if mld.num_targets != 3:
        raise ValueError('number of classes mismatch')

    if len(mld.feature_names) != 4:
        raise ValueError('length of feature names do not match number of features')

    # print(mld)


test_data_type()