import os, sys
import numpy as np
from os.path import join as pjoin, exists as pexists, realpath, basename, dirname, isfile

sys.dont_write_bytecode = True

from pytest import raises, warns

from sys import version_info

if version_info.major==2 and version_info.minor==7:
    from pyradigm import MLDataset
elif version_info.major > 2:
    try:
        from pyradigm.pyradigm import MLDataset
    except ImportError:
        from pyradigm import MLDataset
    except:
        raise ImportError('could not import pyradigm')
else:
    raise NotImplementedError('pyradigm supports only 2.7.13 or 3+. '
                              'Upgrade to Python 3+ is recommended.')

out_dir  = '.'

num_classes  = np.random.randint( 2, 50)
class_sizes  = np.random.randint(10, 100, num_classes)
num_features = np.random.randint(10, 100)

class_set    = np.array([ 'C{:05d}'.format(x) for x in range(num_classes)])
feat_names   = np.array([ str(x) for x in range(num_features) ])

test_dataset = MLDataset()
for class_index, class_id in enumerate(class_set):
    for sub_ix in range(class_sizes[class_index]):
        subj_id = '{}_S{:05d}'.format(class_set[class_index],sub_ix)
        feat = np.random.random(num_features)
        test_dataset.add_sample(subj_id, feat, class_index, class_id, feat_names)

out_file = os.path.join(out_dir,'random_example_dataset.pkl')
test_dataset.save(out_file)

# same IDs, new features
same_ids_new_feat = MLDataset()
for sub_id in test_dataset.keys:
    feat = np.random.random(num_features)
    same_ids_new_feat.add_sample(sub_id, feat,
                                 test_dataset.labels[sub_id],
                                 test_dataset.classes[sub_id])

same_ids_new_feat.feature_names = np.array([ 'new_f{}'.format(x) for x in range(
        num_features) ])

test_dataset.description = 'test dataset'
print(test_dataset)
print('default format:\n {}'.format(test_dataset))
print('full repr     :\n {:full}'.format(test_dataset))
print('string/short  :\n {:s}'.format(test_dataset))

class_set, label_set, class_sizes = test_dataset.summarize_classes()

reloaded_dataset = MLDataset(filepath=out_file, description='reloaded test_dataset')

copy_dataset = MLDataset(in_dataset=test_dataset)

rand_index = np.random.randint(0,len(class_set),1)[0]
random_class_name = class_set[rand_index]
random_class_ds = test_dataset.get_class(random_class_name)

other_classes_ds = test_dataset - random_class_ds

other_class_set = set(class_set)-set([random_class_name])
other_classes_get_with_list = test_dataset.get_class(other_class_set)

recombined = other_classes_ds + random_class_ds

empty_dataset = MLDataset()

test2 = MLDataset()
test3 = MLDataset()

# TODO write tests for CLI

def test_empty():
    assert not empty_dataset

def test_num_classes():
    assert test_dataset.num_classes == num_classes

def test_num_features():
    assert test_dataset.num_features == num_features

def test_num_features_setter():
    with raises(AttributeError):
        test_dataset.num_features = 0

def test_num_samples():
    assert test_dataset.num_samples == sum(class_sizes)

def test_subtract():
    assert other_classes_ds.num_samples == sum(class_sizes) - class_sizes[rand_index]

def test_get_class_list():
    assert other_classes_ds == other_classes_get_with_list

def test_add():
    a = other_classes_ds + random_class_ds
    n = a.num_samples
    n1 = other_classes_ds.num_samples
    n2 = random_class_ds.num_samples
    assert n1 + n2 == n

    assert set(a.sample_ids) == set(other_classes_ds.sample_ids+random_class_ds.sample_ids)
    assert a.num_features == other_classes_ds.num_features == random_class_ds.num_features
    assert all(a.feature_names == other_classes_ds.feature_names)

    comb_ds = test_dataset + same_ids_new_feat
    comb_names = np.concatenate([ test_dataset.feature_names,
                            same_ids_new_feat.feature_names])
    if not all(comb_ds.feature_names == comb_names):
        raise ValueError('feature names were not carried forward in combining two '
                         'datasets with same IDs and different feature names!')

def test_label_validity():

    feat = np.random.random(num_features)
    for invalid_label in [np.NaN, 1j, [2, 3], 'label']:
        with raises((ValueError, TypeError)):
            test_dataset.add_sample('subj_id', feat, invalid_label, class_set[0])

def test_class_id_validity():

    feat = np.random.random(num_features)

    # class_id can not be an iterable
    for invalid_id in [('df', 2), [34, ]]:
        with raises((ValueError, TypeError)):
            test_dataset.add_sample('subj_id', feat, label_set[0], invalid_id)

    # label_set[ix], ix=1 to n-1 can only be associated with class_set[ix]
    num_labels = len(label_set)
    for ix in range(num_labels):
        if ix > 1:
            new_ix = ix-1
        else:
            new_ix = ix+1
        with raises((ValueError, TypeError)):
            test_dataset.add_sample('subj_id', feat, label_set[ix], class_set[new_ix])


def test_cant_read_nonexisting_file():
    with raises(IOError):
        a = MLDataset('/nonexistentrandomdir/disofddlsfj/arbitrary.noname.pkl')

def test_cant_write_to_nonexisting_dir():
    with raises(IOError):
        test_dataset.save('/nonexistentrandomdir/jdknvoindvi93/arbitrary.noname.pkl')

def test_invalid_constructor():
    with raises(ValueError):
        a = MLDataset(in_dataset='/nonexistentrandomdir/disofddlsfj/arbitrary.noname.pkl')

    with raises(ValueError):
        # data simply should not be a dict
        b = MLDataset(filepath=None, in_dataset=None, data=list())

    with raises(ValueError):
        c = MLDataset(filepath=None,
                      in_dataset=None, data=None, labels=None,
                      classes='invalid_value')

def test_return_data_labels():
    matrix, vec_labels, sub_ids = test_dataset.data_and_labels()
    assert len(vec_labels)==len(sub_ids)
    assert len(vec_labels)==matrix.shape[0]

def test_init_with_dict():
    new_ds = MLDataset(data=test_dataset.data, labels=test_dataset.labels, classes=test_dataset.classes)
    assert new_ds == test_dataset

def test_labels_setter():
    fewer_labels = test_dataset.labels
    label_keys = list(fewer_labels.keys())
    fewer_labels.pop(label_keys[0])

    with raises(ValueError):
        test_dataset.labels = fewer_labels

    same_len_diff_key = fewer_labels
    same_len_diff_key[u'sldiursvdkvjs'] = 1
    with raises(ValueError):
        test_dataset.labels = same_len_diff_key

    # must be dict
    with raises(ValueError):
        test_dataset.labels = None

def test_classes_setter():
    fewer_classes = test_dataset.classes
    classes_keys = list(fewer_classes.keys())
    fewer_classes.pop(classes_keys[0])

    with raises(ValueError):
        test_dataset.classes = fewer_classes

    same_len_diff_key = fewer_classes
    same_len_diff_key['sldiursvdkvjs'] = 'lfjd'
    with raises(ValueError):
        test_dataset.classes = same_len_diff_key

def test_feat_names_setter():

    # fewer
    with raises(ValueError):
        test_dataset.feature_names = feat_names[0:test_dataset.num_features-2]

    # too many
    with raises(ValueError):
        test_dataset.feature_names = np.append(feat_names, 'blahblah')

def test_add_existing_id():
    sid = test_dataset.sample_ids[0]
    with raises(ValueError):
        test_dataset.add_sample(sid, None, None)

def test_add_new_id_diff_dim():
    new_id = 'dsfdkfslj38748937439kdshfkjhf38'
    sid = test_dataset.sample_ids[0]
    data_diff_dim = np.random.rand(test_dataset.num_features+1,1)
    with raises(ValueError):
        test_dataset.add_sample(new_id, data_diff_dim, label_set[0], class_set[0])

def test_del_nonexisting_id():
    nonexisting_id = u'dsfdkfslj38748937439kdshfkjhf38'
    with warns(UserWarning):
        test_dataset.del_sample(nonexisting_id)

def test_get_nonexisting_class():
    nonexisting_id = u'dsfdkfslj38748937439kdshfkjhf38'
    with raises(ValueError):
        test_dataset.get_class(nonexisting_id)

def test_rand_feat_subset():
    nf = copy_dataset.num_features
    subset_len = np.random.randint(1, nf)
    subset= np.random.randint(1, nf-1, size=subset_len )
    subds = copy_dataset.get_feature_subset(subset)
    assert subds.num_features == subset_len

def test_eq_self():
    assert test_dataset == test_dataset

def test_eq_copy():
    new_copy = MLDataset(in_dataset=copy_dataset)
    assert new_copy == copy_dataset

def test_unpickling():
    out_file = os.path.join(out_dir, 'random_pickled_dataset.pkl')
    copy_dataset.save(out_file)
    reloaded_dataset = MLDataset(filepath=out_file, description='reloaded test_dataset')
    assert copy_dataset == reloaded_dataset

def test_subset_class():
    assert random_class_ds.num_samples == class_sizes[rand_index]


def test_get_subset():
    assert random_class_ds == reloaded_dataset.get_class(random_class_name)

    nonexisting_id = u'dsfdkfslj38748937439kdshfkjhf38'
    with warns(UserWarning):
        test_dataset.get_subset(nonexisting_id)

def test_membership():
    rand_idx = np.random.randint(0, test_dataset.num_samples)
    member = test_dataset.sample_ids[rand_idx]
    not_member = u'sdfdkshfdsk34823058wdkfhd83hifnalwe8fh8t'
    assert member in test_dataset
    assert not_member not in test_dataset

def rand_ints_range(n, k):
    return np.random.randint(1, n, min(n, k))

def test_glance():
    for k in np.random.randint(1, test_dataset.num_samples-1, 10):
        glanced_subset = test_dataset.glance(k)
        assert len(glanced_subset) == k


def test_random_subset():
    for perc in np.arange(0.1, 1, 0.2):
        subset = copy_dataset.random_subset(perc_in_class=perc)
        # separating the calculation by class to mimic the implementation in the class
        expected_size = sum([np.int64(np.floor(n_in_class*perc)) for n_in_class in class_sizes])
        assert subset.num_samples == expected_size

def test_random_subset_by_count():

    smallest_size = min(class_sizes)
    for count in np.random.randint(1, smallest_size, 7):
        subset = copy_dataset.random_subset_ids_by_count(count_per_class=count)
        assert len(subset) == num_classes*count

def test_train_test_split_ids_count():
    smallest_size = min(class_sizes)
    for count in np.random.randint(1, smallest_size, 7):
        subset_train, subset_test = copy_dataset.train_test_split_ids(count_per_class=count)
        assert len(subset_train) == num_classes*count
        assert len(subset_test ) == copy_dataset.num_samples-num_classes*count
        assert len(set(subset_train).intersection(subset_test))==0

    with raises(ValueError):
        copy_dataset.train_test_split_ids(count_per_class=-1)

    with raises(ValueError):
        copy_dataset.train_test_split_ids(count_per_class=copy_dataset.num_samples+1.0)

    with raises(ValueError):
        # both cant be specified at the same time
        copy_dataset.train_test_split_ids(count_per_class=2, train_perc=0.5)

def test_train_test_split_ids_perc():

    for perc in np.arange(0.25, 1.0, 0.1):
        subset_train, subset_test = copy_dataset.train_test_split_ids(train_perc=perc)
        expected_train_size = sum(np.floor(class_sizes*perc))
        assert len(subset_train) == expected_train_size
        assert len(subset_test) == copy_dataset.num_samples-expected_train_size
        assert len(set(subset_train).intersection(subset_test))==0

    with raises(ValueError):
        subset_train, subset_test = copy_dataset.train_test_split_ids(train_perc=0.00001)

    with raises(ValueError):
        copy_dataset.train_test_split_ids(train_perc=1.1)

    with raises(ValueError):
        copy_dataset.train_test_split_ids(train_perc=-1)

# ------------------------------------------------
# different file formats
# ------------------------------------------------

def test_load_arff():
    arff_path = realpath(pjoin(dirname(__file__),'../../example_datasets/iris.arff'))
    mld = MLDataset(arff_path=arff_path)

    if mld.num_samples != 150:
        raise ValueError('number of samples mismatch')

    if mld.num_features != 4:
        raise ValueError('number of features mismatch')

    if mld.num_classes != 3:
        raise ValueError('number of classes mismatch')

    if len(mld.feature_names) != 4:
        raise ValueError('length of feature names do not match number of features')

    # print(mld)

# test_load_arff()
test_add()