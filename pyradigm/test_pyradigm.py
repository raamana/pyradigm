import tempfile
import os, sys
import numpy as np

sys.dont_write_bytecode = True

from pytest import raises, warns, set_trace

from pyradigm import MLDataset

out_dir  = '.'

num_classes  = np.random.randint( 2, 50)
class_sizes  = np.random.randint(10, 1000, num_classes)
num_features = np.random.randint(10, 500)

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

def test_empty():
    assert not empty_dataset

def test_num_classes():
    assert test_dataset.num_classes == num_classes

def test_num_features():
    assert test_dataset.num_features == num_features

def test_num_features_setter():
    with raises(AttributeError):
        test_dataset.num_features == 0

def test_num_samples():
    assert test_dataset.num_samples == sum(class_sizes)

def test_substract():
    assert other_classes_ds.num_samples == sum(class_sizes) - class_sizes[rand_index]

def test_get_class_list():
    assert other_classes_ds == other_classes_get_with_list

def test_add():
    a = other_classes_ds + random_class_ds
    n = a.num_samples
    n1 = other_classes_ds.num_samples
    n2 = random_class_ds.num_samples
    assert n1 + n2 == n

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
    fewer_labels.pop(fewer_labels.keys()[0])

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
    fewer_classes.pop(fewer_classes.keys()[0])

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
        test_dataset.add_sample(new_id, data_diff_dim, None, None)

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
    subset= np.random.random_integers(1, nf-1, size=subset_len )
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
    return np.random.random_integers(1, n, min(n, k))

def test_glance():
    for k in np.random.randint(1, test_dataset.num_samples, 10):
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
    for count in np.random.randint(1, smallest_size, 10):
        subset = copy_dataset.random_subset_ids_by_count(count_per_class=count)
        assert len(subset) == num_classes*count

def test_train_test_split_ids_count():
    smallest_size = min(class_sizes)
    for count in np.random.randint(1, smallest_size, 10):
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
