import tempfile
import os
import numpy as np

from pyradigm import MLDataset

out_dir  = '.' # '/Users/Reddy/rotman/neuropredict/test_MLdatasets'
for ii in range(1):
    num_classes = np.random.randint(2, 150, 1)
    class_set = [ chr(x+65)+str(x) for x in range(num_classes)]
    class_sizes = np.random.randint(5, 200, num_classes)
    num_features = np.random.randint(1, 300, 1).take(0)

    test_dataset = MLDataset()
    for class_index, class_id in enumerate(class_set):
        for sub_ix in xrange(class_sizes[class_index]):
            subj_id = class_set[class_index] + str(sub_ix)
            feat = np.random.random(num_features)
            test_dataset.add_sample(subj_id, feat, class_index, class_id)

    out_file = os.path.join(out_dir,'random_dataset{}.pkl'.format(ii))
    test_dataset.save(out_file)

class_set, label_set, class_sizes = test_dataset.summarize_classes()

reloaded_dataset = MLDataset(filepath=out_file, description='reloaded test_dataset')

copy_dataset = MLDataset(in_dataset=test_dataset)

class1_name = class_set[1]
class1 = test_dataset.get_class(class1_name)
other_classes = test_dataset - class1

empty_dataset = MLDataset()

test2 = MLDataset()
test3 = MLDataset()

def test_empty():
    assert not empty_dataset


def test_eq_self():
    assert test_dataset == test_dataset


def test_eq_copy():
    assert test_dataset == copy_dataset


def test_unpickling():
    assert test_dataset == reloaded_dataset


def test_num_samples():
    assert test_dataset.num_samples == sum(class_sizes)


def test_num_features():
    assert test_dataset.num_features == num_features


def test_substract():
    assert other_classes.num_samples == sum(class_sizes) - class_sizes[1]


def test_add():
    a = other_classes + class1
    n = a.num_samples
    n1 = other_classes.num_samples
    n2 = class1.num_samples
    assert n1 + n2 == n


def test_subset_class():
    assert class1.num_samples == class_sizes[1]


def test_get_subset():
    assert class1 == reloaded_dataset.get_class(class1_name)


def test_glance():
    for k in xrange(1, test_dataset.num_samples, 2):
        glanced_subset = test_dataset.glance(k)
        assert len(glanced_subset) == k


def test_random_subset():
    for perc in np.arange(0.1, 1, 0.1):
        subset = test_dataset.random_subset(perc_in_class=perc)
        # separating the calculation by class to mimic the implementation in the class
        expected_size = sum([int(np.floor(n_in_class*perc)) for n_in_class in class_sizes])
        assert subset.num_samples == expected_size

def test_random_subset_by_count():

    smallest_size = min(class_sizes)
    for count in range(1,int(smallest_size)):
        subset = test_dataset.random_subset_ids_by_count(count_per_class=count)
        assert len(subset) == num_classes*count

def test_train_test_split_ids_count():
    smallest_size = min(class_sizes)
    for count in range(1, int(smallest_size)):
        subset_train, subset_test = test_dataset.train_test_split_ids(count_per_class=count)
        assert len(subset_train) == num_classes*count
        assert len(subset_test) == test_dataset.num_samples-num_classes*count
        assert len(set(subset_train).intersection(subset_test))==0

def test_train_test_split_ids_perc():

    for perc in np.arange(0.1, 0.9, 0.1):
        subset_train, subset_test = test_dataset.train_test_split_ids(train_perc=perc)
        expected_train_size = sum(np.floor(class_sizes*perc))
        assert len(subset_train) == expected_train_size
        assert len(subset_test) == test_dataset.num_samples-expected_train_size
        assert len(set(subset_train).intersection(subset_test))==0


