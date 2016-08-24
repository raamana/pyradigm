import tempfile

import numpy as np

from pyradigm import MLDataset

# example set of classes
class_set = ['A', 'B', 'C']
# creating a toy test_dataset
class_sizes = np.random.randint(3, 30, len(class_set))
num_features = np.random.randint(1, 20, 1).take(0)

empty_dataset = MLDataset()

test_dataset = MLDataset()
test2 = MLDataset()
test3 = MLDataset()

for class_index, class_id in enumerate(class_set):
    for sub_ix in xrange(class_sizes[class_index]):
        subj_id = class_set[class_index] + str(sub_ix)
        feat = np.random.random(num_features)
        test_dataset.add_sample(subj_id, feat, class_index, class_id)

out_file = tempfile.mktemp()
test_dataset.save(out_file)
reloaded_dataset = MLDataset(filepath=out_file, description='reloaded test_dataset')

copy_dataset = MLDataset(in_dataset=test_dataset)

class1_name = class_set[1]
class1 = test_dataset.get_class(class1_name)
other_classes = test_dataset - class1


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
        subset = test_dataset.random_subset(perc_per_class=perc)
        # separating the calculation by class to mimic the implementation in the class
        expected_size = sum([int(round(n_in_class*perc)) for n_in_class in test_dataset.class_sizes.values()])
        assert subset.num_samples == expected_size
