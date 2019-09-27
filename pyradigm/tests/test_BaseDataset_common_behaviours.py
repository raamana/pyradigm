"""

Tests to ensure certain behaviours among all child classes of BaseDataset

For example,
dataset_path= arg must be accepted during init
copy construction must be supported: Dataset(dataset_instance) returns a copy

"""

from inspect import signature
from collections.abc import Iterable
from pyradigm import (ClassificationDataset as ClfDataset,
                      RegressionDataset as RegrDataset)
from pyradigm.utils import make_random_ClfDataset
from pyradigm.base import is_iterable_but_not_str, PyradigmException, \
    ConstantValuesException, InfiniteOrNaNValuesException, EmptyFeatureSetException
from pytest import raises, warns
import numpy as np
import random
import os
from os.path import join as pjoin, realpath, dirname
from warnings import warn

class_list = (ClfDataset, RegrDataset)

constructor_must_offer_param_list = ('dataset_path',
                                     'data', 'targets', 'dtype',
                                     'description', 'feature_names')

cur_dir = dirname(realpath(__file__))
out_dir = pjoin(cur_dir, 'tmp')
os.makedirs(out_dir, exist_ok=True)
out_file = pjoin(out_dir, 'random_example_dataset.pkl')

ds = make_random_ClfDataset()


def _not_equal(array_one, array_two):

    if isinstance(array_one, dict):
        not_equal = False
        for key, val in array_one.items():
            if _not_equal(val, array_two[key]):
                not_equal = True
                break
    elif is_iterable_but_not_str(array_one):
        not_equal = False
        try:
            for v1, v2 in zip(array_one, array_two):
                if v1 != v2:
                    not_equal = True
                    break
        except:
            print('not_equal type of array_one {},'
                  'array_two {}'.format(type(array_one), type(array_two)))
            raise
    else:
        not_equal = array_one != array_two

    if not isinstance(not_equal, bool):
        not_equal = any(not_equal)

    return not_equal



def test_constructor_must_offer_params():
    for cls in class_list:
        cls_sign = signature(cls)
        for param in constructor_must_offer_param_list:
            if not param in cls_sign.parameters:
                raise SyntaxError('Class {} does not offer {} as an argument '
                                  'during init!'.format(cls, param))


def test_attributes():
    """Creation, access and properties"""

    id_list = np.array(ds.samplet_ids)

    # ensuring strings can't be added to float attributes
    ds.add_attr('age', id_list[0], 43)
    for mismatched_type in ['43', 2 + 3j]:
        with raises(TypeError):
            ds.add_attr('age', id_list[2], mismatched_type)

    # ensuring floats can't be added to string attributes
    ds.add_attr('gender', id_list[3], 'female')
    for mismatched_type in [43, 2 + 3j]:
        with raises(TypeError):
            ds.add_attr('gender', id_list[4], mismatched_type)

    # adding to multiple samplets at a time
    # this should work
    ds.add_attr('gender',
                id_list[:3],
                ('female', 'male', 'male'))
    # but not this:
    with raises(ValueError):
        ds.add_attr('gender',
                    id_list[:3],
                    ('female', 'male',))

    # dataset attributes
    try:
        ds.add_dataset_attr('version', 2.0)
        # ds.add_dataset_attr('params', ['foo', 'bar', 20, 12, '/work/path'])
        # arbitrary values are causing problems with np.not_equal checks
        #   using simple values for now
        ds.add_dataset_attr('params', ['foo', 'bar', '/work/path'])
    except:
        raise AttributeError('Unable to add dataset attributes')

    # retrieval
    random_ids = id_list[random.sample(range(50), 5)]
    values_set = np.random.rand(5)
    ds.add_attr('random_float', random_ids, values_set)
    retrieved = ds.get_attr('random_float', random_ids)
    if not all(np.isclose(retrieved, values_set)):
        raise ValueError('Retrieved attribute values do not match the originals!')

    # ensuring ds.get_subset() returns subset for each attribute
    for index, id_ in enumerate(id_list):
        ds.add_attr('sid', id_, id_)
        ds.add_attr('index', id_, index)

    id_subset = id_list[:3]
    sub = ds.get_subset(id_subset)
    if set(sub.attr['sid']) != set(id_subset) or \
            set(sub.attr['index']) != set(id_subset):
        raise ValueError('attrs are not being propagated during .get_subset()')

    with raises(KeyError):
        ds.get_attr('non_existing_attr')

    with raises(KeyError):
        # existing but not all of them are set
        ds.get_attr('random_float', ds.samplet_ids)

    with warns(UserWarning):
        ds.del_attr('non_existing_attr')

    try:
        ds.del_attr('random_float')
    except:
        raise AttributeError('Attr deletion failed!')


def test_save_load():

    ds.save(file_path=out_file)
    reloaded_ds = ds.__class__(dataset_path=out_file)

    if ds != reloaded_ds:
        raise IOError('Error in save/load implementation!')

    must_have_attr = ('_data', '_targets',
                      '_dtype', '_target_type', '_description',
                      '_num_features', '_feature_names',
                      '_attr', '_attr_dtype', '_dataset_attr')

    for attr in must_have_attr:
        if not hasattr(reloaded_ds, attr):
            raise AttributeError('Attribute {} missing after reload from disk'
                                 ''.format(attr))

        orig_val = getattr(ds, attr)
        reloaded = getattr(reloaded_ds, attr)

        not_equal = False
        try:
            if isinstance(orig_val, dict):
                for key, val in orig_val.items():
                    if _not_equal(val, reloaded[key]):
                        warn('Values differ for attr {} in samplet {}'
                                         ' when reloaded from disk'.format(attr, key))
                        not_equal = True
                        break
            elif is_iterable_but_not_str(orig_val):
                for aa, bb in zip(orig_val, reloaded):
                    if aa != bb:
                        not_equal= True
                        break
                # not_equal = any(np.not_equal(orig_val, reloaded))
            elif np.issubdtype(type(orig_val), np.generic):
                not_equal = reloaded != orig_val
            else:
                raise TypeError('Unrecognized type {} for attr {}'
                                ''.format(type(orig_val), attr))
        except:
            raise

        if not isinstance(not_equal, bool):
            not_equal = any(not_equal)

        if not_equal:
            raise AttributeError('Attribute {} differs between the reloaded'
                                 ' and the original datasets'.format(attr))


def test_nan_inf_values():

    target_val = 3
    for cls_type in (RegrDataset, ClfDataset):

        cds_clean = cls_type(allow_nan_inf=False)
        for invalid_value in [np.NaN, np.Inf]:
            with raises(InfiniteOrNaNValuesException):
                cds_clean.add_samplet('a', [1, invalid_value, 3], target_val)

        cds_dirty = cls_type(allow_nan_inf=True)
        for sid, valid_value in zip(('a', 'b'), [np.NaN, np.Inf]):
            try:
                cds_dirty.add_samplet(sid, [1, valid_value, 3], target_val)
            except:
                raise


def test_sanity_checks():
    """Ensure that sanity checks are performed, and as expected."""


    ### -------------- as you add them to dataset --------------
    with raises(EmptyFeatureSetException):
        ds.add_samplet('empty_features', [], 'target')

    ### -------------- as you save them to disk --------------

    ds.add_samplet('all_zeros', np.zeros((ds.num_features,1)), 'target')
    with raises(ConstantValuesException):
        ds.save(out_file)

    ds.del_samplet('all_zeros')

    # checking for random constant value!
    const_value = np.random.randint(10, 100)
    const_feat_set =  np.full((ds.num_features, 1), const_value)
    ds.add_samplet('all_constant', const_feat_set, 'target')
    with raises(ConstantValuesException):
        ds.save(out_file)


    # now checking for constants across samplets
    #   this is easily achieved by adding different samplets with same features
    #   such a bug is possible, when user made a mistake querying
    #   the right files for the right samplet ID
    const_ds = ClfDataset()
    rand_feat_same_across_samplets = np.random.randn(10)
    for index in range(np.random.randint(10, 100)):
        const_ds.add_samplet(str(index), rand_feat_same_across_samplets, index)

    with raises(ConstantValuesException):
        const_ds.save(out_file)

test_attributes()
# test_save_load()
# test_sanity_checks()
test_nan_inf_values()