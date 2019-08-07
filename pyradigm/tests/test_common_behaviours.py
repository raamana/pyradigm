"""

Tests to ensure certain behaviours among all child classes of BaseDataset

For example,
dataset_path= arg must be accepted during init
copy construction must be supported: Dataset(dataset_instance) returns a copy

"""

from inspect import signature

from pyradigm import (ClassificationDataset as ClfDataset,
                      RegressionDataset as RegrDataset)
from pyradigm.utils import make_random_ClfDataset
from pytest import raises, warns
import numpy as np
import random

class_list = (ClfDataset, RegrDataset)

constructor_must_offer_param_list = ('dataset_path',
                                     'data', 'targets', 'dtype',
                                     'description', 'feature_names')


def test_constructor_must_offer_params():
    for cls in class_list:
        cls_sign = signature(cls)
        for param in constructor_must_offer_param_list:
            if not param in cls_sign.parameters:
                raise SyntaxError('Class {} does not offer {} as an argument '
                                  'during init!'.format(cls, param))


def test_attributes():
    """Creation, access and properties"""

    ds = make_random_ClfDataset()
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
        ds.add_dataset_attr('params', ['foo', 'bar', 20, 12, '/work/path'])
    except:
        raise AttributeError('Unable to add dataset attributes')

    # retrieval
    random_ids = id_list[random.sample(range(50), 5)]
    values_set = np.random.rand(5)
    ds.add_attr('random_float', random_ids, values_set)
    retrieved = ds.get_attr('random_float', random_ids)
    if not all(np.isclose(retrieved, values_set)):
        raise ValueError('Retrieved attribute values do not match the originals!')

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


test_attributes()
