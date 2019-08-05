
"""

Tests to ensure certain behaviours among all child classes of BaseDataset

For example,
dataset_path= arg must be accepted during init
copy construction must be supported: Dataset(dataset_instance) returns a copy

"""


from pyradigm import ClassificationDataset as ClfDataset, MultiDataset, \
    RegressionDataset as RegrDataset
from pyradigm.utils import make_random_ClfDataset
from inspect import signature
from pytest import raises

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
    id_list = ds.samplet_ids

    # ensuring strings can't be added to float attributes
    ds.add_attr('age', id_list[0], 43)
    for mismatched_type in ['43', 2+3j ]:
        with raises(TypeError):
            ds.add_attr('age', id_list[2], mismatched_type)

    # ensuring floats can't be added to string attributes
    ds.add_attr('gender', id_list[3], 'female')
    for mismatched_type in [43, 2+3j ]:
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
                    ('female', 'male', ))

    try:
        ds.add_dataset_attr('version', 2.0)
        ds.add_dataset_attr('params', ['foo', 'bar', 20, 12, '/work/path'])
    except:
        raise AttributeError('Unable to add dataset attributes')


test_attributes()