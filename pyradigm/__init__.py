
__all__ = [ 'ClassificationDataset', 'RegressionDataset', 'MultiDataset',
            'BaseDataset', 'check_compatibility_BaseDataset',
            'pyradigm', 'MLDataset', 'cli_run']

from sys import version_info


if version_info.major >= 3:
    from pyradigm.base import BaseDataset, check_compatibility_BaseDataset
    from pyradigm.classify import ClassificationDataset
    from pyradigm.regress import RegressionDataset
    from pyradigm.pyradigm import MLDataset, cli_run
    from pyradigm.multiple import MultiDataset
else:
    raise NotImplementedError('pyradigm supports only Python 3 or higher! '
                              'Upgrade to Python 3+ is recommended.')


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
