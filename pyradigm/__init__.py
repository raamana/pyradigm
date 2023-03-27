__all__ = ['ClassificationDataset', 'RegressionDataset',
           'MultiDatasetClassify', 'MultiDatasetRegress',
           'BaseDataset', 'check_compatibility', 'MLDataset', 'cli_run']

from sys import version_info

if version_info.major >= 3:
    from pyradigm.base import BaseDataset
    from pyradigm.classify import ClassificationDataset
    from pyradigm.regress import RegressionDataset
    from pyradigm.pyradigm import MLDataset, cli_run
    from pyradigm.multiple import MultiDatasetClassify, MultiDatasetRegress
    from pyradigm.utils import check_compatibility
else:
    raise NotImplementedError('pyradigm supports only Python 3 or higher! '
                              'Upgrade to Python 3+ is recommended.')

try:
    from ._version import __version__
except ImportError:
    __version__ = "0+unknown"
