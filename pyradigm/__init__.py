__all__ = [ 'pyradigm', 'MLDataset', 'MultiDataset', 'cli_run',
            'check_compatibility_BaseDataset']

from sys import version_info

if version_info.major==2:
    raise NotImplementedError()
elif version_info.major > 2:
    from pyradigm.base import BaseDataset, check_compatibility_BaseDataset
    from pyradigm.classify import ClassificationDataset
    from pyradigm.pyradigm import MLDataset, cli_run
    from pyradigm.multiple import MultiDataset
else:
    raise NotImplementedError('pyradigm supports only 2.7 or 3+. '
                              'Upgrade to Python 3+ is recommended.')



from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
