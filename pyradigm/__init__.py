
from sys import version_info

if version_info.major==2 and version_info.minor==7:
    from pyradigm import MLDataset
elif version_info.major > 2:
    from pyradigm.pyradigm import MLDataset
else:
    raise NotImplementedError('pyradigm supports only 2.7.13 or 3+. Upgrate to Python 3+ is recommended.')

__all__ = [ 'pyradigm', 'MLDataset' ]

