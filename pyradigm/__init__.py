__all__ = [ 'pyradigm', 'MLDataset', 'cli_run' ]

from sys import version_info

if version_info.major==2 and version_info.minor==7:
    from pyradigm import MLDataset, cli_run
elif version_info.major > 2:
    from pyradigm.pyradigm import MLDataset, cli_run
else:
    raise NotImplementedError('pyradigm supports only 2.7 or 3+. Upgrate to Python 3+ is recommended.')



from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
