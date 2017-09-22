
from sys import version_info

if version_info.major==2 and version_info.minor==7:
    import pyradigm
elif version_info.major > 2:
    from pyradigm import pyradigm
else:
    raise NotImplementedError('pyradigm supports only 2.7.13 or 3+. Upgrate to Python 3+ is recommended.')

def main():
    "Entry point."

    pyradigm.cli_run()

if __name__ == '__main__':
    main()
