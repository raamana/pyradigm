from sys import version_info

if version_info.major > 2:
    from pyradigm import pyradigm
else:
    raise NotImplementedError('Python 3 or higher is required to run pyradigm. '
                              'Please upgrade.')
del version_info


def main():
    "Entry point."

    pyradigm.cli_run()


if __name__ == '__main__':
    main()
