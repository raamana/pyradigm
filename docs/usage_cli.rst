-------------------------
Command line interface
-------------------------

In addition to the recommended use of pyradigm via the :doc:`API`, there are use cases for command line usage.

For example, it's convenient to quickly display info about pyradigm's MLDatasets already saved to disk, without you having to write some code. Sometimes, you may even want to simply merge two or more such MLDatasets (combine disjoint feature sets, or add more classes to the dataset). This interface would help you with such use cases.


.. argparse::
   :filename: pyradigm/pyradigm.py
   :func: get_parser
   :prog: pyradigm
   :nodefault:
   :nodefaultconst:

