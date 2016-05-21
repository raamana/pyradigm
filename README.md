# mldataset
A Python class to encapsulate a machine learning dataset based on dictionaries.  This class is greatly suited for neuroimaging applications (or any other domain), where each sample needs to be uniquely identified with a subject ID (or something similar). Key-level correspondence across data, labels (1 or 2), classnames ('healthy', 'disease') and the related helps maintain data integrity, in addition to enabling traceback to original sources from where the features have been originally derived.

Please refer to the notebook for an illustration of the usage: [MLDataset Example](MLDatasetExample.ipynb)
