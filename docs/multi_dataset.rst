MultiDataset
--------------


MultiDataset is a container data structure to hold and manage multiple MLDataset instances. pyradigm also offers two "meta" data structures that can hold multiple pyradigm MLDatasets in a more convenient and efficient way. The main purpose of these containers is to automatically perform checks for compatibility of a collection of Datasets, such as

    - ensuring same set of samplet IDs exist in all tables
    - they all link to same set of targets and attributes etc)

Such compatibility checks are often necessary when performing comparisons in machine learning e.g. cross-validation (CV).


Key uses:

    - Uniform processing individual MLDatasets e.g. querying same set of IDs
    - ensuring correspondence across multiple datasets in cross-validation
    - reduce redundancy, improving integrity in linked tables as well as saving space and time


A schematic illustrating the function of the ``MultiDataset`` is shown below, wherein a single ``MultiDataset`` links and encapsulates 4 data tables ``X1`` to ``X4`` with the same set of targets ``y`` and attributes ``A``:

.. image:: flyer_multimodal.png
    :height: 500