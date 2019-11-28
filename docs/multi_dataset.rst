MultiDataset
--------------


pyradigm also offers two "meta" data structures that can hold multiple pyradigm MLDatasets in a more convenient and efficient way. The main purpose of these containers is to automatically perform checks for compatibility of a collection of Datasets (same set of samplet IDs, targets and attributes etc) which is often necessary when performing comparisons in cross-validation.