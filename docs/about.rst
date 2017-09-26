--------------------------------------------------------------------------------------------------
About
--------------------------------------------------------------------------------------------------

Background
----------

A common problem for machine learning developers is keeping track of the source of the features extracted, and to ensure integrity of the dataset (e.g. not getting data mixed up from different subjects and/or classes). This is incredibly hard as the number of projects grow, or personnel changes are frequent. These aspects can break the chain of hyper-local info about various datasets, such as where did the original data come from, how was it processed or quality controlled, how was it put together, by who and what does some columns in the table mean etc. This package aims to provide a Python data structure to encapsulate a machine learning dataset with key info greatly suited for neuroimaging applications (or similar domains), where each sample needs to be uniquely identified with a subject ID (or something similar). Key-level correspondence across data, labels (e.g. 1 or 2), classnames (e.g. 'healthy', 'disease') and the related attributes helps maintain data integrity. Moreover, attributes like free-text description help annotate all the important information. The class methods offer the ability to arbitrarilty combine and subset datasets, while automatically updating their description reduces burden to keep track of the original source of features.


Check the :doc:`usage` and :doc:`API` pages, and let me know your comments.


Context
-------

For users of `Pandas <http://pandas.pydata.org/>`_, some of the elements in `pyradigm`'s API/interface may look familiar. However, the aim of this data structure is not to offer an alternative to pandas, but to ease the machine learning workflow for neuroscientists by 

 1) offering several well-knit methods and useful attributes specifically geared towards neuroscience research, 
 2) aiming to offer utilities that combines multiple or advanced patterns of routine dataset handling and 
 3) using a more accessible language (compared to hard to read pandas docs aimed at econometric audience) to better cater to neuroscience developers (esp. the novice).


Thanks for checking out. Your feedback will be appreciated.
