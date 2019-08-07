import numpy as np
from pyradigm import ClassificationDataset, RegressionDataset
from pyradigm.pyradigm import MLDataset

feat_generator = np.random.randn


def make_random_dataset(max_num_classes=20,
                        min_class_size=20,
                        max_class_size=50,
                        max_dim=100,
                        stratified=True,
                        class_type=ClassificationDataset):
    "Generates a random Dataset for use in testing."

    smallest = min(min_class_size, max_class_size)
    max_class_size = max(min_class_size, max_class_size)
    largest = max(50, max_class_size)
    largest = max(smallest + 3, largest)

    if max_num_classes != 2:
        num_classes = np.random.randint(2, max_num_classes, 1)
    else:
        num_classes = 2

    if type(num_classes) == np.ndarray:
        num_classes = num_classes[0]
    if not stratified:
        class_sizes = np.random.random_integers(smallest, largest, num_classes)
    else:
        class_sizes = np.repeat(np.random.randint(smallest, largest), num_classes)

    num_features = np.random.randint(min(3, max_dim), max(3, max_dim), 1)[0]
    # feat_names = [ str(x) for x in range(num_features)]

    class_ids = list()
    labels = list()
    for cl in range(num_classes):
        if issubclass(class_type, RegressionDataset):
            class_ids.append(cl)
        else:
            class_ids.append('class-{}'.format(cl))
        labels.append(int(cl))

    ds = class_type()
    for cc, class_ in enumerate(class_ids):
        subids = ['s{}-c{}'.format(ix, cc) for ix in range(class_sizes[cc])]
        for sid in subids:
            if isinstance(ds, MLDataset):
                ds.add_sample(sid, feat_generator(num_features), int(cc), class_)
            else:
                ds.add_samplet(sid, feat_generator(num_features), class_)

    return ds


def make_random_ClfDataset(max_num_classes=20,
                           min_class_size=20,
                           max_class_size=50,
                           max_dim=100,
                           stratified=True):
    "Generates a random ClassificationDataset for use in testing."

    return make_random_dataset(max_num_classes=max_num_classes,
                               min_class_size=min_class_size,
                               max_class_size=max_class_size,
                               max_dim=max_dim,
                               stratified=stratified,
                               class_type=ClassificationDataset)


def make_random_RegrDataset(max_num_classes=20,
                            min_class_size=20,
                            max_class_size=50,
                            max_dim=100,
                            stratified=True):
    "Generates a random ClassificationDataset for use in testing."

    return make_random_dataset(max_num_classes=max_num_classes,
                               min_class_size=min_class_size,
                               max_class_size=max_class_size,
                               max_dim=max_dim,
                               stratified=stratified,
                               class_type=RegressionDataset)


def make_random_MLdataset(max_num_classes=20,
                          min_class_size=20,
                          max_class_size=50,
                          max_dim=100,
                          stratified=True):
    "Generates a random MLDataset for use in testing."

    return make_random_dataset(max_num_classes=max_num_classes,
                               min_class_size=min_class_size,
                               max_class_size=max_class_size,
                               max_dim=max_dim,
                               stratified=stratified,
                               class_type=MLDataset)
