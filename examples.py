
import sys, os
import numpy as np
from mldataset import MLDataset
import cPickle as pickle

def read_thickness(path):
    """Dummy function to minic a data reader."""

    # in your actural routine, this might be:
    #   pysurfer.read_thickness(path).values()
    return np.random.random(8)


def get_features(work_dir, subj_id):
    """Returns the whole brain cortical thickness for a given subject ID."""

    # extension to identify the data file; this could be .curv, anything else you choose
    ext_thickness = '.thickness'

    thickness = dict()
    for hemi in ['lh', 'rh']:
        path_thickness = os.path.join(work_dir, subj_id, hemi + ext_thickness)
        thickness[hemi] = read_thickness(path_thickness)

    # concatenating them to build a whole brain feature set
    thickness_wb = np.concatenate([thickness['lh'], thickness['rh']])

    return thickness_wb


work_dir = '/project/ADNI/FreesurferThickness_v4p3'
class_set = ['Ctrl', 'Alzr']

dataset = MLDataset()
dataset.description = 'ADNI1 baseline: cortical thickness features from Freesurfer v4.3, QCed.'

for class_index, class_id in enumerate(class_set):
    print('Working on class {:>5}'.format(class_id))

    target_list_path = os.path.join(work_dir,'scripts','test_sample.{}'.format(class_id))
    with open(target_list_path,'r') as tf:
        target_list = tf.readlines()
        target_list = [sub.strip() for sub in target_list]

    for subj_id in target_list:
        print('\t reading subject {:>15}'.format(subj_id))
        thickness_wb = get_features(work_dir, subj_id)

        # adding the sample to the dataset
        dataset.add_sample(subj_id, thickness_wb, class_index, class_id)


print 'Constructed dataset:'
print dataset

out_file = os.path.join(work_dir,'test.pkl')

# saving the dataset to disk
try:
    path = os.path.abspath(out_file)
    with open(path, 'wb') as df:
        pickle.dump(dataset, df)
    print('saved.')
except:
    raise

# reloading it
try:
    path = os.path.abspath(out_file)
    with open(path, 'rb') as df:
        reloaded = pickle.load(df)

except:
    raise

print '\n\n reloaded dataset:'
print reloaded


