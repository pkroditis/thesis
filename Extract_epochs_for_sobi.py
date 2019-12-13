#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mne
import numpy as np
from mne.io import concatenate_raws
from braindecode.datautil.signal_target import SignalAndTarget

# First 80 subjects as train
physionet_paths = [ mne.datasets.eegbci.load_data(sub_id,[4,8,12,]) for sub_id in range(1,110) if sub_id not in [88,89,90,91,92,93,94,100,101,102,103,104,105,106,107,108]]
physionet_paths = np.concatenate(physionet_paths)
parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
         for path in physionet_paths]

raw = concatenate_raws(parts)

picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Find the events in this dataset
events, _ = mne.events_from_annotations(raw)

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epoched = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=1, tmax=4.1, proj=False, picks=picks,
                baseline=None, preload=True)



train_X = (epoched.get_data() * 1e6).astype(np.float32)
train_y = (epoched.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1


del  epoched, events, picks, raw, parts, physionet_paths



outpath_signals = "..\\test_signals\\"
outpath_labels = '..\\test_labels\\'
from scipy.io import savemat

for i in range(len(train_X)):
    savemat(outpath_signals + str(i)+ '_signal' + '.mat', {'record':train_X[i]})

matfile = outpath_labels + 'label.mat'
savemat(matfile, mdict = {'out': train_y}, oned_as = 'row') 

