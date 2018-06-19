import os
import numpy as np
import datetime
import h5py
from copy import copy

#local
import data

"""
Runtime_data_handler
"""

class Runtime_data_handler(object):

    def __init__(self,h5_path,batch_size=64,seq_len=64,use_labels=True):

        self.h5_path = h5_path
        self.batch_size = batch_size
        self.use_labels = use_labels
        self.output_seq_len = seq_len

        with h5py.File(self.h5_path, 'r') as h5:
            self.h5_num_rows = h5['tags'].shape[0]
            self.h5_seq_len = h5['tags'].shape[1]

        assert self.output_seq_len <= self.h5_seq_len

        self.num_batches_per_epoch = self.h5_num_rows // self.batch_size # skipping the last, non-complete, batch

        self.h5_curr_batch_pointer = - self.batch_size

    def epoch_start(self,start_batch_id = 0):

        #TODO - shuffle !
        self.h5_curr_batch_pointer = start_batch_id * self.batch_size

    def get_num_batches_per_epoch(self):
        return self.num_batches_per_epoch

    def get_batch(self):

        with h5py.File(self.h5_path, 'r') as h5:
            tags = np.array(h5['tags'][self.h5_curr_batch_pointer:self.h5_curr_batch_pointer+self.batch_size])
            labels = np.array(h5['labels'][self.h5_curr_batch_pointer:self.h5_curr_batch_pointer+self.batch_size])

        print("batch [%0d : %0d]"%(self.h5_curr_batch_pointer,self.h5_curr_batch_pointer+self.batch_size)) # for debug

        # process
        tags = self.cut_pad_tags(tags)

        # increment batch pointer
        self.h5_curr_batch_pointer += self.batch_size

        if self.use_labels:
            return tags, labels
        else:
            return tags

    def cut_pad_tags(self,tags):
        _tags = copy(tags)[:,:self.output_seq_len]
        start_tags = data.char_dict['START'] * np.ones([self.batch_size,1],dtype=np.uint8)
        end_tags = data.char_dict['END'] * np.ones([self.batch_size, 1], dtype=np.uint8)
        return np.concatenate((start_tags,_tags,end_tags),axis=1)


