import numpy as np
import tensorflow as tf
import h5py

import os, sys

class Dataset:
    def __init__(self, data_path, split, input_length, data_desync=0):
        self.data_path = data_path
        self.split = split
        self.input_length = input_length
        self.data_desync = data_desync

        corpus = h5py.File(data_path, 'r')
        if split == 'train':
            split_key = 'Profiling_traces'
        elif split == 'test':
            split_key = 'Attack_traces'

        self.traces = corpus[split_key]['traces'][:, :(self.input_length+self.data_desync)]
        self.labels = np.reshape(corpus[split_key]['labels'][()], [-1, 1])
        self.labels = self.labels.astype(np.int64)
        self.num_samples = self.traces.shape[0]

        #assert (self.input_length + self.data_desync) <= self.traces.shape[1] 
        #self.traces = self.traces[:, :(self.input_length+self.data_desync)]

        max_split_size = 2000000000//self.input_length
        split_idx = list(range(max_split_size, self.num_samples, max_split_size))
        self.traces = np.split(self.traces, split_idx, axis=0)
        self.labels = np.split(self.labels, split_idx, axis=0)

        #self.traces = self.traces.astype(np.float32)

        self.plaintexts = self.GetPlaintexts(corpus[split_key]['metadata'])
        self.masks = self.GetMasks(corpus[split_key]['metadata'])
        self.keys = self.GetKeys(corpus[split_key]['metadata'])

    
    def GetPlaintexts(self, metadata):
        plaintexts = []
        for i in range(len(metadata)):
            plaintexts.append(metadata[i]['plaintext'][2])
        return np.array(plaintexts)


    def GetKeys(self, metadata):
        keys = []
        for i in range(len(metadata)):
            keys.append(metadata[i]['key'][2])
        return np.array(keys)


    def GetMasks(self, metadata):
        masks = []
        for i in range(len(metadata)):
            masks.append(np.array(metadata[i]['masks']))
        masks = np.stack(masks, axis=0)
        return masks


    def GetTFRecords(self, batch_size, training=False):
        dataset = tf.data.Dataset.from_tensor_slices((self.traces[0], self.labels[0]))
        for traces, labels in zip(self.traces[1:], self.labels[1:]):
            temp_dataset = tf.data.Dataset.from_tensor_slices((traces, labels))
            dataset.concatenate(temp_dataset)

        def shift(x, max_desync):
            ds = tf.random.uniform([1], 0, max_desync+1, tf.dtypes.int32)
            ds = tf.concat([[0], ds], 0)
            x = tf.slice(x, ds, [-1, self.input_length])
            return x

        if training == True:
            if self.input_length < self.traces[0].shape[1]:
                return dataset.repeat() \
                              .shuffle(self.num_samples) \
                              .batch(batch_size//2) \
                              .map(lambda x, y: (shift(x, self.data_desync), y)) \
                              .unbatch() \
                              .batch(batch_size, drop_remainder=True) \
                              .map(lambda x, y: (tf.cast(x, tf.float32), y)) \
                              .prefetch(10)
            else:
                return dataset.repeat() \
                              .shuffle(self.num_samples) \
                              .batch(batch_size, drop_remainder=True) \
                              .map(lambda x, y: (tf.cast(x, tf.float32), y)) \
                              .prefetch(10)

        else:
            if self.input_length < self.traces[0].shape[1]:
                return dataset.batch(batch_size, drop_remainder=True) \
                              .map(lambda x, y: (shift(x, 0), y)) \
                              .map(lambda x, y: (tf.cast(x, tf.float32), y)) \
                              .prefetch(10)
            else:
                return dataset.batch(batch_size, drop_remainder=True) \
                              .map(lambda x, y: (tf.cast(x, tf.float32), y)) \
                              .prefetch(10)


    def GetDataset(self):
        return self.traces, self.labels

    
if __name__ == '__main__':
    data_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    split = sys.argv[3]

    dataset = Dataset(data_path, split, 5)

    print("traces    : "+str(dataset.traces.shape))
    print("labels    : "+str(dataset.labels.shape))
    print("plaintext : "+str(dataset.plaintexts.shape))
    print("keys      : "+str(dataset.keys.shape))
    print("traces ty : "+str(dataset.traces.dtype))
    print("")
    print("")

    tfrecords = dataset.GetTFRecords(batch_size, training=True)
    iterator = iter(tfrecords)
    for i in range(1):
        tr, lbl = iterator.get_next()
        print(str(tr.shape)+' '+str(lbl.shape))
        print(str(tr.dtype)+' '+str(lbl.dtype))
        print(str(tr[:, :10]))
        print(str(lbl[:, :]))
        print("")

