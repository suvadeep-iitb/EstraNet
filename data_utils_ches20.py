import numpy as np
import tensorflow as tf
import h5py

import os, sys


def sbox_layer(x):
    y1 = (x[0] & x[1]) ^ x[2]
    y0 = (x[3] & x[0]) ^ x[1]
    y3 = (y1 & x[3]) ^ x[0]
    y2 = (y0 & y1) ^ x[3]
    return np.stack([y0, y1, y2, y3], axis=1)


class Dataset:
    def __init__(self, data_path, split, input_length, data_desync=0):
        self.data_path = data_path
        self.split = split
        self.input_length = input_length
        self.data_desync = data_desync

        data = np.load(data_path)
        self.traces = data['traces']
        self.nonces = data['nonces']
        self.umsk_keys = data['umsk_keys']

        shift = 17
        self.nonces = self.nonces >> shift
        self.umsk_keys = self.umsk_keys >> shift
        if len(self.umsk_keys.shape) == 1:
            self.umsk_keys = np.reshape(self.umsk_keys, [1, -1])
        
        sbox_in = np.bitwise_xor(self.nonces, self.umsk_keys)
        sbox_in = sbox_in.T
        sbox_out = sbox_layer(sbox_in)
        self.labels = (sbox_out & 0x1)
        self.labels = self.labels.astype(np.float32)

        assert (self.input_length + self.data_desync) <= self.traces.shape[1] 
        self.traces = self.traces[:, :(self.input_length+self.data_desync)]

        self.num_samples = self.traces.shape[0]

        max_split_size = 2000000000//self.input_length
        split_idx = list(range(max_split_size, self.num_samples, max_split_size))
        self.traces = np.split(self.traces, split_idx, axis=0)
        self.labels = np.split(self.labels, split_idx, axis=0)

    
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
                              .batch(batch_size//4) \
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

