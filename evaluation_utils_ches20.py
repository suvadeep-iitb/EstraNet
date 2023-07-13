import numpy as np
from scipy import special
import h5py
import tensorflow as tf

import os, sys


def sbox_layer(x):
    y1 = (x[0] & x[1]) ^ x[2]
    y0 = (x[3] & x[0]) ^ x[1]
    y3 = (y1 & x[3]) ^ x[0]
    y2 = (y0 & y1) ^ x[3]
    return np.stack([y0, y1, y2, y3], axis=1)


def shuffle_all(predictions, nonces):
    perm = np.random.permutation(predictions.shape[0])
    predictions = predictions[perm]
    nonces = nonces[perm]

    return predictions, nonces


def get_log_prob(predictions, plaintext):
    predictions = np.squeeze(predictions)
    n_classes = predictions.shape[0]
    keys = np.arange(n_classes, dtype=int)
    x_xor_k = np.bitwise_xor(keys, plaintext)
    z = np.take(sbox, x_xor_k)
    log_prob = np.take(predictions, z)

    return log_prob


def gen_key_bits():
    values = np.arange(16, dtype=np.uint8).reshape(-1, 1)
    key_bits = np.unpackbits(values, axis=1)[:, -4:]
    for k in range(16):
        t = key_bits[k, 0]
        key_bits[k, 0] = key_bits[k, 3]
        key_bits[k, 3] = t
        t = key_bits[k, 1]
        key_bits[k, 1] = key_bits[k, 2]
        key_bits[k, 2] = t
    return key_bits


def compute_key_rank(predictions, nonces, keys):
    n_samples, n_classes = predictions.shape
    nonces = (nonces[:n_samples] & 0x1)
    keys = np.squeeze(keys)

    predictions, nonces = shuffle_all(predictions, nonces)

    def get_corr_key(keys):
        corr_key = ((keys[0] & 0x1) << 0)
        corr_key |= ((keys[1] & 0x1) << 1)
        corr_key |= ((keys[2] & 0x1) << 2)
        corr_key |= ((keys[3] & 0x1) << 3)
        return corr_key
    corr_key = get_corr_key(keys)

    key_bits = gen_key_bits()
    n_keys = key_bits.shape[0]

    neg_log_prob = np.zeros((n_samples, n_keys))
    for k in range(n_keys):
        key_rep = np.reshape(key_bits[k, :], [1, -1])
        sbox_in = (nonces ^ key_rep).T
        sbox_out = (sbox_layer(sbox_in) & 0x1)
        sbox_out = sbox_out.astype(np.float32)
        scores = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(sbox_out, predictions),
            axis = 1
        ).numpy()
        neg_log_prob[:, k] = scores

    cum_neg_log_prob = np.zeros((n_samples, n_keys))
    last_neg_log_prob = np.zeros((1, n_keys))
    for i in range(n_samples):
        last_neg_log_prob += neg_log_prob[i]
        cum_neg_log_prob[i, :] = last_neg_log_prob

    sorted_keys = np.argsort(cum_neg_log_prob, axis=1)
    key_ranks = np.zeros((n_samples), dtype=int) - 1
    for i in range(n_samples):
        for j in range(n_keys):
            if sorted_keys[i, j] == corr_key:
                key_ranks[i] = j
                break

    for i in range(n_samples):
        assert key_ranks[i] >= 0, "Assertion failed at index %s" % i
        
    return key_ranks


if __name__ == '__main__':
    data_path = sys.argv[1]

    data = h5py.File(data_path, 'r')

    for i in range(10):
        label = data['Profiling_traces']['labels'][i]
        ptest = data['Profiling_traces']['metadata'][i]['plaintext'][2]
        key = data['Profiling_traces']['metadata'][i]['key'][2]

        print(str(label)+'/'+str(sbox[np.bitwise_xor(ptest, key)])+'/'+str(key))

