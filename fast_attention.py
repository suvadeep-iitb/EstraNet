import math
import numpy as np
import tensorflow as tf


def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def gen_projection_matrix(m, d, seed=0):
    n_block = m // d
    block_list = []
    cur_seed = seed
    for _ in range(n_block):
        block = tf.random.normal((d, d), seed=cur_seed)
        q, _ = tf.linalg.qr(block)
        q = tf.transpose(q)
        block_list.append(q)
        cur_seed += 1
    rem_rows = m - n_block * d
    if rem_rows > 0:
        block = tf.random.normal((d, d), seed=cur_seed)
        q, _ = tf.linalg.qr(block)
        q = tf.transpose(q)
        block_list.append(q[0:rem_rows])
    proj_matrix = tf.experimental.numpy.vstack(block_list)
    cur_seed += 1

    multiplier = tf.norm(tf.random.normal((m, d), seed=cur_seed), axis=1)

    return tf.linalg.matmul(tf.linalg.diag(multiplier), proj_matrix)


def positive_kernel_transformation(data,
                                   is_query,
                                   projection_matrix=None,
                                   numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / (tf.dtypes.cast(data.shape[-1], tf.float32) ** 0.25)
    data= data_normalizer * data
    ratio = 1.0 / (tf.dtypes.cast(projection_matrix.shape[0], tf.float32) ** 0.5)
    data_dash = tf.einsum("blhd,md->blhm", data, projection_matrix)
    diag_data = tf.square(data)
    diag_data = tf.reduce_sum(diag_data, axis=-1, keepdims=True)
    last_dims_t = (len(data_dash.shape) - 1,)
    attention_dims_t = (len(data_dash.shape) - 3,)
    if is_query:
        data_dash = ratio * (
            tf.math.exp(data_dash - diag_data - tf.math.reduce_max(
                data_dash, axis=-1, keepdims=True)) + numerical_stabilizer)
    else:
        data_dash = ratio * (
            tf.math.exp(data_dash - diag_data - tf.math.reduce_max(
                data_dash, axis=[-3, -1], keepdims=True)) + numerical_stabilizer)

    return data_dash


def fourier_kernel_transformation(data, projection_matrix):
    data_normalizer = 1.0 / (tf.dtypes.cast(data.shape[-1], tf.float32) ** 0.25)
    data = data_normalizer * data
    ratio = 1.0 / (tf.cast(projection_matrix.shape[0], tf.float32) ** 0.5)
    data_dash = tf.einsum("blhd,md->blhm", data, projection_matrix)
    data_sin = ratio * tf.math.sin(data_dash)
    data_cos = ratio * tf.math.cos(data_dash)

    return tf.concat([data_sin, data_cos], axis=-1)


def attention_numerator(qs, ks, vs):
    kvs = tf.einsum("lbhm,lbhd->bhmd", ks, vs)
    return tf.einsum("lbhm,bhmd->lbhd", qs, kvs)


def attention_denominator(qs, ks):
    all_ones = tf.ones([ks.shape[0]])
    ks_sum = tf.einsum("lbhm,l->bhm", ks, all_ones)
    return tf.einsum("lbhm,bhm->lbh", qs, ks_sum)


def linear_attention(value,
                     query_pos_ft,
                     key_pos_ft,
                     projection_matrix=None,
                     feature_map_type='fourier',
                     normalize_attn=False):
    if feature_map_type == 'fourier':
        query_prime = fourier_kernel_transformation(query_pos_ft, projection_matrix)  # [B,L,H,M]
        key_prime = fourier_kernel_transformation(key_pos_ft, projection_matrix)  # [B,L,H,M]
    elif feature_map_type == 'positive':
        query_prime = positive_kernel_transformation(query_pos_ft, True, projection_matrix)  # [B,L,H,M]
        key_prime = positive_kernel_transformation(key_pos_ft, False, projection_matrix)  # [B,L,H,M]
    else:
        assert False, "feature_type must be in ['trig', 'positive']"

    query_prime = tf.transpose(query_prime, [1, 0, 2, 3])  # [L,B,H,M]
    key_prime = tf.transpose(key_prime, [1, 0, 2, 3])  # [L,B,H,M]
    value = tf.transpose(value, [1, 0, 2, 3])  # [L,B,H,D]

    av_attention = attention_numerator(query_prime, key_prime, value)
    av_attention = tf.transpose(av_attention, [1, 0, 2, 3])
    if normalize_attn:
        attention_normalizer = attention_denominator(query_prime, key_prime)
        attention_normalizer = tf.transpose(attention_normalizer, [1, 0, 2])
        attention_normalizer = tf.expand_dims(attention_normalizer,
                                              len(attention_normalizer.shape))
        av_attention = av_attention / attention_normalizer
    return [av_attention, query_prime, key_prime]


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self,
               d_model,
               d_head,
               n_head,
               attention_dropout,
               feature_map_type='fourier',
               normalize_attn=False,
               d_kernel_map=128,
               head_init_range=(0, 1),
               **kwargs):

        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.size_per_head = d_head
        self.n_head = n_head
        self.attention_dropout = attention_dropout
        self.d_kernel_map = d_kernel_map
        self.feature_map_type = feature_map_type
        self.normalize_attn = normalize_attn
        self.head_init_range = head_init_range

        def _glorot_initializer(fan_in, fan_out):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

        attention_initializer = _glorot_initializer(self.d_model, self.n_head*self.size_per_head)

        self.value_weight = self.add_weight(
                "value_weight",
                shape=(self.d_model, self.n_head, self.size_per_head),
                initializer=attention_initializer,
                dtype=tf.float32,
                trainable=True)
        self.pos_ft_weight = self.add_weight(
                "pos_ft_weight",
                shape=(self.d_model, self.n_head, self.size_per_head),
                initializer=attention_initializer,
                dtype=tf.float32,
                trainable=False)
        self.pos_ft_scale = self.add_weight(
            "pos_ft_scale",
            shape=(1, 1, self.n_head, 1),
            initializer=tf.keras.initializers.Constant(1),
            dtype=tf.float32,
            trainable=True)

        head_left = self.head_init_range[0]
        head_right = self.head_init_range[1]
        head_range = head_right - head_left
        head_pos = tf.range(head_left+head_range/(2.*self.n_head), head_right, head_range/self.n_head)
        self.pos_ft_offsets = self.add_weight(
            "pos_ft_offests",
            shape=(1, 1, self.n_head, 1),
            initializer=tf.keras.initializers.Constant(head_pos),
            dtype=tf.float32,
            trainable=True)

        output_initializer = _glorot_initializer(self.n_head*self.size_per_head, self.d_model)
        self.output_weight = self.add_weight(
                "output_weight",
                shape=(self.n_head*self.size_per_head, self.d_model),
                initializer=output_initializer,
                dtype=tf.float32,
                trainable=True)
        self.output_dropout = tf.keras.layers.Dropout(self.attention_dropout)

        seed = np.random.randint(1e8, dtype=np.int32)
        projection_matrix = gen_projection_matrix(
                  self.d_kernel_map, self.size_per_head, seed=seed)
        initializer = tf.keras.initializers.Constant(projection_matrix)
        self.projection_matrix = self.add_weight(
            "projection_matrix",
            shape=projection_matrix.shape,
            initializer=initializer,
            dtype=projection_matrix.dtype,
            trainable=False)


    def call(self,
             source_input,
             pos_ft,
             pos_ft_slopes,
             training):
        value = tf.einsum("bnm,mhd->bnhd", source_input, self.value_weight)
        pos_ft_projected = tf.einsum("bnm,mhd->bnhd", pos_ft, self.pos_ft_weight)
        pos_ft_slopes_projected = tf.einsum("bnm,mhd->bnhd", pos_ft_slopes, self.pos_ft_weight)

        query_pos_ft = self.pos_ft_scale*pos_ft_projected
        slope_pos = self.pos_ft_scale*pos_ft_slopes_projected
        key_pos_ft = query_pos_ft + self.pos_ft_offsets*slope_pos

        attention_outputs = linear_attention(value,
                                             query_pos_ft, key_pos_ft,
                                             self.projection_matrix,
                                             self.feature_map_type,
                                             self.normalize_attn)

        bsz, slen = shape_list(attention_outputs[0])[:2]

        norms = tf.norm(pos_ft_slopes_projected, axis=-1, keepdims=True)/float(slen)
        attention_outputs[0] = norms*attention_outputs[0]

        attention_outputs[0] = tf.reshape(attention_outputs[0], [bsz, slen, -1])
        attention_outputs[0] = tf.einsum("bnm,md->bnd", attention_outputs[0], self.output_weight)
        attention_outputs[0] = self.output_dropout(attention_outputs[0], training=training)

        return attention_outputs


