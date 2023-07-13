import tensorflow as tf
from fast_attention import SelfAttention
from normalization import LayerScaling, LayerCentering
from tensorflow.keras.layers.experimental import SyncBatchNormalization


def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class PositionalFeature(tf.keras.layers.Layer):
    def __init__(self, d_feature, beta_2, **kwargs):
        super().__init__(**kwargs)

        self.slopes = tf.range(d_feature, 0, -4.0, dtype=tf.float32) / d_feature
        self.slopes = self.slopes * beta_2

    def call(self, slen, bsz=None):
        pos_seq = tf.range(0, slen, 1.0, dtype=tf.float32)
        normalized_slopes = (1. / float(slen-1)) * self.slopes
        forward = tf.einsum("i,j->ij", pos_seq, normalized_slopes)
        backward = tf.reverse(forward, axis=[0])
        neg_forward = -tf.identity(forward)
        neg_backward = -tf.identity(backward)
        pos_feature = tf.concat([forward, backward, neg_forward, neg_backward], -1)

        pos_feature_slopes = tf.concat(
                              [tf.identity(normalized_slopes),
                               -tf.identity(normalized_slopes),
                               -tf.identity(normalized_slopes),
                               tf.identity(normalized_slopes)], axis=0)
        pos_feature_slopes = float(slen-1)*tf.reshape(pos_feature_slopes, [1, -1])

        if bsz is not None:
            pos_feature = tf.tile(pos_feature[None, :, :], [bsz, 1, 1])
            pos_feature_slopes = tf.tile(pos_feature_slopes[None, :, :], [bsz, 1, 1])
        else:
            pos_feature = pos_feature[None, :, :]
            pos_feature_slopes = pos_feature_slopes[None, :, :]
        return pos_feature, pos_feature_slopes


class PositionwiseFF(tf.keras.layers.Layer):
    def __init__(self, d_model, d_inner, dropout, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.layer_1 = tf.keras.layers.Dense(
            d_inner, activation=tf.nn.relu, name='layer_1'
        )
        self.drop_1 = tf.keras.layers.Dropout(dropout, name='drop_1')
        self.layer_2 = tf.keras.layers.Dense(d_model, name='layer_2')
        self.drop_2 = tf.keras.layers.Dropout(dropout, name='drop_2')


    def call(self, inp, training=False):
        core_out = inp
        core_out = self.layer_1(core_out)
        core_out = self.drop_1(core_out, training=training)
        core_out = self.layer_2(core_out)
        core_out = self.drop_2(core_out, training=training)

        output = [core_out]
        return output


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_head,
        d_head,
        d_model,
        d_inner,
        dropout,
        feature_map_type,
        normalize_attn,
        d_kernel_map,
        model_normalization,
        head_init_range,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.feature_map_type = feature_map_type
        self.normalize_attn = normalize_attn
        self.d_kernel_map = d_kernel_map
        self.model_normalization = model_normalization
        self.head_init_range = head_init_range

        self.self_attn = SelfAttention(
            d_model=self.d_model,
            d_head=self.d_head,
            n_head=self.n_head,
            attention_dropout=self.dropout,
            feature_map_type=self.feature_map_type,
            normalize_attn=self.normalize_attn,
            d_kernel_map=self.d_kernel_map,
            head_init_range = self.head_init_range,
            name="tran_attn",
        )
        self.pos_ff = PositionwiseFF(
            d_model=self.d_model,
            d_inner=self.d_inner,
            dropout=self.dropout,
            name="pos_ff",
        )

        assert self.model_normalization in ['preLC', 'postLC', 'none'], "model_normalization must be one of 'preLC', 'postLC' or 'none'"
        if self.model_normalization in ['preLC', 'postLC']:
            self.lc1 = LayerCentering()
            self.lc2 = LayerCentering()


    def call(self, inputs, training=False):
        inp, pos_ft, pos_ft_slopes = inputs
        if self.model_normalization == 'preLC':
            attn_in = self.lc1(inp)
        else:
            attn_in = inp
        attn_outputs = self.self_attn(attn_in, pos_ft, pos_ft_slopes,
                                      training=training)
        attn_outputs[0] = attn_outputs[0] + inp
        if self.model_normalization == 'postLC':
            attn_outputs[0] = self.lc1(attn_outputs[0])

        if self.model_normalization == 'preLC':
            ff_in = self.lc2(attn_outputs[0])
        else:
            ff_in = attn_outputs[0]
        ff_output = self.pos_ff(ff_in, training=training)
        ff_output[0] = ff_output[0] + attn_outputs[0]
        if self.model_normalization == 'postLC':
            ff_output[0] = self.lc2(ff_output[0])

        outputs = [ff_output[0]] + attn_outputs[1:]

        return outputs


class SoftmaxAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, d_head, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.q_heads = self.add_weight(
            shape=(self.d_head, self.n_head), name="q_heads"
        )
        self.k_net = tf.keras.layers.Dense(
            self.d_head * self.n_head, name="k_net"
        )
        self.v_net = tf.keras.layers.Dense(
            self.d_head * self.n_head, name="v_net"
        )

        self.scale = 1. / (self.d_head ** 0.5)


    def build(self, input_shape):
        self.softmax_attn_smoothing = self.add_weight(
            "softmax_attn_smoothing",
            shape=(),
            initializer=tf.keras.initializers.Constant(0),
            dtype=tf.float32,
            trainable=False)


    def call(self, inp, softmax_attn_smoothing, training=False):
        bsz, slen = inp.shape[:2]
        if training:
            self.softmax_attn_smoothing.assign(softmax_attn_smoothing)

        k_head = self.k_net(inp)
        v_head = self.v_net(inp)

        k_head = tf.reshape(k_head, [-1, slen, self.d_head, self.n_head])
        v_head = tf.reshape(v_head, [-1, slen, self.d_head, self.n_head])

        attn_score = tf.einsum("bndh,dh->bnh", k_head, self.q_heads)
        attn_score = attn_score * self.scale * self.softmax_attn_smoothing

        attn_prob = tf.nn.softmax(attn_score, axis=1)

        attn_out = tf.einsum("bndh,bnh->bnhd", v_head, attn_prob)
        attn_out = tf.reshape(attn_out, [bsz, slen, -1])

        return attn_out, attn_score


class Transformer(tf.keras.Model):
    def __init__(self, n_layer, d_model, d_head, n_head, d_inner, 
                 d_head_softmax, n_head_softmax, dropout, n_classes, 
                 conv_kernel_size, n_conv_layer, pool_size, d_kernel_map, beta_2, 
                 model_normalization, head_initialization='forward', 
                 softmax_attn=True, output_attn=False):

        super(Transformer, self).__init__()

        self.n_layer = n_layer
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.d_inner = d_inner
        self.d_head_softmax = d_head_softmax
        self.n_head_softmax = n_head_softmax
        self.feature_map_type = 'fourier'
        self.normalize_attn = False
        self.d_kernel_map = d_kernel_map
        self.beta_2 = beta_2
        self.model_normalization = model_normalization
        self.head_initialization = head_initialization
        self.softmax_attn = softmax_attn

        self.dropout = dropout 

        self.n_classes = n_classes

        self.conv_kernel_size = conv_kernel_size
        self.n_conv_layer = n_conv_layer
        self.pool_size = pool_size

        self.output_attn = output_attn

        conv_filters = [min(8*2**i, self.d_model) for i in range(self.n_conv_layer-1)] + [self.d_model]

        self.conv_layers = []
        self.norm_layers = []
        self.relu_layers = []
        self.pool_layers = []

        for l in range(self.n_conv_layer):
            ks = 11 if l is 0 else self.conv_kernel_size
            self.conv_layers.append(tf.keras.layers.Conv1D(conv_filters[l], ks))
            self.relu_layers.append(tf.keras.layers.ReLU())
            self.pool_layers.append(tf.keras.layers.AveragePooling1D(self.pool_size, self.pool_size))

        self.pos_feature = PositionalFeature(self.d_model, self.beta_2)

        head_init_ranges = []
        if self.head_initialization == 'forward':
            for i in range(self.n_layer):
                if i == 0:
                    head_init_ranges.append((0., 0.5))
                else:
                    head_init_ranges.append((0., 1.0))
        elif self.head_initialization == 'backward':
            for i in range(self.n_layer):
                if i == 0:
                    head_init_ranges.append((-0.5, 0.0))
                else:
                    head_init_ranges.append((-1.0, 0.0))
        elif self.head_initialization == 'symmetric':
            for i in range(self.n_layer):
                if i == 0:
                    head_init_ranges.append((-0.5, 0.5))
                else:
                    head_init_ranges.append((-1.0, 1.0))
        else:
            assert False, "head_initialization can be one of ['forward', 'backward', 'symmetric']"

        self.tran_layers = []
        for i in range(self.n_layer):
            self.tran_layers.append(
                TransformerLayer(
                    n_head=self.n_head,
                    d_head=self.d_head,
                    d_model=self.d_model,
                    d_inner=self.d_inner,
                    dropout=self.dropout,
                    feature_map_type=self.feature_map_type,
                    normalize_attn=self.normalize_attn,
                    d_kernel_map=self.d_kernel_map,
                    model_normalization=self.model_normalization,
                    head_init_range = head_init_ranges[i],
                    name='layers_._{}'.format(i)
                )
            )

        self.out_dropout = tf.keras.layers.Dropout(dropout, name='out_drop')

        if self.softmax_attn:
            self.out_attn = SoftmaxAttention(d_model=self.d_model, n_head=self.n_head_softmax, 
                                             d_head=self.d_head_softmax)
        self.fc_output = tf.keras.layers.Dense(self.n_classes)

    def call(self, inp, softmax_attn_smoothing=1, training=False):
        # convert the input dimension from [bsz, len] to [bsz, len, 1]
        inp = tf.expand_dims(inp, axis=-1)

        # apply the convolution blocks
        for l in range(self.n_conv_layer):
            inp = self.conv_layers[l](inp)
            inp = self.relu_layers[l](inp)
            inp = self.pool_layers[l](inp)

        bsz, slen = shape_list(inp)[:2]

        pos_ft, pos_ft_slopes = self.pos_feature(slen, bsz)

        core_out = inp
        out_list = []
        for i, layer in enumerate(self.tran_layers):
            all_out = layer([core_out, pos_ft, pos_ft_slopes], training=training)
            core_out = all_out[0]
            out_list.append(all_out[1:])
        core_out = self.out_dropout(core_out, training=training)

        # take the evarage across the first (len) dimension to get the final representation
        if self.softmax_attn:
            core_out, softmax_attn_score = self.out_attn(core_out, softmax_attn_smoothing, training=training)
        else:
            softmax_attn_score = None
        output = tf.reduce_mean(core_out, axis=1)

        # ge the final scores for all classes
        scores = self.fc_output(output)

        for i in range(len(out_list)):
            for j in range(len(out_list[i])):
                out_list[i][j] = tf.transpose(out_list[i][j], [1, 0, 2, 3])

        if self.output_attn:
            return [scores, out_list, softmax_attn_score]
        else:
            return [scores]
        

