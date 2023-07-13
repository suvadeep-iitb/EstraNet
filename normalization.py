import tensorflow as tf

class LayerScaling(tf.keras.layers.Layer):
    def __init__(self, epsilon=0.001, **kwargs):
        super().__init__(**kwargs)

        self.epsilon = epsilon


    def build(self, input_shape):
        param_shape = [input_shape[-1]]

        self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                initializer='ones',
                regularizer=None,
                constraint=None,
                trainable=True)


    def call(self, inputs):
        input_shape = inputs.shape
        ndims = len(input_shape)
        
        broadcast_shape = [1] * ndims
        broadcast_shape[-1] = input_shape.dims[-1].value
        scale = tf.reshape(self.gamma, broadcast_shape)

        _, variance = tf.nn.moments(inputs, axes=-1, keepdims=True)
        outputs = inputs / tf.math.sqrt(variance + self.epsilon) * scale

        return outputs



class LayerCentering(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def build(self, input_shape):
        param_shape = [input_shape[-1]]

        self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer='zeros',
                regularizer=None,
                constraint=None,
                trainable=True)


    def call(self, inputs):
        input_shape = inputs.shape
        ndims = len(input_shape)
        
        broadcast_shape = [1] * ndims
        broadcast_shape[-1] = input_shape.dims[-1].value
        offset = tf.reshape(self.beta, broadcast_shape)

        mean, _ = tf.nn.moments(inputs, axes=-1, keepdims=True)
        outputs = inputs - mean + offset

        return outputs
