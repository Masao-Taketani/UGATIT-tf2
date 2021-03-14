import tensorflow as tf
import tensorflow_addons as tfa


kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.2)
kernel_regularizer = tf.keras.regularizers.L2(0.0001)


def reflection_pad_2d(pad):
    return tf.pad([[0, 0], [pad, pad], [pad, pad], [0, 0]], "REFLECT")


class Generator(tf.keras.Layers.Layer):

    def __init__(self, first_filters=64, name="Generator"):
        super(Generator, self).__init__(name=name)
        self.pad_1 = reflection_pad_2d(3)
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=first_filters, 
                                               kernel_size=7, 
                                               strides=1, 
                                               use_bias=False, 
                                               kernel_initializer=kernel_initializer,
                                               kernel_regularizer=kernel_regularizer)