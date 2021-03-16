import tensorflow as tf
import tensorflow_addons as tfa


KERNEL_INIT = tf.random_normal_initializer(mean=0.0, stddev=0.02)
KERNEL_REG = tf.keras.regularizers.L2(0.0001)


def reflection_pad_2d(inputs, pad):
    return tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]], "REFLECT")


class Downsample(tf.keras.layers.Layer):

    def __init__(self, 
                 pad, 
                 filters, 
                 kernel_size, 
                 strides, 
                 use_bias, 
                 name="Downsample"):
        super(Downsample, self).__init__(name=name)
        self.pad = pad
        self.conv2d = tf.keras.layers.Conv2D(filters=filters, 
                                             kernel_size=kernel_size, 
                                             strides=strides, 
                                             use_bias=use_bias, 
                                             kernel_initializer=KERNEL_INIT,
                                             kernel_regularizer=KERNEL_REG)
        self.instance_norm = tfa.layers.InstanceNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = reflection_pad_2d(inputs, self.pad)
        x = self.conv2d(x)
        x = self.instance_norm(x)
        x = self.relu(x)

        return x


class ResnetBlock(tf.keras.layers.Layer):

    def __init__(self, dim, use_bias, name="ResnetBlock"):
        super(ResnetBlock, self).__init__(name=name)
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=dim, 
                                               kernel_size=3, 
                                               strides=1, 
                                               use_bias=use_bias, 
                                               kernel_initializer=KERNEL_INIT,
                                               kernel_regularizer=KERNEL_REG)
        self.instance_norm_1 = tfa.layers.InstanceNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=dim, 
                                               kernel_size=3, 
                                               strides=1, 
                                               use_bias=use_bias, 
                                               kernel_initializer=KERNEL_INIT,
                                               kernel_regularizer=KERNEL_REG)
        self.instance_norm_2 = tfa.layers.InstanceNormalization()

    def call(self, inputs):
        x = reflection_pad_2d(inputs, 1)
        x = self.conv2d_1(x)
        x = self.instance_norm_1(x)
        x = self.relu(x)
        x = reflection_pad_2d(x, 1)
        x = self.conv2d_2(x)
        x = self.instance_norm_2(x)

        return x


class ResnetAdaLINBlock(tf.keras.layers.Layer):

    def __init__(self, dim, use_bias, name="ResnetAdaLINBlock"):
        super(ResnetAdaLINBlock, self).__init__(name=name)
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=dim, 
                                               kernel_size=3, 
                                               strides=1, 
                                               use_bias=use_bias, 
                                               kernel_initializer=KERNEL_INIT,
                                               kernel_regularizer=KERNEL_REG)
        self.instance_norm_1 = tfa.layers.InstanceNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=dim, 
                                               kernel_size=3, 
                                               strides=1, 
                                               use_bias=use_bias, 
                                               kernel_initializer=KERNEL_INIT,
                                               kernel_regularizer=KERNEL_REG)
        self.instance_norm_2 = tfa.layers.InstanceNormalization()

    def call(self, inputs):
        x = reflection_pad_2d(inputs, 1)
        x = self.conv2d_1(x)
        x = self.instance_norm_1(x)
        x = self.relu(x)
        x = reflection_pad_2d(x, 1)
        x = self.conv2d_2(x)
        x = self.instance_norm_2(x)

        return x        


class AdaLIN(tf.keras.layers.Layer):

    def __init__(self, dim, use_bias):
        super(self, )


class Upsample(tf.keras.layers.Layer):

    def __init__(self,
                 name='Upsample')
        super(Upsample, self).__init__(name=name)



class Generator(tf.keras.Layers.Layer):

    def __init__(self, first_filters=64, name="Generator"):
        super(Generator, self).__init__(name=name)
        # Used for Encoder Down-sampling part
        self.downsample_1 = Downsample(pad=3, 
                                       filters=first_filters, 
                                       kernel_size=7, 
                                       strides=1, 
                                       use_bias=False,
                                       kernel_init=kernel_init,
                                       kernel_reg=kernel_reg)
        self.downsample_2 = Downsample(pad=1,
                                       filters= 2 * first_filters,
                                       kernel_size=3,
                                       strides=2,
                                       use_bias=False)
        self.downsample_3 = Downsample(pad=1,
                                       filters= 4 * first_filters,
                                       kernel_size=3,
                                       strides=2,
                                       use_bias=False)
        # Used for Encoder Bottleneck part
        self.resnet_block_1 = ResnetBlock(4 * first_filters,
                                          use_bias=False)
        self.resnet_block_2 = ResnetBlock(4 * first_filters,
                                          use_bias=False)
        self.resnet_block_3 = ResnetBlock(4 * first_filters,
                                          use_bias=False)
        self.resnet_block_4 = ResnetBlock(4 * first_filters,
                                          use_bias=False)
        # Used for CAM of Generator part
        self.gap_fc = tf.keras.layers.Dense(units=1, 
                                            use_bias=False, 
                                            kernel_initializer=KERNEL_INIT,
                                            kernel_regularizer=KERNEL_REG)
        self.gmp_fc = tf.keras.layers.Dense(units=1, 
                                            use_bias=False, 
                                            kernel_initializer=KERNEL_INIT,
                                            kernel_regularizer=KERNEL_REG)
        self.conv1x1 = tf.keras.layers.Conv2D(filters=4 * first_filters,
                                              kernel_size=1,
                                              strides=1,
                                              use_bias=True,
                                              kernel_initializer=KERNEL_INIT,
                                              kernel_regularizer=KERNEL_REG)
        self.relu = tf.keras.layers.ReLU()
        # Used for Gamma, Beta part

        # Uesd for Decoder Bottleneck part

