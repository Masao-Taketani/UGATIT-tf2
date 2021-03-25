import tensorflow as tf
import tensorflow_addons as tfa


KERNEL_INIT = tf.random_normal_initializer(mean=0.0, stddev=0.02)
KERNEL_REG = tf.keras.regularizers.L2(0.0001)


def reflection_pad_2d(inputs, pad):
    return tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]], "REFLECT")


def upsample_images(x, scale_factor=2):
    _, h, w, _ = x.shape
    new_size = [h * scale_factor, w * scale_factor]
    tf.image.resize(x, size=new_size, method="nearest")


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


def Upsample(tf.keras.layers.Layer):

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
        self.lin = LIN(filters)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = upsample_images(inputs)
        x = reflection_pad_2d(x, self.pad)
        x = self.conv2d(x)
        x = self.lin(x)
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

        return x + inputs


class ResnetAdaLINBlock(tf.keras.layers.Layer):

    def __init__(self, dim, use_bias, smooth=True, name="ResnetAdaLINBlock"):
        super(ResnetAdaLINBlock, self).__init__(name=name)

        init_val = 0.9 if smooth else 0.1

        self.conv2d_1 = tf.keras.layers.Conv2D(filters=dim, 
                                               kernel_size=3, 
                                               strides=1, 
                                               use_bias=use_bias, 
                                               kernel_initializer=KERNEL_INIT,
                                               kernel_regularizer=KERNEL_REG)
        self.adalin_1 = AdaLIN(dim, init_val)
        self.relu = tf.keras.layers.ReLU()
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=dim, 
                                               kernel_size=3, 
                                               strides=1, 
                                               use_bias=use_bias, 
                                               kernel_initializer=KERNEL_INIT,
                                               kernel_regularizer=KERNEL_REG)
        self.adalin_2 = AdaLIN(dim, init_val)

    def call(self, inputs, gamma, beta):
        x = reflection_pad_2d(inputs, 1)
        x = self.conv2d_1(x)
        x = self.adalin_1(x, gamma, beta)
        x = self.relu(x)
        x = reflection_pad_2d(x, 1)
        x = self.conv2d_2(x)
        x = self.adalin_2(x, gamma, beta)

        return x + inputs   


class AdaLIN(tf.keras.layers.Layer, name="AdaLIN"):
    """
    Referred to the following pages.
    [Batch Normalization, Instance Normalization, Layer Normalization: Structural Nuances](https://becominghuman.ai/all-about-normalization-6ea79e70894b)
    [tf.Variable](https://www.tensorflow.org/api_docs/python/tf/Variable)
    [taki0112/UGATIT](https://github.com/taki0112/UGATIT/blob/d508e8f5188e47000d79d8aecada0cc9119e0d56/ops.py#L179)
    [znxlwm/UGATIT-pytorch](https://github.com/znxlwm/UGATIT-pytorch/blob/b8c4251823673189999484d07e97fdcb9300e9e0/networks.py#L157)
    """

    def __init__(self, dim, init_val):
        super(AdaLIN, self).__init__(name=name)
        self.epsilon = 1e-5
        self.rho = tf.Variable(initial_value=init_val, 
                               trainable=True, 
                               name="rho", 
                               constraint=lambda v: tf.clip_by_value(v, 
                                                                     clip_value_min=0.0, 
                                                                     clip_value_max=1.0)
                               shape=dim)

    def call(self, inputs, gamma, beta):
        in_mean, in_var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        in_out = (inputs - in_mean) / tf.sqrt(in_var + self.epsilon)
        ln_mean, ln_var = tf.nn.moments(inputs, axes=[1, 2, 3], keepdims=True)
        ln_out = (inputs - ln_mean) / tf.sqrt(ln_var + self.epsilon)
        out = self.rho * in_out + (1 - self.rho) * ln_out
        out = out * gamma + beta

        return out


class LIN(tf.keras.layers.Layer, name="LIN"):

    def __init__(self, dim):
        super(AdaLIN, self).__init__(name=name)
        self.epsilon = 1e-5
        self.rho = tf.Variable(initial_value=0.0, 
                               trainable=True, 
                               name="rho", 
                               constraint=lambda v: tf.clip_by_value(v, 
                                                                     clip_value_min=0.0, 
                                                                     clip_value_max=1.0)
                               shape=ch_dims)
        self.gamma = tf.Variable(initial_value=1.0, 
                                 trainable=True, 
                                 name="gamma", 
                                 shape=ch_dims)
        self.beta = tf.Variable(initial_value=0.0, 
                                 trainable=True, 
                                 name="beta", 
                                 shape=ch_dims)

    def call(self, inputs):
        in_mean, in_var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True) # shape [N, 1, 1, C]
        in_out = (inputs - in_mean) / tf.sqrt(in_var + self.epsilon)
        ln_mean, ln_var = tf.nn.moments(inputs, axes=[1, 2, 3], keepdims=True) # shape [N, 1, 1, 1]
        ln_out = (inputs - ln_mean) / tf.sqrt(ln_var + self.epsilon)
        out = self.rho * in_out + (1 - self.rho) * ln_out
        out = out * self.gamma + self.beta

        return out


class Generator(tf.keras.Layers.Layer):

    def __init__(self, first_filters=64, img_size=256, name="Generator"):
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
        self.Dense_1 = tf.keras.layers.Dense(4 * first_filters,
                                             use_bias=False, 
                                             kernel_initializer=KERNEL_INIT,
                                             kernel_regularizer=KERNEL_REG)
        self.relu_1 = tf.keras.layers.ReLU()
        self.Dense_2 = tf.keras.layers.Dense(4 * first_filters,
                                             use_bias=False, 
                                             kernel_initializer=KERNEL_INIT,
                                             kernel_regularizer=KERNEL_REG)
        self.relu_2 = tf.keras.layers.ReLU()
        self.gamma = tf.keras.layers.Dense(4 * first_filters,
                                           use_bias=False, 
                                           kernel_initializer=KERNEL_INIT,
                                           kernel_regularizer=KERNEL_REG)
        self.bias = tf.keras.layers.Dense(4 * first_filters,
                                          use_bias=False, 
                                          kernel_initializer=KERNEL_INIT,
                                          kernel_regularizer=KERNEL_REG)

        # Uesd for Decoder Bottleneck part
        self.resnet_adalin_block_1 = ResnetAdaLINBlock(4 * first_filters,
                                                       use_bias=False)
        self.resnet_adalin_block_2 = ResnetAdaLINBlock(4 * first_filters,
                                                       use_bias=False)
        self.resnet_adalin_block_3 = ResnetAdaLINBlock(4 * first_filters,
                                                       use_bias=False)
        self.resnet_adalin_block_4 = ResnetAdaLINBlock(4 * first_filters,
                                                       use_bias=False)
        
        # used for Decoder Up-sampling part
