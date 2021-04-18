import tensorflow as tf
import tensorflow_addons as tfa


KERNEL_INIT = tf.random_normal_initializer(mean=0.0, stddev=0.02)
KERNEL_REG = tf.keras.regularizers.L2(0.0001)


def reflection_pad_2d(inputs, pad):
    return tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]], "REFLECT")


def upsample_images(x, scale_factor=2):
    _, h, w, _ = x.shape
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize(x, size=new_size, method="nearest")


class Downsample(tf.keras.layers.Layer):

    def __init__(self, 
                 pad, 
                 filters, 
                 kernel_size, 
                 strides, 
                 use_bias,
                 is_generator,
                 act="lk_relu",
                 name="downsample"):
        super(Downsample, self).__init__(name=name)
        self.pad = pad
        self.act = act
        self.is_generator = is_generator
        conv = tf.keras.layers.Conv2D(filters=filters, 
                                      kernel_size=kernel_size, 
                                      strides=strides, 
                                      use_bias=use_bias, 
                                      kernel_initializer=KERNEL_INIT,
                                      kernel_regularizer=KERNEL_REG)

        if self.is_generator:
            self.conv2d = conv
            self.layer_norm = tfa.layers.InstanceNormalization()
            self.activation = tf.keras.layers.ReLU()
        else:
            self.conv2d = tfa.layers.SpectralNormalization(conv)
            if self.act == "lk_relu":
                self.activation = tf.keras.layers.LeakyReLU(0.2)

    def call(self, inputs):
        x = reflection_pad_2d(inputs, self.pad)
        x = self.conv2d(x)
        if self.is_generator:
            x = self.layer_norm(x)
        if self.is_generator or self.act == "lk_relu":
            x = self.activation(x)

        return x


class Upsample(tf.keras.layers.Layer):

    def __init__(self, 
                 pad, 
                 filters, 
                 kernel_size, 
                 strides, 
                 use_bias,
                 use_upsample_imgs=True,
                 use_relu=True,
                 use_mp=False,
                 name="upsample"):
        super(Upsample, self).__init__(name=name)
        
        self.use_ups = use_upsample_imgs
        self.pad = pad
        self.conv2d = tf.keras.layers.Conv2D(filters=filters, 
                                             kernel_size=kernel_size, 
                                             strides=strides, 
                                             use_bias=use_bias, 
                                             kernel_initializer=KERNEL_INIT,
                                             kernel_regularizer=KERNEL_REG)
        self.lin = LIN(filters, use_mp=use_mp)
        if use_relu:
            self.activation = tf.keras.layers.ReLU()
        else:
            self.activation = tf.keras.activations.tanh

    def call(self, inputs):
        x = inputs
        if self.use_ups:
            x = upsample_images(x)
        x = reflection_pad_2d(x, self.pad)
        x = self.conv2d(x)
        x = self.lin(x)
        x = self.activation(x)

        return x


class ResnetBlock(tf.keras.layers.Layer):

    def __init__(self, dim, use_bias, name="resnet_block"):
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

    def __init__(self, dim, use_bias, smoothing=True, use_mp=False, name="resnet_adalin_block"):
        super(ResnetAdaLINBlock, self).__init__(name=name)
        init_val = 0.9 if smoothing else 1.0

        self.conv2d_1 = tf.keras.layers.Conv2D(filters=dim, 
                                               kernel_size=3, 
                                               strides=1, 
                                               use_bias=use_bias, 
                                               kernel_initializer=KERNEL_INIT,
                                               kernel_regularizer=KERNEL_REG)
        self.adalin_1 = AdaLIN(dim, init_val, use_mp=use_mp)
        self.relu = tf.keras.layers.ReLU()
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=dim, 
                                               kernel_size=3, 
                                               strides=1, 
                                               use_bias=use_bias, 
                                               kernel_initializer=KERNEL_INIT,
                                               kernel_regularizer=KERNEL_REG)
        self.adalin_2 = AdaLIN(dim, init_val, use_mp=use_mp)

    def call(self, inputs, gamma, beta):
        x = reflection_pad_2d(inputs, 1)
        x = self.conv2d_1(x)
        x = self.adalin_1(x, gamma, beta)
        x = self.relu(x)
        x = reflection_pad_2d(x, 1)
        x = self.conv2d_2(x)
        x = self.adalin_2(x, gamma, beta)

        return x + inputs   


class AdaLIN(tf.keras.layers.Layer):
    """
    Referred to the following pages.
    [Batch Normalization, Instance Normalization, Layer Normalization: Structural Nuances](https://becominghuman.ai/all-about-normalization-6ea79e70894b)
    [tf.Variable](https://www.tensorflow.org/api_docs/python/tf/Variable)
    [taki0112/UGATIT](https://github.com/taki0112/UGATIT/blob/d508e8f5188e47000d79d8aecada0cc9119e0d56/ops.py#L179)
    [znxlwm/UGATIT-pytorch](https://github.com/znxlwm/UGATIT-pytorch/blob/b8c4251823673189999484d07e97fdcb9300e9e0/networks.py#L157)
    """

    def __init__(self, dim, init_val, use_mp=False, name="adalin"):
        super(AdaLIN, self).__init__(name=name)
        self.epsilon = 1e-5
        dtype = tf.float16 if use_mp else tf.float32

        self.rho = tf.Variable(initial_value=tf.constant(init_val, shape=[dim], dtype=dtype), 
                               trainable=True, 
                               name="rho",
                               constraint=lambda v: tf.clip_by_value(v, 
                                                                     clip_value_min=0.0, 
                                                                     clip_value_max=1.0))

    def call(self, inputs, gamma, beta):
        """[Instance Normalization part]
        calcuate mean and variance for each channel of for each batch
        shape of mean and variance: [bs, h, w, ch] -> [bs, 1, 1, ch]
        """
        in_mean, in_var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        in_out = (inputs - in_mean) / tf.sqrt(in_var + self.epsilon)
        """[Layer Normalization part]
        calcuate mean and variance for each image of for each batch
        shape of mean and variance: [bs, h, w, ch] -> [bs, 1, 1, 1]
        """
        ln_mean, ln_var = tf.nn.moments(inputs, axes=[1, 2, 3], keepdims=True)
        ln_out = (inputs - ln_mean) / tf.sqrt(ln_var + self.epsilon)
        # [ch] * [bs, 1, 1, ch] + [ch] * [bs, 1, 1, 1] -> [bs, 1, 1, ch]
        out = self.rho * in_out + (1 - self.rho) * ln_out
        out = out * gamma + beta

        return out


class LIN(tf.keras.layers.Layer):

    def __init__(self, dim, use_mp=False, name="lin"):
        super(LIN, self).__init__(name=name)
        self.epsilon = 1e-5
        dtype = tf.float16 if use_mp else tf.float32

        self.rho = tf.Variable(initial_value=tf.zeros(dim, dtype=dtype), 
                               trainable=True, 
                               name="rho", 
                               constraint=lambda v: tf.clip_by_value(v, 
                                                                     clip_value_min=0.0, 
                                                                     clip_value_max=1.0))
        self.gamma = tf.Variable(initial_value=tf.ones(dim, dtype=dtype), 
                                 trainable=True, 
                                 name="gamma")
        self.beta = tf.Variable(initial_value=tf.zeros(dim, dtype=dtype), 
                                 trainable=True, 
                                 name="beta") 

    def call(self, inputs):
        # [bs, h, w, ch] -> [bs, 1, 1, ch]
        in_mean, in_var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        in_out = (inputs - in_mean) / tf.sqrt(in_var + self.epsilon)
        # bs, h, w, ch] -> [bs, 1, 1, 1]
        ln_mean, ln_var = tf.nn.moments(inputs, axes=[1, 2, 3], keepdims=True)
        ln_out = (inputs - ln_mean) / tf.sqrt(ln_var + self.epsilon)
        out = self.rho * in_out + (1 - self.rho) * ln_out
        out = out * self.gamma + self.beta

        return out

class CAM(tf.keras.layers.Layer):

    def __init__(self, filters, is_generator, name="CAM"):
        super(CAM, self).__init__(name=name)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.global_max_pool = tf.keras.layers.GlobalMaxPool2D()
        if is_generator:
            self.gap_fc = tf.keras.layers.Dense(units=1, 
                                                use_bias=False, 
                                                kernel_initializer=KERNEL_INIT,
                                                kernel_regularizer=KERNEL_REG,
                                                name="g_gap_fc")
            self.gmp_fc = tf.keras.layers.Dense(units=1, 
                                                use_bias=False, 
                                                kernel_initializer=KERNEL_INIT,
                                                kernel_regularizer=KERNEL_REG,
                                                name="g_gmp_fc")
            self.activation = tf.keras.layers.ReLU()
            kind = "g"
        else:
            self.gap_fc = tfa.layers.SpectralNormalization(tf.keras.layers.Dense(units=1, 
                                                use_bias=False, 
                                                kernel_initializer=KERNEL_INIT,
                                                kernel_regularizer=KERNEL_REG,
                                                name="d_gap_fc"))
            self.gmp_fc = tfa.layers.SpectralNormalization(tf.keras.layers.Dense(units=1, 
                                                use_bias=False, 
                                                kernel_initializer=KERNEL_INIT,
                                                kernel_regularizer=KERNEL_REG,
                                                name="d_gmp_fc"))
            self.activation = tf.keras.layers.LeakyReLU(0.2)
            kind = "d"

        self.conv1x1 = tf.keras.layers.Conv2D(filters=filters,
                                              kernel_size=1,
                                              strides=1,
                                              use_bias=True,
                                              kernel_initializer=KERNEL_INIT,
                                              kernel_regularizer=KERNEL_REG,
                                              name=kind+"_conv1x1")

    def call(self, x):
        # [bs, h, w, ch] -> [bs, ch]
        gap = self.global_avg_pool(x)
        # [bs, ch] -> [bs, 1]
        gap_logit = self.gap_fc(gap)
        # [ch, 1] -> [ch]
        gap_weight = tf.squeeze(self.gap_fc.trainable_variables[0])
        # [bs, h, w, ch] * [ch] (multiply various weights for each channel)
        gap = x * gap_weight

        gmp = self.global_max_pool(x)
        gmp_logit = self.gmp_fc(gmp)
        # [bs, h, w, ch] * [ch] (multiply various weights for each channel)
        gmp_weight = tf.squeeze(self.gmp_fc.trainable_variables[0])
        gmp = x * gmp_weight
        # gap_logit: [bs, 1], gmp_logit: [bs, 1] -> [bs, 2]
        cam_logit = tf.concat([gap_logit, gmp_logit], axis=1)

        x = tf.concat([gap, gmp], axis=3)
        x = self.activation(self.conv1x1(x))
        # [bs, h, w, ch] -> [bs, h, w, 1]
        heatmap = tf.math.reduce_sum(x, axis=3, keepdims=True)

        return x, cam_logit, heatmap


class Generator(tf.keras.layers.Layer):

    def __init__(self, first_filters=64, img_size=256, use_mp=False, name="generator"):
        super(Generator, self).__init__(name=name)
        self.img_size = img_size
        self.use_mp = use_mp
        # For Encoder Down-sampling part
        self.downsample_1 = Downsample(pad=3, 
                                       filters=first_filters, 
                                       kernel_size=7, 
                                       strides=1, 
                                       use_bias=False,
                                       is_generator=True,
                                       name="g_downsample_1")
        self.downsample_2 = Downsample(pad=1,
                                       filters=2 * first_filters,
                                       kernel_size=3,
                                       strides=2,
                                       use_bias=False,
                                       is_generator=True,
                                       name="g_downsample_2")
        self.downsample_3 = Downsample(pad=1,
                                       filters=4 * first_filters,
                                       kernel_size=3,
                                       strides=2,
                                       use_bias=False,
                                       is_generator=True,
                                       name="g_downsample_3")

        # For Encoder Bottleneck part
        self.resnet_block_1 = ResnetBlock(4 * first_filters,
                                          use_bias=False,
                                          name="g_resnet_block_1")
        self.resnet_block_2 = ResnetBlock(4 * first_filters,
                                          use_bias=False,
                                          name="g_resnet_block_2")
        self.resnet_block_3 = ResnetBlock(4 * first_filters,
                                          use_bias=False,
                                          name="g_resnet_block_3")
        self.resnet_block_4 = ResnetBlock(4 * first_filters,
                                          use_bias=False,
                                          name="g_resnet_block_4")

        # For CAM of Generator part
        self.cam = CAM(4 * first_filters, is_generator=True, name="g_CAM")

        # For Gamma, Beta part
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(4 * first_filters,
                                             use_bias=False, 
                                             kernel_initializer=KERNEL_INIT,
                                             kernel_regularizer=KERNEL_REG)
        self.relu_1 = tf.keras.layers.ReLU()
        self.dense_2 = tf.keras.layers.Dense(4 * first_filters,
                                             use_bias=False, 
                                             kernel_initializer=KERNEL_INIT,
                                             kernel_regularizer=KERNEL_REG)
        self.relu_2 = tf.keras.layers.ReLU()
        self.gamma = tf.keras.layers.Dense(4 * first_filters,
                                           use_bias=False, 
                                           kernel_initializer=KERNEL_INIT,
                                           kernel_regularizer=KERNEL_REG,
                                           name="g_gamma")
        self.beta = tf.keras.layers.Dense(4 * first_filters,
                                          use_bias=False, 
                                          kernel_initializer=KERNEL_INIT,
                                          kernel_regularizer=KERNEL_REG,
                                          name="g_beta")

        # For Decoder Bottleneck part
        self.resnet_adalin_block_1 = ResnetAdaLINBlock(4 * first_filters,
                                                       use_bias=False,
                                                       use_mp=self.use_mp,
                                                       name="g_resnet_adalin_block_1")
        self.resnet_adalin_block_2 = ResnetAdaLINBlock(4 * first_filters,
                                                       use_bias=False,
                                                       use_mp=self.use_mp,
                                                       name="g_resnet_adalin_block_2")
        self.resnet_adalin_block_3 = ResnetAdaLINBlock(4 * first_filters,
                                                       use_bias=False,
                                                       use_mp=self.use_mp,
                                                       name="g_resnet_adalin_block_3")
        self.resnet_adalin_block_4 = ResnetAdaLINBlock(4 * first_filters,
                                                       use_bias=False,
                                                       use_mp=self.use_mp,
                                                       name="g_resnet_adalin_block_4")
        
        # For Decoder Up-sampling part
        self.upsample_1 = Upsample(pad=1, 
                                   filters=2 * first_filters, 
                                   kernel_size=3, 
                                   strides=1, 
                                   use_bias=False,
                                   use_mp=use_mp,
                                   name="g_upsample_1")
        self.upsample_2 = Upsample(pad=1, 
                                   filters=first_filters, 
                                   kernel_size=3, 
                                   strides=1, 
                                   use_bias=False,
                                   use_mp=use_mp,
                                   name="g_upsample_2")
        self.upsample_3 = Upsample(pad=3, 
                                   filters=3, 
                                   kernel_size=7, 
                                   strides=1, 
                                   use_bias=False,
                                   use_upsample_imgs=False,
                                   use_relu=False,
                                   use_mp=use_mp,
                                   name="g_upsample_3")
        if self.use_mp:
            # This linear activation is used to make the dtype back to tf.float32 for Mixed Precision
            self.cast_last_output = tf.keras.layers.Activation("linear", dtype="float32")

    def call(self, inputs):
        x = self.downsample_1(inputs)
        x = self.downsample_2(x)
        x = self.downsample_3(x)

        x = self.resnet_block_1(x)
        x = self.resnet_block_2(x)
        x = self.resnet_block_3(x)
        x = self.resnet_block_4(x)

        x, cam_logit, heatmap = self.cam(x)
        
        x_ = self.dense_1(self.flatten(x))
        x_ = self.relu_1(x_)
        x_ = self.dense_2(x_)
        x_ = self.relu_2(x_)
        gamma, beta = self.gamma(x_), self.beta(x_)

        x = self.resnet_adalin_block_1(x, gamma, beta)
        x = self.resnet_adalin_block_2(x, gamma, beta)
        x = self.resnet_adalin_block_3(x, gamma, beta)
        x = self.resnet_adalin_block_4(x, gamma, beta)

        x = self.upsample_1(x)
        x = self.upsample_2(x)
        out = self.upsample_3(x)

        if self.use_mp:
            out = self.cast_last_output(out)
            cam_logit = self.cast_last_output(cam_logit)
            heatmap = self.cast_last_output(heatmap)

        return out, cam_logit, heatmap

    def summary(self):
        x = tf.keras.layers.Input(shape=(self.img_size, self.img_size, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name=self.name)
        return model.summary()


class Discriminator(tf.keras.layers.Layer):
    
    def __init__(self, is_global, first_filters=64, pad=1, img_size=256, use_mp=False, name="discriminator"):
        super(Discriminator, self).__init__(name=name)
        self.is_global = is_global
        self.img_size = img_size
        self.use_mp = use_mp
        # For Encoder Down-sampling part
        self.downsample_1 = Downsample(pad=pad, 
                                       filters=first_filters,
                                       kernel_size=4,
                                       strides=2,
                                       use_bias=True,
                                       is_generator=False,
                                       name="d_downsample_1")
        self.downsample_2 = Downsample(pad=pad, 
                                       filters=first_filters * 2,
                                       kernel_size=4,
                                       strides=2,
                                       use_bias=True,
                                       is_generator=False,
                                       name="d_downsample_2")
        self.downsample_3 = Downsample(pad=pad, 
                                       filters=first_filters * 4,
                                       kernel_size=4,
                                       strides=2,
                                       use_bias=True,
                                       is_generator=False,
                                       name="d_downsamle_3")
        enc_last_filters = first_filters * 8

        if self.is_global:
            self.downsample_4 = Downsample(pad=pad, 
                                           filters=first_filters * 8,
                                           kernel_size=4,
                                           strides=2,
                                           use_bias=True,
                                           is_generator=False,
                                           name="d_conv2d_4")
            self.downsample_5 = Downsample(pad=pad, 
                                           filters=first_filters * 16,
                                           kernel_size=4,
                                           strides=2,
                                           use_bias=True,
                                           is_generator=False,
                                           name="d_conv2d_5")
            enc_last_filters = first_filters * 32

        self.downsample_last = Downsample(pad=pad, 
                                          filters=enc_last_filters,
                                          kernel_size=4,
                                          strides=1,
                                          use_bias=True,
                                          is_generator=False,
                                          name="d_downsample_last")
        # For Discriminator part
        self.cam = CAM(enc_last_filters, is_generator=False, name="d_CAM")

        # For Classifier part
        self.classifier = Downsample(pad=pad, 
                                     filters=1, 
                                     kernel_size=4, 
                                     strides=1, 
                                     use_bias=False,
                                     is_generator=False,
                                     act="",
                                     name="classifier")
        if self.use_mp:
            # This linear activation is used to make the dtype back to tf.float32 for Mixed Precision
            self.cast_last_output = tf.keras.layers.Activation("linear", dtype="float32")
        
    def call(self, inputs):
        x = self.downsample_1(inputs)
        x = self.downsample_2(x)
        x = self.downsample_3(x)
        if self.is_global:
            x = self.downsample_4(x)
            x = self.downsample_5(x)
        x = self.downsample_last(x)

        x, cam_logit, heatmap = self.cam(x)

        out = self.classifier(x)

        if self.use_mp:
            out = self.cast_last_output(out)
            cam_logit = self.cast_last_output(cam_logit)
            heatmap = self.cast_last_output(heatmap)

        return out, cam_logit, heatmap

    def summary(self):
        x = tf.keras.layers.Input(shape=(self.img_size, self.img_size, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name=self.name)
        return model.summary()


def build_models():
    gen = Generator()
    local_disc = Discriminator(is_global=False, name="local_discriminator")
    global_disc = Discriminator(is_global=True, name="global_discriminator")
    print(gen.summary())
    print(local_disc.summary())
    print(global_disc.summary())
    
    return gen, local_disc, global_disc


if __name__ == "__main__":
    gen_test, local_disc_test, global_disc_test = build_models()