

import tensorflow as tf

from layers import Generator, Discriminator
from losses import *


class UGATIT(tf.keras.Model):

    def __init__(self, args):
        self.num_epochs = args.num_epochs
        self.num_iters = args.num_iters
        self.batch_size = args.batch_size
        self.decay_epoch = args.decay_epoch
        self.lr = args.lr
        self.lambda_gp = args.lambda_gp
        self.lambda_adv = args.lambda_adv
        self.lambda_cyc = args.lambda_cyc
        self.lambda_id = args.lambda_id
        self.lambda_cam = args.lambda_cam
        self.ckpt_dir = args.ckpt_dir
        self.eval_dir = args.eval_dir
        self.logdir = args.logdir

    def build_model(self):
        """ Define Generators and Discriminators """
        self.genA2B = Generator(name="A2B_generator")
        self.genB2A = Generator(name="B2A_generator")
        self.global_disA = Discriminator(is_global=True, name="global_discriminatorA")
        self.global_disB = Discriminator(is_global=True, name="global_discriminatorB")
        self.local_disA = Discriminator(is_global=False, name="local_discriminatorA")
        self.local_disB = Discriminator(is_global=False, name="local_discriminatorB")

        """ Define losses """
        self.L1_loss = L1_loss
        self.MSE_loss = MSE_loss
        self.BCE_loss = BCE_loss

        """ Define optimizers """
        self.G_optim = tf.keras.optimizers.Adam(self.lr, 0.5, 0.999)
        self.D_optim = tf.keras.optimizers.Adam(self.lr, 0.5, 0.999)

    def train(self):
        