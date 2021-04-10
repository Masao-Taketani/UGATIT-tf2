from time import time

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
        
        print("training start!")
        start_time = time()
        iteration = 0
        for real_A, real_B in dataset:
            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

            self.real_global_A_logit, self.real_global_A_cam_logit, _ = self.global_disA(real_A)
            self.real_local_A_logit, self.real_local_A_cam_logit, _ = self.local_disA(real_A)
            self.real_global_B_logit, self.real_global_B_cam_logit, _ = self.global_disB(real_B)
            self.real_local_B_logit, self.real_local_B_cam_logit, _ = self.local_disB(real_B)

            self.fake_global_A_logit, self.fake_global_A_cam_logit, _ = self.global_disA(fake_B2A)
            self.fake_local_A_logit, self.fake_local_A_cam_logit, _ = self.local_disA(fake_B2A)
            self.fake_global_B_logit, self.fake_global_B_cam_logit, _ = self.global_disB(fake_A2B)
            self.fake_local_B_logit, self.fake_local_B_cam_logit, _ = self.local_disB(fake_A2B)

            self.calculate_D_avd_losses()
            self.calculate_D_losses()

            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)
            
            self.fake_global_A_logit, self.fake_global_A_cam_logit, _ = self.global_disA(fake_B2A)
            self.fake_local_A_logit, self.fake_local_A_cam_logit, _ = self.local_disA(fake_B2A)
            self.fake_global_B_logit, self.fake_global_B_cam_logit, _ = self.global_disB(fake_A2B)
            self.fake_local_B_logit, self.fake_local_B_cam_logit, _ = self.local_disB(fake_A2B)
            
            self.calculate_G_adv_losses()
            self.calculate_G_recon_losses(real_A, real_B, fake_A2B2A, fake_B2A2B)
            self.calculate_G_id_losses(real_A, real_B, fake_A2A, fake_B2B)



    def calculate_D_avd_losses(self):

        self.D_global_A_adv_loss = self.MSE_loss(tf.ones_like(self.real_global_A_logit), 
                                                 self.real_global_A_logit) + \
                                   self.MSE_loss(tf.zeros_like(self.fake_global_A_cam_logit),
                                                 self.fake_global_A_logit)
        self.D_global_A_cam_adv_loss = self.MSE_loss(tf.ones_like(self.real_global_A_cam_logit),
                                                     self.real_global_A_cam_logit) + \
                                       self.MSE_loss(tf.zeros_like(self.fake_global_A_cam_logit),
                                                     self.fake_global_A_cam_logit)
        self.D_local_A_adv_loss = self.MSE_loss(tf.ones_like(self.real_local_A_logit),
                                            self.real_local_A_logit) + \
                                  self.MSE_loss(tf.zeros_like(self.fake_local_A_logit),
                                                self.fake_local_A_logit)
        self.D_local_A_cam_adv_loss = self.MSE_loss(tf.ones_like(self.real_local_A_cam_logit),
                                                    self.real_local_A_cam_logit) + \
                                      self.MSE_loss(tf.zeros_like(self.fake_local_A_cam_logit),
                                                    self.fake_local_A_cam_logit)
        self.D_global_B_adv_loss = self.MSE_loss(tf.ones_like(self.real_global_B_logit),
                                                 self.real_global_B_logit) + \
                                   self.MSE_loss(tf.zeros_like(self.fake_global_B_logit),
                                                 self.fake_global_B_logit)
        self.D_global_B_cam_adv_loss = self.MSE_loss(tf.ones_like(self.real_global_B_cam_logit),
                                                     self.real_global_B_cam_logit) + \
                                       self.MSE_loss(tf.zeros_like(self.fake_global_B_cam_logit),
                                                     self.fake_global_B_cam_logit)
        self.D_local_B_adv_loss = self.MSE_loss(tf.ones_like(self.real_local_B_logit),
                                                self.real_local_B_logit) + \
                                  self.MSE_loss(tf.zeros_like(self.fake_local_B_logit),
                                                self.fake_local_B_logit)
        self.D_local_B_cam_adv_loss = self.MSE_loss(tf.ones_like(self.real_local_B_cam_logit),
                                                    self.real_local_B_cam_logit) + \
                                      self.MSE_loss(tf.zeros_like(self.fake_local_B_cam_logit),
                                                    self.fake_local_B_cam_logit)
    
    def calculate_D_losses(self):
        self.D_A_loss = self.lambda_adv * (self.D_global_A_adv_loss + \
                                               self.D_global_A_cam_adv_loss + \
                                               self.D_local_A_adv_loss + \
                                               self.D_local_A_cam_adv_loss)
        self.D_B_loss = self.lambda_adv * (self.D_global_B_adv_loss + \
                                            self.D_global_B_cam_adv_loss + \
                                            self.D_local_B_adv_loss + \
                                            self.D_local_B_cam_adv_loss)

    def calculate_G_adv_losses(self):
        self.G_global_A_adv_loss = self.MSE_loss(tf.ones_like(self.fake_global_A_logit),
                                                 self.fake_global_A_logit)
        self.G_global_A_cam_adv_loss = self.MSE_loss(tf.ones_like(self.fake_global_A_cam_logit,
                                                     self.fake_global_A_cam_logit))
        self.G_local_A_adv_loss = self.MSE_loss(tf.ones_like(self.fake_local_A_logit,
                                                self.fake_local_A_logit))
        self.G_local_A_adv_loss = self.MSE_loss(tf.ones_like(self.fake_local_A_cam_logit,
                                                self.fake_local_A_cam_logit))
        self.G_global_B_adv_loss = self.MSE_loss(tf.ones_like(self.fake_global_B_logit),
                                                 self.fake_global_B_logit)
        self.G_global_B_cam_adv_loss = self.MSE_loss(tf.ones_like(self.fake_global_B_logit),
                                                     self.fake_global_B_logit)
        self.G_local_B_adv_loss = self.MSE_loss(tf.ones_like(self.fake_local_B_logit),
                                                self.fake_local_B_logit)
        self.G_local_B_cam_adv_loss = self.MSE_loss(tf.ones_like(self.fake_local_B_cam_logit),
                                                    self.fake_local_B_cam_logit)

    def calculate_G_recon_losses(self, real_A, real_B, fake_A2B2A, fake_B2A2B):
        self.G_A_recon_loss = self.L1_loss(real_A, fake_A2B2A)
        self.G_B_recon_loss = self.L1_loss(real_B, fake_B2A2B)
    
    def calculate_G_id_losses(self, real_A, real_B, fake_A2A, fake_B2B):
        self.G_A_id_loss = self.L1_loss(real_A, fake_A2A)
        self.G_B_id_loss = self.L1_loss(real_B, fake_B2B)

    def calculate_G_cam_losses(self, ):
        
    def calculate_G_losses(self):
