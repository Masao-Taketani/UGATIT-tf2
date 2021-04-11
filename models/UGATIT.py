from time import time

import tensorflow as tf

from layers import Generator, Discriminator
from losses import *


class UGATIT(tf.keras.Model):

    def __init__(self, args):
        self.num_iters = args.num_iters
        self.batch_size = args.batch_size
        self.decay_iter = args.decay_iter
        self.lr = args.lr
        self.init_lr = args.lr
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

        self.set_checkpoint_manager()
        
        print("training start!")
        start_time = time()
            
        for real_A, real_B in dataset:
            if self.ckpt.iteration < self.num_iters:
                self.ckpt.iteration.assign_add(1)

                if self.ckpt.iteration > self.decay_iter:
                    self.update_lr()

                with tf.GradientTape() as dis_tape():
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

                global_dis_A_grads = dis_tape.gradient(self.D_A_loss, 
                                                       self.global_disA.trainable_variables)
                local_dis_A_grads = dis_tape.gradient(self.D_A_loss, 
                                                      self.local_disA.trainable_variables)
                global_dis_B_grads = dis_tape.gradient(self.D_B_loss, 
                                                       self.global_disB.trainable_variables)
                local_dis_B_grads = dis_tape.gradient(self.D_B_loss, 
                                                      self.local_disB.trainable_variables)
                self.D_optim.apply_gradients(zip(global_dis_A_grads, 
                                                 self.global_disA.trainable_variables))
                self.D_optim.apply_gradients(zip(local_dis_A_grads,
                                                 self.local_disA.trainable_variables))
                self.D_optim.apply_gradients(zip(global_dis_B_grads,
                                                 self.global_disB.trainable_variables))
                self.D_optim.apply_gradients(zip(local_dis_B_grads,
                                                 self.local_disB.trainable_variables))

                with tf.GradientTape() as gen_tape():
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
                    self.calculate_G_cam_losses(fake_B2A_cam_logit, 
                                                fake_A2A_cam_logit,
                                                fake_A2B_cam_logit,
                                                fake_B2B_cam_logit)
                    self.calculate_G_losses()
                
                genA2B_grads = gen_tape.gradient(self.G_B_loss, self.genA2B.trainable_variables)
                genB2A_grads = gen_tape.gradient(self.G_A_loss, self.genB2A.trainable_variables)
                self.G_optim.apply_gradients(zip(genA2B_grads, self.genA2B.trainable_variables))
                self.G_optim.apply_gradients(zip(genB2A_grads, self.genB2A.trainable_variables))

            else:
                break

        print("training is done!")

    def set_checkpoint_manager(self):
        self.ckpt = tf.train.Checkpoint(iteration=tf.Variable(0, dtype=tf.int64),
                                        genA2B=self.genA2B,
                                        genB2A=self.genB2A,
                                        global_disA=self.global_disA,
                                        global_disB=self.global_disB,
                                        local_disA=self.local_disA, 
                                        local_disB=self.local_disB,
                                        G_optim=self.G_optim,
                                        D_optim=self.D_optim)

        ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                  self.ckpt_dir,
                                                  max_to_keep=5)

        # If a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Latest checkpoint is restored!")

    def update_lr(self):
        self.lr = self.init_lr * (self.num_iters - self.ckpt.iteration) / (self.num_iters - self.decay_iter)

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
        self.G_local_A_cam_adv_loss = self.MSE_loss(tf.ones_like(self.fake_local_A_cam_logit,
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

    def calculate_G_cam_losses(self, 
                               fake_B2A_cam_logit, 
                               fake_A2A_cam_logit,
                               fake_A2B_cam_logit,
                               fake_B2B_cam_logit):

        self.G_A_cam_loss = self.BCE_loss(tf.ones_like(fake_B2A_cam_logit),
                                          fake_B2A_cam_logit) + \
                            self.BCE_loss(tf.zeros_like(fake_A2A_cam_logit),
                                          fake_A2A_cam_logit)
        self.G_B_cam_loss = self.BCE_loss(tf.ones_like(fake_A2B_cam_logit),
                                          fake_A2B_cam_logit) + \
                            self.BCE_loss(tf.zeros_like(fake_B2B_cam_logit),
                                          fake_B2B_cam_logit)

    def calculate_G_losses(self):
        self.G_A_loss = self.lambda_adv * (self.G_global_A_adv_loss + \
                                           self.G_global_A_cam_adv_loss + \
                                           self.G_local_A_adv_loss + \
                                           self.G_local_A_cam_adv_loss)
                        + self.lambda_cyc * G_A_recon_loss + \
                        + self.lambda_id * G_A_id_loss + \
                        + self.lambda_cam * G_A_cam_loss
        self.G_B_loss = self.lambda_adv * (self.G_global_B_adv_loss + \
                                           self.G_global_B_cam_adv_loss + \
                                           self.G_local_B_adv_loss + \
                                           self.G_local_B_cam_adv_loss)
                        + self.lambda_cyc * G_B_recon_loss + \
                        + self.lambda_id * G_B_id_loss + \
                        + self.lambda_cam * G_B_cam_loss
                        