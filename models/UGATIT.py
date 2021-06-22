import os
import time
import datetime

from tqdm import tqdm
import tensorflow as tf

from models.layers import Generator, Discriminator
from models.losses import *
from data_handler.tfrecords import parse_tfrecords


class UGATIT(tf.keras.Model):

    def __init__(self, args, name="UGATIT"):
        super(UGATIT, self).__init__(name=name)
        self.num_iters = args.num_iters
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.decay_iter = args.decay_iter
        self.loss_freq = args.loss_freq
        self.eval_freq = args.eval_freq
        self.save_freq = args.save_freq
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
        self.tfrecord_dir = args.tfrecord_dir
        self.use_mp = args.use_mp
        self.use_light = args.use_light

    def build_model(self):
        """ Define Generators and Discriminators """
        self.genA2B = Generator(use_mp=self.use_mp, name="A2B_generator")
        self.genB2A = Generator(use_mp=self.use_mp, name="B2A_generator")
        self.global_disA = Discriminator(is_global=True, use_mp=self.use_mp, name="global_discriminatorA")
        self.global_disB = Discriminator(is_global=True, use_mp=self.use_mp, name="global_discriminatorB")
        self.local_disA = Discriminator(is_global=False, use_mp=self.use_mp, name="local_discriminatorA")
        self.local_disB = Discriminator(is_global=False, use_mp=self.use_mp, name="local_discriminatorB")

        """ Define losses """
        self.L1_loss = L1_loss
        self.MSE_loss = MSE_loss
        self.BCE_loss = BCE_loss

        """ Define optimizers """
        self.G_optim = tf.keras.optimizers.Adam(self.lr, 0.5, 0.999)
        self.D_optim = tf.keras.optimizers.Adam(self.lr, 0.5, 0.999)
        if self.use_mp:
            self.G_optim = tf.keras.mixed_precision.LossScaleOptimizer(self.G_optim)
            self.D_optim = tf.keras.mixed_precision.LossScaleOptimizer(self.D_optim)

    def train(self):
        self.build_model()
        self.set_checkpoint_manager()

        trainA_dataset = self.set_train_dataset("trainA")
        trainB_dataset = self.set_train_dataset("trainB")

        testA_dataset = self.set_test_dataset("testA")
        testB_dataset = self.set_test_dataset("testB")
        fixed_testA_list = []
        fixed_testB_list = []
        num_eval = 5
        for a, b in zip(testA_dataset.take(num_eval), testB_dataset.take(num_eval)):
            fixed_testA_list.append(a)
            fixed_testB_list.append(b)
        
        print("Starts training!")
        start_time = time.time()
            
        for real_A, real_B in tqdm(zip(trainA_dataset, trainB_dataset)):
            if self.ckpt.iteration < self.num_iters:
                self.ckpt.iteration.assign_add(1)

                if self.ckpt.iteration > self.decay_iter:
                    self.update_lr()

                with tf.GradientTape(persistent=True) as dis_tape:
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

                with tf.GradientTape(persistent=True) as gen_tape:
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

                if self.ckpt.iteration % self.loss_freq == 0:
                    time_elapsed = round(time.time() - start_time)
                    log = f'[{self.ckpt.iteration.numpy():,}/{self.num_iters:,} time: {time_elapsed}]'
                    print(log)

                if self.ckpt.iteration % self.eval_freq == 0:
                    genA_results, genB_results = self.predict_for_eval(fixed_testA_list, fixed_testB_list)
                    genA_results = (genA_results * 0.5 + 0.5) * 255
                    genB_results = (genB_results * 0.5 + 0.5) * 255
                    logdir = os.path.join(self.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                    # Create a file writer for the log directory
                    file_writer = tf.summary.create_file_writer(logdir)
                    # Using the file writer, log the reshaped image.
                    with file_writer.as_default():
                        tf.summary.image("Generated A", genA_results, max_outputs=num_eval, step=self.ckpt.iteration)
                        tf.summary.image("Generated B", genB_results, max_outputs=num_eval, step=self.ckpt.iteration)

                if self.ckpt.iteration % self.save_freq == 0:
                    self.save()

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

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                  self.ckpt_dir,
                                                  max_to_keep=5)

        # If a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint is restored!")

    def set_train_dataset(self, dir_name):
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        tfr = os.path.join(self.tfrecord_dir, dir_name, "*.tfrecord")
        train_dataset = tf.data.Dataset.list_files(tfr)
        train_dataset = train_dataset.interleave(tf.data.TFRecordDataset,
                                                 num_parallel_calls=AUTOTUNE,
                                                 deterministic=False)
        train_dataset = train_dataset.map(parse_tfrecords)
        train_dataset = train_dataset.map(self.preprocess_for_training,
                                          num_parallel_calls=AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size=self.batch_size, 
                                            drop_remainder=True)
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        return train_dataset

    def set_test_dataset(self, dir_name):
        tfr = os.path.join(self.tfrecord_dir, dir_name, "*.tfrecord")
        test_dataset = tf.data.Dataset.list_files(tfr)
        test_dataset = test_dataset.interleave(tf.data.TFRecordDataset,
                                                 deterministic=True)
        test_dataset = test_dataset.map(parse_tfrecords)
        test_dataset = test_dataset.map(self.preprocess_for_testing)
        test_dataset = test_dataset.batch(batch_size=self.batch_size)
        return test_dataset

    def preprocess_for_training(self, img):
        img = tf.cast(img, tf.float32)
        img = img / 255.0
        img = tf.image.random_flip_left_right(img)
        img = tf.image.resize(img, [self.img_size + 30, self.img_size + 30])
        img = tf.image.random_crop(img, size=[self.img_size, self.img_size, 3])
        img = 2 * img - 1
        return img

    def preprocess_for_testing(self, img):
        img = tf.cast(img, tf.float32)
        img = img / 255.0
        img = tf.image.resize(img, [self.img_size, self.img_size])
        img = 2 * img - 1
        return img

    @tf.function
    def update_lr(self, dtype=tf.float32):
        float_num_iters = tf.cast(self.num_iters, dtype)
        float_curr_iter = tf.cast(self.ckpt.iteration, dtype)
        float_decay_iter = tf.cast(self.decay_iter, dtype)
        self.lr = self.init_lr * (float_num_iters - float_curr_iter) / (float_num_iters - float_decay_iter)

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
        self.G_global_A_cam_adv_loss = self.MSE_loss(tf.ones_like(self.fake_global_A_cam_logit),
                                                     self.fake_global_A_cam_logit)
        self.G_local_A_adv_loss = self.MSE_loss(tf.ones_like(self.fake_local_A_logit),
                                                self.fake_local_A_logit)
        self.G_local_A_cam_adv_loss = self.MSE_loss(tf.ones_like(self.fake_local_A_cam_logit),
                                                self.fake_local_A_cam_logit)
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
                                           self.G_local_A_cam_adv_loss) \
                        + self.lambda_cyc * self.G_A_recon_loss \
                        + self.lambda_id * self.G_A_id_loss \
                        + self.lambda_cam * self.G_A_cam_loss
        self.G_B_loss = self.lambda_adv * (self.G_global_B_adv_loss + \
                                           self.G_global_B_cam_adv_loss + \
                                           self.G_local_B_adv_loss + \
                                           self.G_local_B_cam_adv_loss) \
                        + self.lambda_cyc * self.G_B_recon_loss \
                        + self.lambda_id * self.G_B_id_loss \
                        + self.lambda_cam * self.G_B_cam_loss

    def predict_for_eval(self, testA_list, testB_list):
        genA_results = None
        genB_results = None
        for testA, testB in zip(testA_list, testB_list):
            testA = tf.reshape(testA, [-1, self.img_size, self.img_size, 3])
            testB = tf.reshape(testB, [-1, self.img_size, self.img_size, 3])
            B_result, _, _ = self.genA2B(testA)
            A_result, _, _ = self.genB2A(testB)
            genA_inp_out_concat = tf.concat([testB, A_result], 0)
            genB_inp_out_concat = tf.concat([testA, B_result], 0)
            
            if genA_results is None:
                genA_results = genA_inp_out_concat
                genB_results = genB_inp_out_concat
            else:
                genA_results = tf.concat([genA_results, genA_inp_out_concat], 0)
                genB_results = tf.concat([genB_results, genB_inp_out_concat], 0)

        return genA_results, genB_results

    def save(self):
        ckpt_save_path = self.ckpt_manager.save()
        print(f"Saving a checkpoint for iter {self.ckpt.iteration.numpy()} at {ckpt_save_path}")