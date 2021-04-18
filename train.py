import os

from absl import app
from absl import flags

import tensorflow as tf

from models.UGATIT import UGATIT


FLAGS = flags.FLAGS

#flags.DEFINE_integer("num_epochs", 100, "number of epochs to train")
flags.DEFINE_integer("num_iters", 1000000, "number of iterations to train")
flags.DEFINE_integer("batch_size", 1, "batch size used to train")
flags.DEFINE_integer("img_size", 256, "the size of images used for training")
flags.DEFINE_integer("decay_iter", 500000, "iteration that starts decaying the learning rate")
flags.DEFINE_integer("loss_freq", 1000, "frequency to print loss log for every specified iterations")
flags.DEFINE_integer("eval_freq", 50000, "frequency to evaluate for every specified iterations")
flags.DEFINE_integer("save_freq", 50000, "frequency to save models for every specified iterations")
flags.DEFINE_float("lr", 0.0001, "learning rate to train")
flags.DEFINE_float("lambda_gp", 10.0, "lambda for the gradient penalty")
flags.DEFINE_float("lambda_adv", 1.0, "lambda for the adversarial loss")
flags.DEFINE_float("lambda_cyc", 10.0, "lambda for the cycle loss")
flags.DEFINE_float("lambda_id", 10.0, "lambda for the identity loss")
flags.DEFINE_float("lambda_cam", 1000.0, "lambda for the CAM loss")
flags.DEFINE_string("ckpt_dir", "ckpts", "dir to save checkpoints")
flags.DEFINE_string("eval_dir", "test_results", "dir to save evaluated results")
flags.DEFINE_string("logdir", "logs/", "dir to save logs")
flags.DEFINE_string("tfrecord_dir", "dataset/tfrecords/", "dir to load tftrecords")
flags.DEFINE_bool("use_mp", True, "whether to use mixed precision for training")
flags.DEFINE_bool("use_light", True, "whether to use light version of the generator")


def main(argv):
    if FLAGS.use_mp:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    os.makedirs(FLAGS.ckpt_dir, exist_ok=True)
    os.makedirs(FLAGS.eval_dir, exist_ok=True)
    os.makedirs(FLAGS.logdir, exist_ok=True)

    ugatit = UGATIT(FLAGS)
    ugatit.train()
    

if __name__ == "__main__":
    app.run(main)