import os

from absl import app
from absl import flags

import tensorflow as tf

from models.UGATIT import UGATIT


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_epochs", 100, "number of epochs to train")
flags.DEFINE_integer("num_iters", 10000, "number of iterations to train")
flags.DEFINE_integer("batch_size", 1, "batch size used to train")
flags.DEFINE_integer("decay_epoch", 50, "epoch that starts decaying the learning rate")
flags.DEFINE_float("lr", 0.0001, "learning rate to train")
flags.DEFINE_float("lambda_gp", 10.0, "lambda for the gradient penalty")
flags.DEFINE_float("lambda_adv", 1.0, "lambda for the adversarial loss")
flags.DEFINE_float("lambda_cyc", 10.0, "lambda for the cycle loss")
flags.DEFINE_float("lambda_id", 10.0, "lambda for the identity loss")
flags.DEFINE_float("lambda_cam", 1000.0, "lambda for the CAM loss")
flags.DEFINE_string("ckpt_dir", "ckpts", "dir to save checkpoints")
flags.DEFINE_string("eval_dir", "test_results", "dir to save evaluated results")
flags.DEFINE_string("logdir", "logs/", "dir to save logs")


def main(argv):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    os.makedirs(FLAGS.ckpt_dir, exist_ok=True)
    os.makedirs(FLAGS.eval_dir, exist_ok=True)
    os.makedirs(FLAGS.logdir, exist_ok=True)

    UGATIT(FLAGS)
    
    




if __name__ == "__main__":
    app.run(main)