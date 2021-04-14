import os
import sys
import math
from glob import glob

from tqdm import tqdm
from absl import app
from absl import flags

import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", "dataset/", "dir path of the input dataset")
flags.DEFINE_string("output_dir", "dataset/tfrecords", "dir path to output tfrecords")


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.
def _bytes_feature(value):
    """ Returns a byte_list from a string / byte. """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """ Returns an int64_list from a bool / enum / int / uint. """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_data_to_tfrecords(imgs, num_split, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("Start converting data into TFRecords for {}.\n".format(out_dir))
    num_data = len(imgs)
    num_per_shard = math.ceil(num_data / num_split)

    for shard_id in tqdm(range(num_split)):
        tfr_name = os.path.join(out_dir,
                                "selfie2anime-{:02d}-of-{:02d}.tfrecord".format(shard_id,
                                                                                num_split-1))
        with tf.io.TFRecordWriter(tfr_name) as writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_data)
            for i in range(start_idx, end_idx):
                example = image_example(imgs[i])
                writer.write(example.SerializeToString())
    
    print("Converting data into TFRecords is done for {}!\n".format(out_dir))


def image_example(img_path):
    _, ext = os.path.splitext(img_path)
    ext_lower = ext.lower()
    #img_string = open(img_path, "rb").read()
    img_string = tf.io.read_file(img_path)
    if "jpg" in ext_lower or "jpeg" in ext_lower:
        height, width, channel = tf.io.decode_jpeg(img_string).shape
        ext = 0
    elif "png" in ext_lower:
        height, width, channel = tf.io.decode_png(img_string).shape
        ext = 1

    feature = {
        "image": _bytes_feature(img_string),
        "height": _int64_feature(height),
        "width": _int64_feature(width),
        "channel": _int64_feature(channel),
        # As for ext, 0: "jpg", 1: "png"
        "ext": _int64_feature(ext)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_tf_records(input_dir, output_dir):
    dir_list = glob(os.path.join(input_dir, "*"))
    assert dir_list != [], "trainA, trainB, testA, testB must be placed before executing the program!"
    for dpath in dir_list:
        dname = dpath.split("/")[-1]
        if dname == "tfrecords":
            continue
        to_dir = os.path.join(output_dir, dname)
        os.makedirs(to_dir, exist_ok=True)
        img_list = glob(os.path.join(dpath, "*"))
        convert_data_to_tfrecords(img_list, 1, to_dir)


def parse_tfrecords_test(example_proto):
    # Parse the input tf.train.Example protocol buffer using the dictionary below
    image_feature_discription = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channel": tf.io.FixedLenFeature([], tf.int64),
         "ext": tf.io.FixedLenFeature([], tf.int64)
    }

    return tf.io.parse_single_example(example_proto, image_feature_discription)


def parse_tfrecords(example_proto):
    # Parse the input tf.train.Example protocol buffer using the dictionary below
    image_feature_discription = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channel": tf.io.FixedLenFeature([], tf.int64),
         "ext": tf.io.FixedLenFeature([], tf.int64)
    }

    image_features =  tf.io.parse_single_example(example_proto, image_feature_discription)
    if image_features["ext"] == 0:
        image = tf.io.decode_jpeg(image_features["image"])
    else:
        image = tf.io.decode_png(image_features["image"])

    return image


def main(argv):
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    create_tf_records(FLAGS.input_dir, FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)