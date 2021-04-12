import os
import math

from tqdm import tqdm

import tensorflow as tf


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.
def _bytes_feature(value):
    """ Returns a byte_list from a string / byte. """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """ Returns a float_list from a float / double. """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """ Returns an int64_list from a bool / enum / int / uint. """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(list_val):
    features = [tf.train.Feature(
        float_list=tf.train.FloatList(value=[value]) for val in list_val)]
    return tf.train.FeatureList(feature=features)


def convert_data_to_tfrecords(imgs, num_split, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("Start converting data into TFRecords.\n")
    num_data = len(imgs)
    num_per_shard = math.ceil(num_data / num_split)

    for shard_id in tqdm(range(num_split)):
        tfr_name = os.path.join(out_dir,
                                "selfie2anime-{:02d}-of-{:02d}.tfrecord".format(shard_id,
                                                                                num_split))
        with tf.io.TFRcordWriter(tfr_name) as writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_data)
            for i in range(start_idx, end_idx):
                example = image_example(imgs[i])
                writer.write(example.SerializeToString())
    
    print("Converting data into TFRecords is done!")


def image_example(img_path):
    _, ext = os.path.splitext(img_path)
    ext_lower = ext.lower()
    #img_string = open(img_path, "rb").read()
    img_string = tf.io.read_file(img_path)
    if "jpg" in ext_lower or "jpeg" in ext_lower:
        height, width, channel = tf.io.decode_jpeg(img_string).shape
    elif "png" in ext_lower:
        height, width, channel = tf.io.decode_png(img_string).shape

    feature = {
        "image": _bytes_feature(img_string),
        "height": _int64_feature(height),
        "width": _int64_feature(width),
        "channel": _int64_feature(channel)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))