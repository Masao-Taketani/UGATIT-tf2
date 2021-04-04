import tensorflow as tf


 def L1_loss(x_real, x_rec):
     return tf.reduce_mean(tf.abs(x_real - x_rec))


def MSE_loss(true, pred):
    return tf.recude_mean(tf.keras.losses.MSE(true, pred))


def BCE_loss(true, pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return tf.recude_mean(bce(true, pred))