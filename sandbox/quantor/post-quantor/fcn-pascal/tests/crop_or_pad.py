# coding: utf-8
import tensorflow as tf
import numpy as np

# a = tf.ones([276, 362, 3])
# resized_image = tf.image.resize_image_with_crop_or_pad(a, 384, 384)

# a = tf.ones([200, 300, 3])
# resized_image = tf.image.resize_image_with_crop_or_pad(a, 300, 300)

a = tf.ones([200, 300])
resized_image = tf.image.resize_image_with_crop_or_pad(a, 300, 300)

with tf.Session() as sess:
    val = sess.run(resized_image)
    print(val)
    import ipdb
    ipdb.set_trace()
    print(val)
