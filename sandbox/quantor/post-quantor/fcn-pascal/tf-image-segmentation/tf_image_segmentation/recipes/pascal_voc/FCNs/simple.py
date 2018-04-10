import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input
from keras.objectives import categorical_crossentropy
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.Session()
K.set_session(sess)

img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

inputs = Input(image_size)
x = Dense(128, activation='relu')(inputs)
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess.run(tf.global_variables_initializer())

# TODO remove default
with sess.as_default():
    for i in range(1000):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})
    print(loss.eval(feed_dict={img: mnist_data.test.images, labels: mnist_data.test.labels}))