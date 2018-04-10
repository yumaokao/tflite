
import time
import csv
import os
import datetime

import tensorflow as tf
import numpy as np
import skimage.io as io
import os
import sys
from PIL import Image
import set_paths


from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors

from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

from tf_image_segmentation.utils.training import get_valid_logits_and_labels

from tf_image_segmentation.utils.augmentation import (distort_randomly_image_color,
                                                      flip_randomly_left_right_image_with_annotation,
                                                      scale_randomly_image_with_annotation_with_fixed_size_output)

# from tf_image_segmentation.models.densenet_fcn import layers
# from tf_image_segmentation.models.densenet_fcn import densenet_fc
from tf_image_segmentation.models import unet

import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras_contrib.applications import densenet

# FLAGS = set_paths.FLAGS
# sys.path.append(FLAGS.tf_image_seg_dir)
# sys.path.append(FLAGS.slim_path)
# sys.path.append(FLAGS.slim_path + '/preprocessing')

# http://stackoverflow.com/a/5215012/99379


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def get_model(image_train_size=None, tensor=None, model_type=None, batch_size=None):
    print('creating ' + model_type + ' model')
    if model_type == 'densenet':
        model = densenet.DenseNetFCN(image_train_size, classes=number_of_classes,
                            upscaling_type='deconv', tensor=tensor, batch_size=batch_size, reduction=0.5)
    elif model_type == 'unet':
        model = unet.get_unet(
            image_train_size, number_of_classes, tensor=tensor)
    return model

if __name__ == '__main__':
    dirname = timeStamped('batch_densenet_fcn')


    FLAGS = set_paths.FLAGS
    # model_type options: densenet, unet
    model_type = 'densenet'
    out_dir = FLAGS.checkpoints_dir + dirname + '/'
    sess = tf.Session()
    K.set_session(sess)

    checkpoints_dir = FLAGS.checkpoints_dir
    log_dir = FLAGS.log_dir + model_type + "/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    slim = tf.contrib.slim
    batch_size = 1
    image_train_size = [384, 384, 3]
    image_2d_train_size = [image_train_size[0], image_train_size[1]]
    number_of_classes = 21
    # train_with_api options: keras, tf
    train_with_api = 'tf'
    number_of_epochs = 20

    #img_placeholder = tf.placeholder(tf.float32, shape=(None, image_train_size[0], image_train_size[1], image_train_size[2]))
    #label_placeholder = tf.placeholder(tf.float32, shape=(None, image_train_size[0], image_train_size[1],1))

    tfrecord_filename = 'pascal_augmented_train.tfrecords'
    pascal_voc_lut = pascal_segmentation_lut()
    class_labels = pascal_voc_lut.keys()

    densenet_checkpoint = FLAGS.save_dir + 'model_' + model_type + '_final.ckpt'

    filename_queue = tf.train.string_input_producer(
        [tfrecord_filename], num_epochs=10)

    image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(
        filename_queue)
    image = tf.cast(image, tf.float32)
    annotation = tf.cast(annotation, tf.float32)

    tfrecord_val_filename = 'pascal_augmented_val.tfrecords'

    filename_val_queue = tf.train.string_input_producer(
        [tfrecord_filename], num_epochs=1)

    val_image, val_annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(
        filename_val_queue)

    if train_with_api is 'tf':
        print('Running with tf training, initializing batches...')
        from keras.objectives import categorical_crossentropy
        #softmax = model.output
        #output_tensor = K.argmax(softmax)
        #output_tensor = model.output
        # Various data augmentation stages
        image, annotation = flip_randomly_left_right_image_with_annotation(
            image, annotation)

        # image = distort_randomly_image_color(image)

        resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(
            image, annotation, image_2d_train_size)

        resized_annotation = tf.squeeze(resized_annotation)

        image_batch, annotation_batch = tf.train.shuffle_batch([resized_image, resized_annotation],
                                                               batch_size=batch_size,
                                                               capacity=3000,
                                                               num_threads=2,
                                                               min_after_dequeue=1000)

        model = get_model(image_train_size=image_train_size,
                          model_type=model_type, tensor=image_batch, batch_size=batch_size)

        valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(annotation_batch_tensor=annotation_batch,
                                                                                           logits_batch_tensor=model.output,
                                                                                           class_labels=class_labels)

        # Assume that image_coords is a tensor of size [H, W, 2] representing the image
        # coordinates of each pixel.
        # Convert softmax to shape [N, H, W, C, 1]
        #softmax = tf.expand_dims(softmax, -1)
        # Convert image coords to shape [H, W, 1, 2]
        #image_coords = tf.expand_dims(image_coords, 2)
        # Multiply (with broadcasting) and reduce over image dimensions to get the result
        # of shape [N, C, 2]
        #spatial_soft_argmax = tf.reduce_sum(softmax * image_coords, reduction_indices=[1, 2])

        cross_entropies = K.categorical_crossentropy(
            valid_logits_batch_tensor, valid_labels_batch_tensor, from_logits=True)

        # Normalize the cross entropy -- the number of elements
        # is different during each step due to mask out regions
        # aka loss
        cross_entropy_sum = tf.reduce_mean(cross_entropies)

        #pred = tf.argmax(upsampled_logits_batch, dimension=3)

        #probabilities = tf.nn.softmax(upsampled_logits_batch)

        with tf.variable_scope("adam_vars"):
            train_step = tf.train.AdamOptimizer(
                learning_rate=1e-5).minimize(cross_entropy_sum)

        merged_summary_op = tf.summary.merge_all()

        summary_string_writer = tf.summary.FileWriter(log_dir)

        # Create the log folder if doesn't exist yet
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # The op for initializing the variables.
        local_vars_init_op = tf.local_variables_initializer()
        global_vars_init_op = tf.global_variables_initializer()

        combined_op = tf.group(local_vars_init_op, global_vars_init_op)

        # We need this to save only model variables and omit
        # optimization-related and other variables.
        model_variables = model.trainable_weights
        saver = tf.train.Saver(model_variables)

        sess.run(combined_op)
        # init_fn(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('starting training...')

        # 10 epochs
        for i in xrange(11127 * number_of_epochs):

            cross_entropy, summary_string, _ = sess.run([cross_entropy_sum,
                                                         merged_summary_op,
                                                         train_step],
                                                        # , model.inputs[0]:image_batch}#,img_placeholder:}
                                                        feed_dict={
                                                            K.learning_phase(): 1}
                                                        )

            print("Current loss: " + str(cross_entropy))

            summary_string_writer.add_summary(summary_string, i)

            if i % 11127 == 0:
                save_path = saver.save(
                    sess, FLAGS.save_dir + "model_" + model_type + "_epoch_" + str(i) + ".ckpt")
                print("Model saved in file: %s" % save_path)

        coord.request_stop()
        coord.join(threads)

        save_path = saver.save(sess, FLAGS.save_dir +
                               "model_" + model_type + "_final.ckpt")
        print("Model saved in file: %s" % save_path)

        summary_string_writer.close()

    if train_with_api is 'keras':
        print('Running with keras training, converting tensors to numpy...')

        # TODO(ahundt) remove conversion to numpy when tensors are directly
        # supported https://github.com/fchollet/keras/issues/5356
        image = image.eval(session=sess)
        annotation = annotation.eval(session=sess)
        val_image = val_image.eval(session=sess)
        val_annotation = val_annotation.eval(session=sess)

        print('completed converting tensors to numpy, compiling model and augmenting image data...')

        model.compile(loss="categorical_crossentropy", optimizer='adam')

        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(
            0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        early_stopper = EarlyStopping(
            monitor='val_acc', min_delta=0.001, patience=10)
        csv_logger = CSVLogger('resnet18_cifar10.csv')

        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(image)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        tensorboard = TensorBoard(
            log_dir=out_dir, histogram_freq=10, write_graph=True)
        csv = CSVLogger(out_dir + dirname + '.csv', separator=',', append=True)
        model_checkpoint = ModelCheckpoint(out_dir + 'weights.hdf5', monitor='val_loss',
                                           verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        callbacks = [lr_reducer, early_stopper, csv]

        print('augmenting image data initialized, training with fit_generator...')
        start_time = time.time()

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(image, annotation,
                                                   batch_size=batch_size),
                                      samples_per_epoch=image.shape[0],
                                      nb_epoch=nb_epoch,
                                      validation_data=(
                                          val_image, val_annotation),
                                      verbose=1, max_q_size=100,
                                      callbacks=callbacks)

        end_fit_time = time.time()
        average_time_per_epoch = (end_fit_time - start_time) / nb_epoch

        print('training complete, timing validation set prediction...')
        model.predict(val_image, batch_size=batch_size, verbose=1)

        end_predict_time = time.time()
        average_time_to_predict = (end_predict_time - end_fit_time) / nb_epoch

        results.append(
            (history, average_time_per_epoch, average_time_to_predict))
        print ('--------------------------------------------------------------------')
        print ('[run_name,batch_size,average_time_per_epoch,average_time_to_predict]')
        print ([dirname, batch_size, average_time_per_epoch, average_time_to_predict])
        print ('--------------------------------------------------------------------')

    # Close the Session when we're done.
    sess.close()
