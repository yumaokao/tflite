import sys
import time
import logging
import argparse
import numpy as np
from dispnet import DispNet
from util import init_logger, ft3d_filenames
import tensorflow as tf

INPUT_SIZE = (384, 768, 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest="dataset_path", required=True, type=str,
            metavar="FILE", help='path to FlyingThings3D dataset')
    parser.add_argument("--ckpt", dest="checkpoint_path", required=True, type=str,
            metavar="FILE", help='model checkpoint path')
    parser.add_argument("--log_step", dest="log_step", type=int, default=100,
            help='log step size')
    parser.add_argument("--batch_size", dest="batch_size", default=4, type=int,
                        help='batch size')
    parser.add_argument("--n_steps", dest="n_steps", type=int, default=None,
            help='test steps')
    parser.add_argument('--use_corr', action='store_true', default=False)

    args = parser.parse_args()

    ft3d_dataset = ft3d_filenames(args.dataset_path)

    tf.logging.set_verbosity(tf.logging.ERROR)
    dispnet = DispNet(mode="test", ckpt_path=args.checkpoint_path, dataset=ft3d_dataset,
                      input_size=INPUT_SIZE, batch_size=args.batch_size, is_corr=args.use_corr)

    ckpt = tf.train.latest_checkpoint(args.checkpoint_path)
    if not ckpt:
        logging.error("no checkpoint in provided path found!")
        sys.exit()
    init_logger(args.checkpoint_path)
    log_step = args.log_step
    if args.n_steps is None:
        N_test = len(ft3d_dataset["TEST"])
    else:
        N_test = args.n_steps

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(graph=dispnet.graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(dispnet.init)
        logging.debug("initialized")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        logging.debug("queue runners started")
        try:
            feed_dict = {}
            logging.info("Restoring from %s" % ckpt)
            dispnet.saver.restore(sess=sess, save_path=ckpt)
            feed_dict[dispnet.loss_weights] = np.zeros((6))
            test_err = 0
            start = time.time()
            for i in range(N_test / args.batch_size):
                cur_step = i + 1
                err = sess.run([dispnet.error], feed_dict=feed_dict)
                test_err += err[0]
                if cur_step % log_step == 0:
                    logging.debug("iter: %d, average forward pass time: %f, error %f" %
                                  (cur_step, ((time.time() - start) / float(log_step)),
                                   test_err / float(cur_step)))
                    start = time.time()
            test_err = test_err / float(N_test / args.batch_size)
            logging.info("Test error %f" % test_err)

        except tf.errors.OutOfRangeError:
            logging.INFO('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))

        finally:
            coord.request_stop()
            coord.join(threads)
            sess.close()
