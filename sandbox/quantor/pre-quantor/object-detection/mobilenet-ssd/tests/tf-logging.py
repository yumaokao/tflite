import tensorflow as tf
import logging


def main(unused_argv):
  logging.basicConfig(level=logging.INFO)
  logging.warning('YMK in tf logging test warning')
  logging.info('YMK in tf logging test info')


if __name__ == '__main__':
  tf.app.run()
