from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf

VERSION = "0.2.0"


def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('{} {} {}'.format(i, node.name, node.op))
        for i, n in enumerate(node.input):
            print('  - {} {}'.format(i, n))


def drop_dropouts(graph_def):
    display_nodes(graph_def.node)


def main():
    parser = argparse.ArgumentParser(description='drop dropouts')
    parser.add_argument('frozen_pbs', nargs='+', help='Frozen graph files (.pb)')
    parser.add_argument('-v', '--version', action='version',
                        version=VERSION, help='show version infomation')
    args = parser.parse_args()

    for pb in args.frozen_pbs:
        with tf.gfile.GFile(pb, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            drop_dropouts(graph_def)


if __name__ == '__main__':
  main()
