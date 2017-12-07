from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from tensorflow.core.framework import graph_pb2
import tensorflow as tf

VERSION = "0.2.0"


def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('{} {} {}'.format(i, node.name, node.op))
        for i, n in enumerate(node.input):
            print('  - {} {}'.format(i, n))


def main():
    parser = argparse.ArgumentParser(description='save summaries')
    parser.add_argument('frozen_pbs', nargs='+', help='Frozen graph files (.pb)')
    parser.add_argument('-v', '--version', action='version',
                        version=VERSION, help='show version infomation')
    args = parser.parse_args()

    for pb in args.frozen_pbs:
        basename = os.path.splitext(os.path.basename(pb))[0]
        dirname = os.path.dirname(pb)
        summarydir = os.path.join(dirname, 'summary', basename)

        with tf.gfile.GFile(pb, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Session() as sess:
            g_in = tf.import_graph_def(graph_def)

        summary_writer = tf.summary.FileWriter(summarydir)
        summary_writer.add_graph(sess.graph)


if __name__ == '__main__':
  main()
