from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.core.framework import graph_pb2

import numpy as np
import tensorflow as tf

VERSION = "0.2.0"


def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('{} {} {}'.format(i, node.name, node.op))
        for i, n in enumerate(node.input):
            print('  - {} {}'.format(i, n))


def find_dropouts(nodes):
    nodes = filter(lambda n: len(n.name.split('/')) > 1, nodes)
    # find dropout/dropout/add Add
    dropns = filter(lambda n: n.name.split('/')[-2].startswith('dropout'), nodes)
    dropns = filter(lambda n: n.name.split('/')[-1] == 'add', dropns)
    drops = map(lambda n: '/'.join(n.name.split('/')[:-1]), dropns)
    # display_nodes(dropns)
    # print(drops)
    return drops


def rename_nodes(graph_def, renames):
    newnodes = graph_def.node[:]
    # display_nodes(newnodes)
    for r in renames:
        rp = r.split(',')
        for n in newnodes:
            if n.op == rp[0]:
                n.op = rp[1]

    # output graph
    display_nodes(newnodes)
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(newnodes)
    return output_graph


def main():
    parser = argparse.ArgumentParser(description='rename ops')
    parser.add_argument('frozen_pbs', nargs='+', help='Frozen graph files (.pb)')
    parser.add_argument('-r', '--renames', nargs='+',
                        help='Ops to be renamed (ORIGIN_OP,NEW_OP)')
    parser.add_argument('-v', '--version', action='version',
                        version=VERSION, help='show version infomation')
    args = parser.parse_args()
    if args.renames is None:
        parser.print_help()
        sys.exit(1)
    if len(filter(lambda r: len(r.split(',')) != 2, args.renames)):
        parser.print_help()
        sys.exit(1)

    for pb in args.frozen_pbs:
        npb = '{}-custom{}'.format(*os.path.splitext(pb))
        with tf.gfile.GFile(pb, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        n_graph_def = rename_nodes(graph_def, args.renames)
        with tf.gfile.GFile(npb, "wb") as f:
            f.write(n_graph_def.SerializeToString())


if __name__ == '__main__':
  main()
