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


def rename_nodes(graph_def, renames, reids):
    newnodes = graph_def.node[:]
    # display_nodes(newnodes)
    if renames is not None:
        for r in renames:
            rp = r.split(',')
            for n in newnodes:
                if n.op == rp[0]:
                    n.op = rp[1]

    if reids is not None:
        for r in reids:
            rp = r.split(',')
            newnodes[int(rp[0])].op = rp[1]

    # output graph
    display_nodes(newnodes)
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(newnodes)
    return output_graph


def main():
    parser = argparse.ArgumentParser(description='rename ops')
    parser.add_argument('frozen_pbs', nargs='+', help='Frozen graph files (.pb)')
    parser.add_argument('-n', '--rename_by_names', nargs='+',
                        help='Ops to be renamed (ORIGIN_OP_NAME,NEW_OP_NAME)')
    parser.add_argument('-i', '--rename_by_ids', nargs='+',
                        help='Ops to be renamed (index,NEW_OP_NAME)')
    parser.add_argument('-v', '--version', action='version',
                        version=VERSION, help='show version infomation')
    args = parser.parse_args()
    if args.rename_by_names is None and args.rename_by_ids is None:
        # if not provide any rename rules, just display_nodes
        for pb in args.frozen_pbs:
            print('display all nodes of {}'.format(pb))
            with tf.gfile.GFile(pb, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                display_nodes(graph_def.node)
        sys.exit(0)

    if args.rename_by_names is not None:
        if len(filter(lambda r: len(r.split(',')) != 2, args.rename_by_names)):
            parser.print_help()

    if args.rename_by_ids is not None:
        if len(filter(lambda r: len(r.split(',')) != 2, args.rename_by_ids)):
            parser.print_help()
            sys.exit(1)

    for pb in args.frozen_pbs:
        npb = '{}-custom{}'.format(*os.path.splitext(pb))
        with tf.gfile.GFile(pb, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        n_graph_def = rename_nodes(graph_def, args.rename_by_names, args.rename_by_ids)
        with tf.gfile.GFile(npb, "wb") as f:
            f.write(n_graph_def.SerializeToString())


if __name__ == '__main__':
  main()
