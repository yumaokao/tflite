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


def drop_dropouts(graph_def):
    dropouts = find_dropouts(graph_def.node)
    newnodes = graph_def.node[:]

    for k in dropouts:
        # find dropout nodes
        dpnodes = filter(lambda n: n.name.startswith(k), newnodes)
        divs = filter(lambda n: n.name == '{}/div'.format(k), dpnodes)
        if len(divs) != 1:
            raise ValueError("There should bw exactly one '{}/div' node".format(k))
        prevnames = filter(lambda i: not i.split('/')[-1].startswith('Placeholder'), divs[0].input)
        if len(prevnames) != 1:
            raise ValueError("There should bw exactly one node previosly attached to '{}'".format(k))
        placeholders = filter(lambda i: i.split('/')[-1].startswith('Placeholder'), divs[0].input)
        if len(placeholders) != 1:
            raise ValueError("There should bw exactly one Placeholder previosly attached to '{}'".format(k))

        # modify newnodes
        newnodes = filter(lambda n: not n.name.startswith(k), newnodes)
        newnodes = filter(lambda n: n.name != placeholders[0], newnodes)
        # attach prev to next nodes
        for n in newnodes:
            for i, iname in enumerate(n.input):
                if iname == '{}/mul'.format(k):
                    n.input[i] = prevnames[0]

    # output graph
    # display_nodes(newnodes)
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(newnodes)
    return output_graph


def main():
    parser = argparse.ArgumentParser(description='drop dropouts')
    parser.add_argument('frozen_pbs', nargs='+', help='Frozen graph files (.pb)')
    parser.add_argument('-v', '--version', action='version',
                        version=VERSION, help='show version infomation')
    args = parser.parse_args()

    for pb in args.frozen_pbs:
        npb = '{}-nodropout{}'.format(*os.path.splitext(pb))
        with tf.gfile.GFile(pb, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        n_graph_def = drop_dropouts(graph_def)
        with tf.gfile.GFile(npb, "wb") as f:
            f.write(n_graph_def.SerializeToString())


if __name__ == '__main__':
  main()
