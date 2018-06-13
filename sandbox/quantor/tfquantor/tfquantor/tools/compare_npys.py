from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np

VERSION = "0.2.0"


def compare_npypair(npypair):
    # TODO: argument thresholds
    # TODO: uint8
    #   https://docs.python.org/2/library/stdtypes.html
    #   Floating point numbers are usually implemented using double in C;
    kRelativeThreshold = 1e-2
    kAbsoluteThreshold = 1e-4

    # diff
    reference = npypair[0]
    computed = npypair[1]

    # debug
    # computed[0][0][0][0] = 1.0
    # computed[0][2][3][4] = 0.5

    diff = np.abs(computed - reference)

    # prepare error thresholds
    errthrs = np.where(reference < kRelativeThreshold,
                kAbsoluteThreshold, reference * kRelativeThreshold)

    # is_large
    islarge = diff > errthrs

    return {'islarge': islarge, 'errthrs': errthrs}


def compare_numpy_arrays(npypairs):
    results = []
    for p in npypairs:
        results.append({'pair': p, 'result': compare_npypair(p['pair'])})
    return results


def load_numpy_arrays(fnpairs):
    # TODO: support npz
    return [{'fn': p[1], 'key': 'npy',
             'pair': (np.load(p[0]), np.load(p[1])) } for p in fnpairs]


def main():
    parser = argparse.ArgumentParser(description='compare numpy arrays')
    parser.add_argument('--references', '-r', nargs='+', help='Reference numpy arrays (.npy)')
    parser.add_argument('--computeds', '-c', nargs='+', help='Computed numpy arrays (.npy)')
    parser.add_argument('-v', '--verbose', action='count', default=1, help='Verbose, more details')
    parser.add_argument('-V', '--version', action='version',
                        version=VERSION, help='show version infomation')

    args = parser.parse_args()
    if args.references is None:
        parser.exit(status=2, message="Error: flag --references is missing\n")
    if args.computeds is None:
        parser.exit(status=2, message="Error: flag --computeds is missing\n")
    if len(args.references) != len(args.computeds):
        parser.exit(status=2, message="Error: please give references and computeds in pairs\n")

    npypairs = load_numpy_arrays(zip(args.references, args.computeds))
    results = compare_numpy_arrays(npypairs)

    # TODO: print detail
    if args.verbose > 1:
        for r in results:
            islarge = r['result']['islarge']
            errors = np.count_nonzero(islarge)
            fn = r['pair']['fn']
            key = r['pair']['key']
            ref = r['pair']['pair'][0]
            com = r['pair']['pair'][1]
            print('Errors({}) in {}:{}'.format(errors, fn, key))
            for errpos in np.transpose(np.nonzero(islarge)):
                if len(errpos) != len(com.shape):
                    parser.exit(status=2, message="Error: shape are not matched\n")
                pos = tuple(errpos)
                print('    output{} did not match [{}] vs reference [{}]'.format(
                    pos, com[pos], ref[pos]))
                if args.verbose < 3:
                    break

    # print result
    status = 0
    for r in results:
        islarge = r['result']['islarge']
        if np.count_nonzero(islarge) > 0:
            status = 2

    message = 'OK' if status == 0 else 'Failed'
    parser.exit(status=status, message='{}\n'.format(message))


if __name__ == '__main__':
  main()
