from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np

VERSION = "0.3.0"


class NumpyComparer:
    '''
        Numpy Compare Class
    '''

    def __init__(self, fnpairs=[],
                 kRelativeThreshold=1e-2, kAbsoluteThreshold=1e-4):
        self.fnpairs = fnpairs
        # for float
        self.kRelativeThreshold = kRelativeThreshold
        self.kAbsoluteThreshold = kAbsoluteThreshold

    def _append_npyarrays(self, fn, ref, com):
        if ref.dtype != com.dtype:
            return {'status': False,
                    'message': 'reference data type != computed data type'}
        if ref.dtype not in [np.float32, np.uint8]:
            return {'status': False,
                    'message': 'reference data type is not flaot32 or uint8'}
        self.npypairs.append({'fn': fn, 'key': 'npy',
                              'dtype': ref.dtype, 'pair': (ref, com)})
        return {'status': True, 'message': 'OK'}

    def load_numpy_arrays(self):
        # TODO: support npz
        self.npypairs = []
        for p in self.fnpairs:
            reference = np.load(p[0])
            computed = np.load(p[1])
            ret = self._append_npyarrays(p[1], reference, computed)
            if ret['status'] is False:
                return ret
        return {'status': True, 'message': 'OK'}

    def set_numpy_arrays(self, npypairs):
        self.npypairs = []
        for p in npypairs:
            ret = self._append_npyarrays('set_numpy_arrays', p[0], p[1])
            if ret['status'] is False:
                return ret
        return {'status': True, 'message': 'OK'}

    def compare(self):
        results = []
        for p in self.npypairs:
            results.append({'pair': p,
                            'result': self.compare_npypair(p['pair'])})
        return results

    def compare_npypair(self, npypair):
        # TODO: argument thresholds
        # TODO: uint8
        reference = npypair[0]
        computed = npypair[1]

        #   https://docs.python.org/2/library/stdtypes.html
        #   Floating point numbers are usually implemented using double in C;
        kRelativeThreshold = 1e-2
        kAbsoluteThreshold = 1e-4

        # debug
        # computed[0][0][0][0] = 1.0
        # computed[0][2][3][4] = 0.5

        # diff
        diff = np.abs(computed - reference)

        # prepare error thresholds
        errthrs = np.where(reference < kRelativeThreshold,
                           kAbsoluteThreshold, reference * kRelativeThreshold)
        # is_large
        islarge = diff > errthrs
        return {'islarge': islarge, 'errthrs': errthrs}
    
    @staticmethod
    def show_results(results, verbose=1):
        if verbose > 1:
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
                    if verbose < 3:
                        break


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

    comparer = NumpyComparer(zip(args.references, args.computeds))
    result = comparer.load_numpy_arrays()
    if result['status'] is False:
        parser.exit(status=2, message="Error: {}\n".format(result['message']))
    results = comparer.compare()

    # print detail
    comparer.show_results(results, verbose=args.verbose)

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
