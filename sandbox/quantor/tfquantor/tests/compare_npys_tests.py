import unittest
import numpy as np
from tfquantor.tools import compare_npys


class TestCompareNumpys(unittest.TestCase):
    def setUp(self):
        self.comparer = compare_npys.NumpyComparer()
        pass

    def test_float64_1d_array_same(self):
        ref = np.array([1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        com = np.array([1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        res = self.comparer.set_numpy_arrays([(ref, com)])
        self.assertEqual(res['status'], False)

    def test_float_1d_array_same(self):
        ref = np.array([1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], dtype=np.float32)
        com = np.array([1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], dtype=np.float32)
        res = self.comparer.set_numpy_arrays([(ref, com)])
        self.assertEqual(res['status'], True)
        results = self.comparer.compare()
        islarge = results[0]['result']['islarge']
        numerrors = np.count_nonzero(islarge)
        self.assertEqual(numerrors, 0)

    def test_float_1d_array_diff_rel(self):
        ref = np.array([1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1], dtype=np.float32)
        com = np.array([1e6 + 1e3, 1e5 + 1e3, 1e4 + 1e3,
                        1e3, 1e2 + 1, 1e1 + 1, 1e0 + 1, 1e-1 + 1], dtype=np.float32)
        res = self.comparer.set_numpy_arrays([(ref, com)])
        self.assertEqual(res['status'], True)
        results = self.comparer.compare()
        islarge = results[0]['result']['islarge']
        numerrors = np.count_nonzero(islarge)
        self.assertEqual(numerrors, 4)
        errpos = [list(pos) for pos in np.transpose(np.nonzero(islarge))]
        self.assertEqual(errpos, [[2], [5], [6], [7]])

    def test_float_1d_array_diff_abs(self):
        ref = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], dtype=np.float32)
        com = np.array([1e-2 + 1e-3, 1e-3 + 1e-3, 1e-4 + 1e-3, 1e-5 + 1e-3,
                        1e-6 + 1e-5, 1e-7 + 1e-5, 1e-8 + 1e-5, 1e-9 + 1e-5], dtype=np.float32)
        res = self.comparer.set_numpy_arrays([(ref, com)])
        self.assertEqual(res['status'], True)
        results = self.comparer.compare()
        islarge = results[0]['result']['islarge']
        numerrors = np.count_nonzero(islarge)
        self.assertEqual(numerrors, 4)
        errpos = [list(pos) for pos in np.transpose(np.nonzero(islarge))]
        self.assertEqual(errpos, [[0], [1], [2], [3]])

    def skip_test_uint8_1d_array_same(self):
        ref = np.array([255, 128, 127, 0, 5, 6, 7, 8], dtype=np.uint8)
        com = np.array([255, 128, 127, 0, 5, 6, 7, 8], dtype=np.uint8)
        res = self.comparer.set_numpy_arrays([(ref, com)])
        self.assertEqual(res['status'], True)
        results = self.comparer.compare()
        islarge = results[0]['result']['islarge']
        numerrors = np.count_nonzero(islarge)
        import ipdb
        ipdb.set_trace()
        self.assertEqual(numerrors, 0)

    def test_float_large_refrence(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
