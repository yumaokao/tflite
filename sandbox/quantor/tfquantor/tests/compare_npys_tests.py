import unittest
import numpy as np
from tfquantor.tools import compare_npys


class TestCompareNumpys(unittest.TestCase):
    def setUp(self):
        pass

    def test_float_1d_array_same(self):
        ref = np.array([1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        com = np.array([1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        results = compare_npys.compare_npypair((ref, com))
        numerrors = np.count_nonzero(results['islarge'])
        self.assertEqual(numerrors, 0)

    def test_float_large_refrence(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
