import unittest

import numpy as np
from norfair.tracker import Detection

from run_algorithm_utils import (get_nr_of_zeros, get_most_dominant_color,
                                 create_mask,
                                 check_intersection,
                                 check_last_touchings_equal)


class TestRunAlgorithmUtils(unittest.TestCase):

    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detections = [Detection(points=np.array([[1, 1]])),
                           Detection(points=np.array([[2, 2]])),
                           Detection(points=np.array([[3, 3]]))]

    def test_get_nr_of_zeros(self):
        mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        image = np.zeros(self.frame.shape, dtype=np.uint8)
        image[100:200, 100:200] = [0, 0, 255]

        result = get_nr_of_zeros(mask, image)

        self.assertEqual(result, 0)

    def test_get_most_dominant_color(self):
        colors = ['red', 'blue', 'green', 'red', 'red']

        result = get_most_dominant_color(colors)

        self.assertEqual(result, 'red')

    def test_create_mask(self):
        result = create_mask(self.frame, self.detections)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.frame.shape[:2])
        self.assertEqual(result[69:200, 160:510].sum(), 0)

    def test_check_intersection(self):
        self.assertTrue(check_intersection(50, 50, 0, 100, 0, 100))
        self.assertTrue(check_intersection(50, 50, 20, 80, 20, 80))
        self.assertTrue(check_intersection(50, 50, 30, 70, 30, 70))

        self.assertFalse(check_intersection(50, 50, 60, 100, 60, 100))
        self.assertFalse(check_intersection(50, 50, 0, 40, 0, 40))
        self.assertFalse(check_intersection(50, 50, 30, 40, 30, 40))

    def test_check_last_touchings_equal(self):
        touchings_array = [1, 1, 1, 1, 1]
        self.assertTrue(check_last_touchings_equal(touchings_array))

        touchings_array = [1, 1, 1, 0, 1]
        self.assertFalse(check_last_touchings_equal(touchings_array))


if __name__ == '__main__':
    unittest.main()
