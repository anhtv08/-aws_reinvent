import unittest
from ai_machine_learning.utils import add_numbers

class TestAddNumbers(unittest.TestCase):
    def test_add_positive_integers(self):
        self.assertEqual(add_numbers(2, 3), 5)

    def test_add_negative_integers(self):
        self.assertEqual(add_numbers(-1, 4), 3)

    def test_add_zeros(self):
        self.assertEqual(add_numbers(0, 0), 0)

    def test_add_floats(self):
        self.assertEqual(add_numbers(2.5, 3.7), 6.2)

if __name__ == '__main__':
    unittest.main()
