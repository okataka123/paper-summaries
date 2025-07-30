import unittest
from implementation import binary_to_gray, gray_to_binary, decode_chromosome

class Test(unittest.TestCase):
    def test_binary_to_gray_9(self):
        self.assertEqual(binary_to_gray(9), 13) # 1001

    def test_gray_to_binary_8(self):
        self.assertEqual(binary_to_gray(8), 12)  # 8(dec) -> 1000(bin) -> 1100(gray) -> binだと思うとdecで12

    def test_gray_to_binary_15(self):
        self.assertEqual(binary_to_gray(15), 8) # 15(dec) -> 1111(bin) -> 1000(gray) -> binだと思うとdecで8

    # def test_decode_chromosome(self):
    #     pass


if __name__ == '__main__':
    unittest.main()