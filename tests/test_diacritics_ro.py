import unittest

from ro_diacritics.diacritics_utils import *
from ro_diacritics import  restore_diacritics

class TestDiacriticsRO(unittest.TestCase):
    def setUp(self) -> None:
        ...

    def tearDown(self) -> None:
        ...

    def test_utils(self):
        str1 = "şir cu Şir, ţâr cu Ţâr, păr cu păr"

        str2 = [
            correct_diacritics(str1),
            remove_diacritics(str1),
            has_interesting_chars(str1),
            has_interesting_chars("bcdefghjklmnopqruvwxyz"),

        ]
        str2_ = [
            "șir cu Șir, țâr cu Țâr, păr cu păr",
            "sir cu Sir, tar cu Tar, par cu par",
            True,
            False,
        ]

        self.assertNotEqual(str1, str2)

        self.assertEqual(str2[0], str2_[0])
        self.assertEqual(str2[1], str2_[1])
        self.assertEqual(str2[2], str2_[2])
        self.assertEqual(str2[3], str2_[3])


    def test_model(self):
        str1 = 'as manca salam dar n-am'
        str2 = restore_diacritics(str1)
        str2_ = 'aș mânca salam dar n-am'

        self.assertEqual(str2, str2_)
