import unittest
from utils import wordSplit


class TestUtils(unittest.TestCase):

    def test_wordSplit(self):
        words = ['ticket', 'service']
        self.assertListEqual(
            wordSplit('ticketservice', words),
            ['ticket', 'service']
        )

        self.assertListEqual(
            wordSplit('ticketinfoservice', words),
            ['ticket', 'info', 'service']
        )

        self.assertListEqual(
            wordSplit('stationservice', words),
            ['station', 'service']
        )

        self.assertListEqual(
            wordSplit('footicketbar', words),
            ['foo', 'ticket', 'bar']
        )


if __name__ == '__main__':
    unittest.main()
