from pandas.core.reshape.merge import merge
from preprocess import embedding, str_process
import unittest
from utils import hump2snake, mergeDict, wordSplit


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
        return

    def test_str_process(self):
        self.assertEqual(
            str_process('ts-preserve-service'),
            'ts/preserve/service'
        )

        self.assertEqual(
            str_process('/{POST}/api/v1/preserveservice/'),
            'post/api/v1/preserve/service'
        )

        self.assertEqual(
            str_process(
                'ts-contacts-service/{GET}/api/v1/contactservice/contacts/id/0'),
            'ts/contacts/service/get/api/v1/contact/service/contacts/id/0'
        )
        return

    # def test_embedding(self):
    #     print(embedding('start'))
    #     print(embedding('end'))
    #     print(embedding('start/end'))

    def test_mergeDict(self):
        self.assertDictEqual(
            mergeDict({}, {'a': 1}),
            {'a': 1}
        )

        self.assertDictEqual(
            mergeDict({'a': 1}, {'b': 2}),
            {'a': 1, 'b': 2}
        )

        self.assertDictEqual(
            mergeDict({'a': 1}, {'b': {'c': 3}}),
            {'a': 1, 'b': {'c': 3}}
        )


if __name__ == '__main__':
    unittest.main()
