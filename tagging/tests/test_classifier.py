# https://docs.python.org/3/library/unittest.html
from unittest import TestCase

from tagging.classifier import feature_dict


class TestFeatureDict(TestCase):

    def test_feature_dict(self):
        sent = 'El gato come pescado .'.split()

        fdict = {
            'lower': 'el',    # lower
            'isupper': False,  # isupper
            'istitle': True,   # istitle
            'isnumber': False,  # isdigit
            'p_lower': '<s>',
            'p_istitle': False,
            'p_isupper':False,
            'p_isnumber':False,
            'n_lower': 'gato',
            'n_istitle': False,
            'n_isupper':False,
            'n_isnumber':False,
        }

        self.assertEqual(feature_dict(sent, 0), fdict)
