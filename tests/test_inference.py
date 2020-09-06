import unittest

import inference


class TestInference(unittest.TestCase):

    def setUp(self):
        self.opts = inference.Options('sample_data/input/pre')

    def test_passed_path(self):
        self.assertEqual('sample_data/input/pre', self.opts.in_pre_path)
        self.assertFalse(self.opts.is_vis)

    def test_default_path(self):
        self.assertEqual('input/post', self.opts.in_post_path)
 