"""Deterministic tests: no scans, models, downloads or optional ML packages."""
import unittest

import cv2
import numpy as np

from src.detection import detect_rulings
from src.reconstruction import corridors, remove_rulings
from src.verification import residual_metrics, native_residual_metrics, verify_removal


class RestorationTests(unittest.TestCase):
    def setUp(self):
        self.shape = (240, 160)
        self.lines = dict(scale=(1, 1), work=(160, 240), h=[], v=[
            dict(x0=79, x1=82, center=80, y0=0, y1=240, score=1)])
        self.paper = np.full(self.shape, 247, np.uint8)
        x = np.arange(160)
        line = 190*np.exp(-((x-80)/2.2)**2)
        self.scan = np.rint(self.paper-line).astype(np.uint8)

    def test_grey_halo_removal_and_allowed_edits(self):
        result, edit, protect, zones, stats = remove_rulings(self.scan, self.lines)
        before, _ = residual_metrics(self.scan, self.lines, (160, 240))
        after, _ = residual_metrics(result, self.lines, (160, 240))
        self.assertGreater(before[0]['median_deficit'], 150)
        self.assertLessEqual(after[0]['median_deficit'], 2)
        self.assertTrue(native_residual_metrics(result, self.lines)[0]['pass'])
        self.assertTrue(np.array_equal(result[:, :70], self.scan[:, :70]))
        self.assertTrue(np.array_equal(result[edit == 0], self.scan[edit == 0]))

    def test_touching_glyph_and_marker_preserved(self):
        # A stroke enters the line; a margin marker is outside every corridor.
        self.scan[90:98, 65:90] = 20
        self.scan[120:126, 10:14] = 30
        result, edit, protect, zones, _ = remove_rulings(self.scan, self.lines)
        np.testing.assert_array_equal(result[90:98, 65:90], self.scan[90:98, 65:90])
        np.testing.assert_array_equal(result[:, 10:14], self.scan[:, 10:14])
        self.assertGreater(len(zones), 0)
        self.assertTrue(np.array_equal(result[protect > 0], self.scan[protect > 0]))

    def test_hollow_grey_outline_fails_with_empty_edit_mask(self):
        hollow = self.paper.copy()
        hollow[:, 76] = 218
        hollow[:, 84] = 218
        zero = np.zeros(self.shape, np.uint8)
        report, passed = verify_removal(self.scan, hollow, self.lines, zero, zero, [], (160, 240))
        checks = {c['check']: c['pass'] for c in report['checks']}
        self.assertFalse(passed)
        self.assertFalse(checks['independent_grey_residual'])
        self.assertFalse(checks['native_grey_residual'])

    def test_noop_fails_and_outside_edit_detected(self):
        zero = np.zeros(self.shape, np.uint8)
        report, _ = verify_removal(self.scan, self.scan, self.lines, zero, zero, [], (160, 240))
        self.assertFalse(next(c['pass'] for c in report['checks'] if c['check']=='not_a_noop'))
        result, edit, protect, _, stats = remove_rulings(self.scan, self.lines)
        result[100, 15] = 0
        report, _ = verify_removal(self.scan, result, self.lines, edit, protect, stats, (160, 240))
        self.assertFalse(next(c['pass'] for c in report['checks'] if c['check']=='outside_geometry_unchanged'))

    def test_horizontal_transpose(self):
        lines = dict(scale=(1, 1), v=[], h=[dict(y0=79,y1=82,center=80,x0=0,x1=240)])
        result, *_ = remove_rulings(self.scan.T.copy(), lines)
        self.assertLessEqual(native_residual_metrics(result, lines)[0]['median_deficit'], 2)

    def test_detector_nine_vertical_two_horizontal(self):
        g = np.full((2048, 1342), 247, np.uint8)
        for x in range(120, 1201, 135):
            cv2.line(g, (x, 250), (x, 1825), 20, 4)
        for y in (250, 1825):
            cv2.line(g, (120, y), (1200, y), 20, 6)
        lines = detect_rulings(g)
        self.assertEqual((len(lines['v']), len(lines['h'])), (9, 2))


if __name__ == '__main__':
    unittest.main()
