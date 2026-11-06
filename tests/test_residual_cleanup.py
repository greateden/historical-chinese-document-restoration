"""Residual pass regressions; actual production seam, no image/model fixtures."""
import unittest
import cv2
import numpy as np
from src.cleanup import cleanup_residuals
from src.reconstruction import corridors


class ResidualTests(unittest.TestCase):
    def setUp(self):
        self.paper=np.full((240,160),247,np.uint8)
        self.original=self.paper.copy(); self.original[30:211,79:83]=20
        self.lines=dict(scale=(1,1),v=[dict(x0=79,x1=82,y0=30,y1=211,center=80)],
                        h=[dict(x0=40,x1=120,y0=y,y1=y+4,center=y+2) for y in (30,207)])

    def test_frame_dots_reconstructed_from_local_paper(self):
        base=self.paper.copy()
        base[30:35,78:85]=20; base[207:212,78:85]=20
        result,records=cleanup_residuals(self.original,base,self.lines)
        self.assertEqual(sum(r['action']=='remove' for r in records),2)
        np.testing.assert_array_equal(result,self.paper)

    def test_broken_verticals_and_permitted_region(self):
        base=self.paper.copy(); base[65:100,79:83]=30; base[125:160,79:83]=30
        result,records=cleanup_residuals(self.original,base,self.lines)
        self.assertEqual(sum(r['action']=='remove' for r in records),2)
        self.assertTrue(np.all(result[65:160,79:83]==247))
        allowed=np.zeros_like(base)
        for _,(x0,x1,y0,y1) in corridors(self.lines,base.shape): allowed[y0:y1,x0:x1]=1
        self.assertFalse(np.any((result!=base)&(allowed==0)))

    def test_real_glyph_and_margin_marker(self):
        base=self.paper.copy(); base[75:120,76:79]=15
        base[70:75,65:79]=15; base[130:135,10:16]=15
        original=self.original.copy(); original[base<150]=base[base<150]
        result,records=cleanup_residuals(original,base,self.lines)
        np.testing.assert_array_equal(result,base)
        self.assertTrue(any(r['action']=='preserve' for r in records))

    def test_horizontal_one_two_three(self):
        # Separate horizontal glyph strokes, including strokes crossing centre.
        base=self.paper.copy()
        for y in (65,110,125,165,180,195): base[y:y+3,65:96]=20
        result,_=cleanup_residuals(self.original,base,self.lines)
        np.testing.assert_array_equal(result,base)

    def test_ambiguous_connected_intersection(self):
        base=self.paper.copy(); base[70:140,79:83]=20; base[95:100,65:97]=20
        result,records=cleanup_residuals(self.original,base,self.lines)
        np.testing.assert_array_equal(result,base)
        self.assertTrue(all(r['action']=='preserve' for r in records))

    def test_isolated_punctuation_not_removed(self):
        base=self.paper.copy(); base[110:114,80:83]=20
        result,records=cleanup_residuals(self.original,base,self.lines)
        np.testing.assert_array_equal(result,base)
        self.assertEqual(records[0]['action'],'preserve')

    def test_dark_flanks_block_reconstruction_and_report_preservation(self):
        base=self.paper.copy(); base[65:100,79:83]=20
        base[65:100,55:59]=170; base[65:100,102:106]=170
        result,records=cleanup_residuals(self.original,base,self.lines)
        np.testing.assert_array_equal(result,base)
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['action'],'preserve')
        self.assertEqual(records[0]['reason'],'no_safe_reconstruction_pixels')

    def test_optional_stage_disabled(self):
        base=self.paper.copy(); base[70:140,79:83]=20
        result,records=cleanup_residuals(self.original,base,self.lines,enabled=False)
        np.testing.assert_array_equal(result,base); self.assertEqual(records,[])

if __name__=='__main__': unittest.main()
