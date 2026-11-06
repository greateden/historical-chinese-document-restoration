"""Output safety and failure reporting at the public CLI boundary."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import restore


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'page.png'
        self.image.write_bytes(b'original bytes')

    def run_cli(self, *args):
        with redirect_stderr(StringIO()):
            return restore.main(list(args))

    def test_existing_directory_and_input_preserved(self):
        self.assertEqual(self.run_cli(str(self.image), '--outdir', str(self.root)), 2)
        self.assertEqual(self.image.read_bytes(), b'original bytes')

    def test_duplicate_stems_rejected_before_output(self):
        out = self.root / 'out'
        self.assertEqual(self.run_cli(str(self.image), str(self.image), '--outdir', str(out)), 2)
        self.assertFalse(out.exists())

    def test_missing_input_rejected_before_output(self):
        out = self.root / 'out'
        self.assertEqual(self.run_cli(str(self.root/'missing.png'), '--outdir', str(out)), 2)
        self.assertFalse(out.exists())

    def test_validation_failure_is_reported_without_claiming_success(self):
        out = self.root / 'out'
        with patch('restore.process', return_value={'passed': False, 'seconds': 0}):
            self.assertEqual(self.run_cli(str(self.image), '--outdir', str(out)), 1)
        self.assertIn('"passed": false', (out/'report.json').read_text())

    def test_failed_image_write_raises(self):
        with patch('restore.cv2.imwrite', return_value=False):
            with self.assertRaises(OSError):
                restore.write_image(self.root/'out.png', None)


if __name__ == '__main__':
    unittest.main()
