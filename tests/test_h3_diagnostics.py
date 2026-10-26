import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class Diagnostics(unittest.TestCase):
    def test_reports_worker_logs_with_bounded_reads_and_no_external_symlinks(self):
        scope={'Path':Path}
        exec((ROOT/'notebooks/h3_complete/diagnostics.py').read_text(),scope)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';run=root/'runs/h3_fixture';run.mkdir(parents=True)
            (root/'browser-jobs').mkdir()
            (root/'browser-jobs/job.log').write_text('Sampling completed in 2.0 min.')
            (run/'sampling.log').write_text('x'*30000+'Both GPU shards loaded')
            (run/'decoding.log').write_text('Video decode on cuda:0')
            secret=Path(tmp)/'outside';secret.write_text('Do not include outside files')
            (run/'last-error.txt').symlink_to(secret)
            report=scope['render_diagnostics'](root)
            self.assertIn('Sampling completed in 2.0 min.',report)
            self.assertIn('Both GPU shards loaded',report)
            self.assertIn('Video decode on cuda:0',report)
            self.assertNotIn('Do not include outside files',report)
            self.assertLess(len(report),26000)
