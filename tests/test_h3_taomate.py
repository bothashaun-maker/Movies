"""Experimental preset isolation, shipped settings and download selection."""
import json
from pathlib import Path
import tempfile
import unittest
from h3_notebook_helpers import notebook_stages

ROOT=Path(__file__).resolve().parents[1]

class TaoMate(unittest.TestCase):
    def test_shipped_experiment_selects_only_matching_assets(self):
        nb=json.loads((ROOT/'minimax_h3_taomate3_experimental.ipynb').read_text())
        stages=notebook_stages(nb)
        with tempfile.TemporaryDirectory() as tmp:
            scope={}
            for name in ('settings','catalog','workflow'):
                source=stages[name].replace("Path('/kaggle/working/minimax_h3_dual_t4')",f'Path({tmp!r})')
                exec(source,scope)
            self.assertEqual(scope['SAMPLING_PRESET'],'taomate3')
            self.assertEqual(scope['STEPS'],3)
            selected=scope['selected_model_assets']()
            names=[a['file'] for a in selected]
            self.assertEqual(len(names),5)
            self.assertTrue(any('taomate' in n for n in names))
            self.assertTrue(any('video_vae_int8' in n for n in names))
            self.assertFalse(any('4step' in n or 'video_vae_fp16' in n for n in names))
            sample=scope['sampling_graph']('test')
            self.assertEqual(sample['11']['class_type'],'ManualSigmas')
            self.assertEqual(sample['9']['inputs']['model'],['5',0])
            self.assertNotIn('6',sample)
            scope['SAMPLING_PRESET']='turbo4';scope['VIDEO_VAE']='fp16'
            names=[a['file'] for a in scope['selected_model_assets']()]
            self.assertEqual(len(names),5)
            self.assertFalse(any('taomate' in n or 'video_vae_int8' in n for n in names))
