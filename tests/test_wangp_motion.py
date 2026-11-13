"""CPU regression tests for failures that must stop before GPU allocation."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'notebooks'/'wan_gp_motion'
sys.path.insert(0, str(SOURCE))
import common
import guard
import launcher
import settings
from dataset import create_dataset
from patch import patch_wgp, patch_h3_dtype


def asset(name, contents, file=None):
    return {'name': name, 'file': file or name, 'bytes': len(contents),
            'sha256': hashlib.sha256(contents).hexdigest(), 'source': 'https://example.invalid/'+name}


class StorageTests(unittest.TestCase):
    def test_missing_models_stop_without_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, cache = root/'input', root/'cache'
            source.mkdir()
            (source/'one').write_bytes(b'one')
            with self.assertRaisesRegex(RuntimeError, 'Missing files'):
                common.stage_models({'assets': [asset('one', b'one'), asset('two', b'two')]}, source, cache)
            self.assertFalse((cache/'one').exists())

    def test_copy_and_nested_layout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root/'input'/'dataset'
            source.mkdir(parents=True)
            (source/'weights').write_bytes(b'weights')
            record = asset('weights', b'weights', 'encoder/weights')
            with patch.object(common.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=8*common.GIB)):
                common.stage_models({'assets': [record]}, root/'input', root/'models')
            self.assertEqual((root/'models/encoder/weights').read_bytes(), b'weights')
            self.assertEqual((source/'weights').read_bytes(), b'weights')

    def test_corruption_stops_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'input').mkdir()
            (root/'input'/'model').write_bytes(b'wrong')
            record = asset('model', b'right')
            with patch.object(common.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=8*common.GIB)):
                with self.assertRaisesRegex(RuntimeError, 'failed verification'):
                    common.stage_models({'assets': [record]}, root/'input', root/'models')
            self.assertEqual((root/'input/model').read_bytes(), b'wrong')
            self.assertFalse((root/'models/model').exists())

    def test_low_space_rejects_before_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'input').mkdir()
            (root/'input/model').write_bytes(b'valid')
            with patch.object(common.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=1)):
                with self.assertRaisesRegex(RuntimeError, 'Not enough local disk'):
                    common.stage_models({'assets': [asset('model', b'valid')]}, root/'input', root/'models')
            self.assertFalse((root/'models/model').exists())

    def test_cache_counts_toward_required_capacity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'kit/models').mkdir(parents=True)
            (root/'kit/models/m').write_bytes(b'1234')
            with patch.object(common.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=5*common.GIB)):
                self.assertEqual(common.choose_scratch(5*common.GIB+4, [root], 'kit', [('models/m', 4)]), root/'kit')


class ProcessTests(unittest.TestCase):
    def test_deadline_kills_owned_process(self):
        with tempfile.TemporaryDirectory() as d:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                common.bounded([sys.executable, '-c', 'import time; time.sleep(30)'], Path(d)/'child.log', .2)
            self.assertLess(time.monotonic()-started, 5)

    def test_nonzero_exit_reports_log(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, 'diagnostic message'):
                common.bounded([sys.executable, '-c', 'print("diagnostic message"); raise SystemExit(7)'], Path(d)/'error.log', 3)

    def test_watchdog_absolute_deadline(self):
        reason, _ = launcher.monitor_health({'active': True, 'started': 0}, {'MemAvailable': 8*common.GIB}, 1201, 1200, 0)
        self.assertIn('wall-time', reason)

    def test_watchdog_sustained_ram_pressure(self):
        count = 0
        for index in range(3):
            reason, count = launcher.monitor_health({'active': False}, {'MemAvailable': common.GIB}, index, 100, count)
        self.assertIn('RAM', reason)
        self.assertEqual(launcher.monitor_health({}, {'MemAvailable': 4*common.GIB}, 0, 100, count), (None, 0))


class ShotTests(unittest.TestCase):
    def inputs(self, guided=False):
        value = settings.shot_settings(guided)
        value['image_refs'] = ['appearance.png']
        if guided:
            value['video_guide'] = 'guide.mp4'
        return value

    def test_ab_has_one_conditioning_difference(self):
        a, b = settings.shot_settings(), settings.shot_settings(True)
        differences = {key for key in a if a[key] != b[key]}
        self.assertEqual(differences, {'video_prompt_type'})
        self.assertEqual(guard.check_shot(settings.MODEL, self.inputs(), False), 'A-baseline')
        self.assertEqual(guard.check_shot(settings.MODEL, self.inputs(True), False), 'B-guided')

    def test_unsupported_settings_rejected(self):
        for change in ({'video_length': 120}, {'resolution': '1920x1080'}, {'seed': -1},
                       {'audio_prompt_type': 'AS'}, {'config': 'distilled_4b'}, {'video_prompt_type': 'IVU'},
                       {'prompt_enhancer': 'TI'}, {'repeat_generation': 2}, {'overrides': 'compile=transformer'},
                       {'multi_prompts_gen_type': 'PW'}, {'prompt': '[/duration 100s] cross'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                guard.check_shot(settings.MODEL, dict(self.inputs(), **change), False)
        with self.assertRaises(ValueError):
            guard.check_shot('minimax_h3_fl2va', self.inputs(), False)

    def test_guide_probe_accepts_exact_metadata(self):
        stream = dict(codec_type='video', width=864, height=480, nb_read_frames='124', avg_frame_rate='24/1', r_frame_rate='24/1')
        result = types.SimpleNamespace(stdout=json.dumps({'streams': [stream]}))
        with patch.object(guard.subprocess, 'run', return_value=result):
            self.assertEqual(guard.probe_guide('guide'), stream)

    def test_guide_probe_rejects_audio_and_wrong_frames(self):
        stream = dict(codec_type='video', width=864, height=480, nb_read_frames='124', avg_frame_rate='24/1', r_frame_rate='24/1')
        for streams in ([dict(stream, nb_read_frames='120')], [stream, dict(codec_type='audio')], [dict(stream, avg_frame_rate='30/1')]):
            result = types.SimpleNamespace(stdout=json.dumps({'streams': streams}))
            with patch.object(guard.subprocess, 'run', return_value=result), self.assertRaises(ValueError):
                guard.probe_guide('guide')


class UploadTests(unittest.TestCase):
    def test_one_native_private_request_and_no_source_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'input').mkdir()
            (root/'input/a').write_bytes(b'a')
            catalog = {'assets': [asset('a', b'a'), asset('b', b'b')]}

            class API:
                config_values = {'username': 'test-owner'}
                calls = []

                def upload_files(self, request, resources, folder, blob_type=None, upload_context=None, quiet=False, dir_mode='skip'):
                    for path in Path(folder).iterdir():
                        self.calls.append((path.name, path.read_bytes()))
                        request.files.append(path.name)

                def dataset_create_new(self, folder, **kwargs):
                    self.options = kwargs
                    self.upload_files(types.SimpleNamespace(files=[]), None, folder)
                    return {'status': 'ok'}

            api = API()
            original = api.upload_files

            def download(record, target):
                Path(target).write_bytes(b'b')
                return Path(target)

            with patch('dataset.transfer', side_effect=download):
                url = create_dataset(catalog, root/'output', root/'input', root/'cache', api=api)
            self.assertIn('/test-owner/jjp-wangp-motion-', url)
            self.assertFalse(api.options['public'])
            self.assertEqual({name for name, _ in api.calls}, {'a', 'b', 'manifest.json'})
            self.assertEqual((root/'input/a').read_bytes(), b'a')
            self.assertFalse((root/'cache/b').exists())
            self.assertEqual(api.upload_files, original)
            self.assertTrue((root/'output/dataset-upload.json').exists())

    def test_upload_failure_never_writes_success_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'input').mkdir()
            (root/'input/a').write_bytes(b'a')

            class API:
                config_values = {'username': 'test-owner'}

                def upload_files(self, request, folder):
                    raise RuntimeError('simulated upload failure')

                def dataset_create_new(self, folder, **kwargs):
                    self.upload_files(types.SimpleNamespace(files=[]), folder)

            with self.assertRaisesRegex(RuntimeError, 'simulated upload failure'):
                create_dataset({'assets': [asset('a', b'a')]}, root/'out', root/'input', root/'cache', api=API())
            self.assertFalse((root/'out/dataset-upload.json').exists())
            self.assertTrue((root/'input/a').exists())


class ArtifactTests(unittest.TestCase):
    def test_catalog_has_complete_pinned_set(self):
        catalog = json.loads((SOURCE/'assets.json').read_text())
        assets = catalog['assets']
        self.assertEqual(len(assets), 13)
        self.assertEqual(len({a['name'] for a in assets}), 13)
        for record in assets:
            self.assertRegex(record['sha256'], '^[0-9a-f]{64}$')
            self.assertIn(catalog['revision'], record['source'])
        self.assertAlmostEqual(sum(a['bytes'] for a in assets)/common.GIB, 40.53187, places=4)

    def test_notebooks_compile_and_are_single_cell(self):
        for name in ['JJP_WanGP_Motion_Test_Kit.ipynb', 'JJP_WanGP_Motion_Models.ipynb']:
            notebook = json.loads((ROOT/name).read_text())
            cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
            self.assertEqual(len(cells), 1)
            compile(''.join(cells[0]['source']), name, 'exec')
            self.assertFalse(cells[0]['outputs'])


if __name__ == '__main__':
    unittest.main()
