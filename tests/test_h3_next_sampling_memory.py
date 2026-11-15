"""Control-flow regressions for the reported full-sequence reference OOM."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

R = Path(__file__).resolve().parents[1]

def module(name):
    spec = importlib.util.spec_from_file_location(name, R/'notebooks/h3_next'/(name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

class SamplingMemory(unittest.TestCase):
    def patched_fixture(self, root):
        patch = module('patch_tp_check')
        path = root/'h3_tensor_parallel.py'
        # Execute the same diagnostic guard that is inserted into pinned block().
        path.write_text('class TP:\n    def block(self, args, original):\n'+patch.OLD+'        return ref\n')
        patch.patch_tp_check(root)
        once = path.read_text()
        patch.patch_tp_check(root)
        self.assertEqual(once, path.read_text())
        scope = {'logger': Mock()}
        exec(compile(once, str(path), 'exec'), scope)
        return scope['TP'], scope['logger']

    def test_video_sequence_never_clones_or_calls_unsplit_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            cls, logger = self.patched_fixture(Path(tmp))
            tp = cls(); tp.checked = False
            image = Mock(); image.shape = (60000, 3072)
            original = Mock(side_effect=AssertionError('full reference allocation'))
            self.assertIsNone(tp.block({'img': image}, original))
            original.assert_not_called(); image.clone.assert_not_called()
            self.assertTrue(tp.checked); logger.info.assert_called_once()

    def test_small_comparison_is_retained_at_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cls, _ = self.patched_fixture(Path(tmp))
            tp = cls(); tp.checked = False
            image = Mock(); image.shape = (4096, 3072)
            expected = Mock(); original = Mock(return_value={'img': expected})
            self.assertIs(tp.block({'img': image}, original), expected.float.return_value)
            original.assert_called_once(); image.clone.assert_called_once()

    def test_unknown_upstream_source_is_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'h3_tensor_parallel.py'; path.write_text('changed source')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                module('patch_tp_check').patch_tp_check(tmp)
            self.assertEqual(path.read_text(), 'changed source')

    def test_root_memory_error_survives_long_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'controller.log'
            path.write_text('RuntimeError: SamplerCustomAdvanced (12): torch.OutOfMemoryError: Allocation on device 0\nRequested: 832.62 MiB\n'+'stack frame\n'*2500)
            result = module('error_summary').next_failure_summary(path)
            self.assertIn('torch.OutOfMemoryError', result)
            self.assertIn('832.62 MiB', result)
            self.assertLess(len(result), 12500)

    def test_generated_runtime_has_fixes_and_no_pdd_registration(self):
        import json
        nb = json.loads((R/'H3_Next_T4.ipynb').read_text())
        source = ''.join(nb['cells'][1]['source'])
        def values(text):
            result = {}
            for node in ast.parse(text).body:
                if isinstance(node, ast.Assign):
                    for name in node.targets:
                        if isinstance(name, ast.Name):
                            try: result[name.id] = ast.literal_eval(node.value)
                            except (ValueError, TypeError): pass
            return result
        stages = values(source)['_H3_PIPELINE_SOURCES']
        install = stages['install']; plugin = values(install)['plugin_source']
        self.assertIn('patch_tp_check(MGPU)', install)
        self.assertIn('_h3_molmo_int8_mm(q, self.qweight)', install)
        self.assertIn('production-shape INT8 accuracy probe', install)
        self.assertIn('suite_stage_attached', stages['catalog'])
        self.assertIn('next_failure_summary(log_path)', plugin)
        self.assertIn("NODE_CLASS_MAPPINGS.pop('H3SuitePDDApply', None)", plugin)

if __name__ == '__main__':
    unittest.main()
