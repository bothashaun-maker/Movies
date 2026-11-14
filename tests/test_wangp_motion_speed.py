"""Optional speed setup/fallback and scoped attention; no simulated GPU arithmetic."""
from pathlib import Path
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/'notebooks/wan_gp_motion'
sys.path.insert(0, str(SOURCE))
import h3_attention
import launcher


class AttentionScopeTests(unittest.TestCase):
    def test_prior_mode_and_value_scaling_restored_after_success_and_error(self):
        for fail in (False, True):
            original = {'_attention': 'sdpa', '_sage_v_scaling': False, 'unrelated': 7}
            state = dict(original)
            try:
                with h3_attention.attention_scope(state, 'sage'):
                    self.assertEqual(state['_attention'], 'sage')
                    self.assertTrue(state['_sage_v_scaling'])
                    if fail:
                        raise RuntimeError('native forward failed')
            except RuntimeError:
                pass
            self.assertEqual(state, original)

    def test_absent_state_keys_are_removed_and_nested_scope_restored(self):
        state = {}
        with h3_attention.attention_scope(state, 'sage'):
            with h3_attention.attention_scope(state, 'sdpa'):
                self.assertEqual(state['_attention'], 'sdpa')
            self.assertEqual(state['_attention'], 'sage')
        self.assertEqual(state, {})

    def test_h3_forward_scopes_backend_and_restores_qwen_vae_default(self):
        state = {'_attention': 'sdpa', '_sage_v_scaling': False}
        mmgp = types.ModuleType('mmgp')
        mmgp.offload = types.SimpleNamespace(shared_state=state)

        @h3_attention.h3_only
        def forward(value, fail=False):
            self.assertEqual(state['_attention'], 'sage')
            self.assertTrue(state['_sage_v_scaling'])
            if fail:
                raise ValueError('model error')
            return value

        with patch.dict(sys.modules, {'mmgp': mmgp}), patch.dict(os.environ, {'JJP_H3_ATTENTION': 'sage'}):
            self.assertEqual(forward('native result'), 'native result')
            with self.assertRaisesRegex(ValueError, 'model error'):
                forward(None, fail=True)
        self.assertEqual(state, {'_attention': 'sdpa', '_sage_v_scaling': False})


class OptionalAccelerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def choose(self, mode='auto'):
        return h3_attention.choose(SOURCE, self.output, 'python', 'uv', 'native-app', {'sentinel': 'kept'}, mode)

    def test_sdpa_never_installs_or_probes(self):
        with patch.object(h3_attention, 'bounded', side_effect=AssertionError('No optional setup in SDPA mode')):
            self.assertEqual(self.choose('sdpa'), 'sdpa')

    def test_acceleration_requires_successful_bounded_probe(self):
        calls = []

        def bounded(command, log, timeout, **kwargs):
            calls.append((command, timeout, kwargs))
            if str(command[1]).endswith('h3_attention.py'):
                (self.output/'attention-probe.json').write_text(json.dumps({'selected': 'sage'}))

        with patch.object(h3_attention, 'bounded', side_effect=bounded):
            self.assertEqual(self.choose(), 'sage')
        self.assertEqual([c[1] for c in calls], [90, 120])
        self.assertIn('--no-deps', calls[0][0])
        self.assertIn('--require-hashes', calls[0][0])
        self.assertEqual(calls[1][2]['cwd'], 'native-app')

    def test_install_and_probe_failures_override_stale_success(self):
        for fail_at, error in ((1, RuntimeError('install failed')), (2, TimeoutError('probe deadline'))):
            (self.output/'attention-probe.json').write_text(json.dumps({'selected': 'sage'}))
            count = 0

            def bounded(*args, **kwargs):
                nonlocal count
                count += 1
                if count == fail_at:
                    raise error

            with patch.object(h3_attention, 'bounded', side_effect=bounded):
                self.assertEqual(self.choose(), 'sdpa')
            receipt = json.loads((self.output/'attention-probe.json').read_text())
            self.assertEqual(receipt['selected'], 'sdpa')
            self.assertIn(str(error), receipt['reason'])

    def test_no_success_receipt_and_malformed_receipt_fall_back(self):
        for contents in (None, 'not json'):
            def bounded(command, *args, **kwargs):
                if contents and str(command[1]).endswith('h3_attention.py'):
                    (self.output/'attention-probe.json').write_text(contents)
            with patch.object(h3_attention, 'bounded', side_effect=bounded):
                self.assertEqual(self.choose(), 'sdpa')

    def test_cancel_propagates_and_does_not_continue_setup(self):
        with patch.object(h3_attention, 'bounded', side_effect=KeyboardInterrupt) as stage:
            with self.assertRaises(KeyboardInterrupt):
                self.choose()
            stage.assert_called_once()

    def test_invalid_mode_rejected_before_setup(self):
        with patch.object(h3_attention, 'bounded', side_effect=AssertionError('Invalid mode must not start setup')):
            with self.assertRaisesRegex(ValueError, 'SPEED_MODE'):
                self.choose('fastest')

    def test_known_bad_speed_stack_stops_before_hardware_or_any_setup(self):
        with patch.object(launcher, 'hardware_preflight', side_effect=AssertionError('Do not start GPU setup')), \
             patch.object(launcher, 'bounded', side_effect=AssertionError('Do not stage/install anything')), \
             patch.object(h3_attention, 'bounded', side_effect=AssertionError('Do not repeat the GPU probe')):
            with self.assertRaisesRegex(RuntimeError, 'reproduced offline'):
                launcher.launch(SOURCE, self.output, speed_mode='auto')
        self.assertEqual(list(self.output.iterdir()), [])

    def test_explicit_sdpa_passes_the_speed_gate(self):
        self.assertIsNone(h3_attention.speed_preflight('sdpa'))


if __name__ == '__main__':
    unittest.main()
