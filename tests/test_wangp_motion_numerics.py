"""Overflow regressions. Uses CPU Torch and, optionally, the pinned native app."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/'notebooks/wan_gp_motion'
sys.path.insert(0, str(SOURCE))
HAS_TORCH = importlib.util.find_spec('torch') is not None
APP = os.environ.get('JJP_WANGP_TEST_APP', '')


@unittest.skipUnless(HAS_TORCH, 'Requires CPU video test dependencies')
class OverflowTests(unittest.TestCase):
    def setUp(self):
        import torch
        import h3_safe_fp16 as safe
        self.torch, self.safe = torch, safe

    def test_projection_reproduces_raw_overflow_and_preserves_fp32_result(self):
        self.safe.self_test('cpu')

    def test_swiglu_and_down_projection_do_not_overflow_or_downcast_residual(self):
        torch = self.torch
        fc1, fc2 = torch.nn.Linear(4, 8, bias=False).half(), torch.nn.Linear(4, 4, bias=False).half()
        with torch.no_grad():
            fc1.weight.fill_(64)
            fc2.weight.fill_(8)
            x = torch.ones(2, 4, dtype=torch.float16)
            raw = fc1(x)
            gate, up = raw.chunk(2, -1)
            self.assertFalse(bool(torch.isfinite(torch.nn.functional.silu(gate)*up).all()))
            residual = torch.full((2, 4), 100000.0)
            scales = torch.tensor([[0.5]*4])
            result = self.safe.safe_mlp(types.SimpleNamespace(fc1=fc1, fc2=fc2), [x],
                                        residual, scales, [(0, 2, 0)])
            expected = 100000.0 + (256.0*256.0)*4*8*0.5
            self.assertIs(result, residual)
            self.assertTrue(bool(torch.isfinite(result).all()))
            torch.testing.assert_close(result, torch.full_like(result, expected))

    def test_scaled_projection_keeps_segment_gates_and_bias(self):
        torch = self.torch
        layer = torch.nn.Linear(4, 4, bias=True).half()
        with torch.no_grad():
            layer.weight.fill_(128)
            layer.bias.fill_(3)
            x = torch.full((4, 4), 256.0, dtype=torch.float16)
            gates = torch.tensor([[0.5]*4, [-0.25]*4])
            residual = torch.full((4, 4), 100000.0)
            result = self.safe.project_add(layer, x, residual, gates, [(0, 2, 0), (2, 4, 1)])
            expected = torch.nn.functional.linear(x.float(), layer.weight.float(), layer.bias.float())
            expected[:2].mul_(0.5).add_(100000.0)
            expected[2:].mul_(-0.25).add_(100000.0)
            # Bias is added to the FP16 projection before its FP32 recovery; allow
            # its ordinary half rounding, not saturation or a multiplied bias.
            torch.testing.assert_close(result, expected, rtol=0.001, atol=2)

    def test_nonfinite_stage_stops_and_writes_diagnostic(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'JJP_OUTPUT': d}):
            for bad in (float('nan'), float('inf'), -float('inf')):
                with self.assertRaisesRegex(FloatingPointError, 'denoising step 1'):
                    self.safe.require_finite(self.torch.tensor([bad]), 'denoising step 1')
                self.assertEqual(json.loads((Path(d)/'numerical-error.json').read_text())['stage'], 'denoising step 1')


@unittest.skipUnless(HAS_TORCH and APP, 'Set JJP_WANGP_TEST_APP to the pinned disposable native checkout')
class NativeOverflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, APP)
        from patch import patch_h3_numerics
        patch_h3_numerics(APP)
        import torch
        assert not torch.cuda.is_available(), 'These numerical regressions intentionally run on CPU'
        # Native shared.attention reads hardware metadata at import. Arithmetic
        # below remains real CPU FP16/FP32; no CUDA tensor or GPU model is used.
        metadata = patch.object(torch.cuda, 'get_device_capability', return_value=(7, 5))
        metadata.start()
        cls.addClassCleanup(metadata.stop)
        current_device = patch.object(torch.cuda, 'current_device', return_value=0)
        current_device.start()
        cls.addClassCleanup(current_device.stop)
        from models.minimax_h3.transformer import MiniMaxH3Model
        from mmgp import offload
        attention = patch.dict(offload.shared_state, {'_attention': 'sdpa'})
        attention.start()
        cls.addClassCleanup(attention.stop)
        cls.torch, cls.Model = torch, MiniMaxH3Model

    def model(self, dtype):
        return self.Model(hidden_size=8, num_layers=2, token_refiner_num_layers=1,
                          num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=16,
                          text_dim=4, timestep_input_dim=8, time_embed_hidden_size=8,
                          time_embed_dim=8, rope_inv_freq_len=1, dtype=dtype)

    def test_native_condition_projection_and_residual_cross_fp16_range(self):
        torch = self.torch
        model = self.model(torch.float16)
        with torch.no_grad():
            model.condition_proj.weight.fill_(96)
            model.condition_proj.bias.zero_()
            text = torch.full((1, 3, 4), 250.0, dtype=torch.float16)
            projected = model.condition_proj(text[0].float())
            self.assertEqual(float(projected.max()), 96000.0)
            self.assertEqual(projected.dtype, torch.float32)
            self.assertFalse(bool(torch.isfinite(projected.half()).all()))
            refined = model.preprocess_text_embeds(text)
            self.assertTrue(bool(torch.isfinite(refined).all()))
            hidden = torch.full((3, 8), 100000.0)
            temb = torch.ones(3, 8)
            for block in model.blocks:
                hidden = block([hidden], temb, [(0, 3, 0)], rope=None)
                self.assertEqual(hidden.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(hidden).all()))
            self.assertGreater(float(hidden.abs().max()), 65504)
            self.assertEqual(model.blocks[0].attn.out_proj.weight.dtype, torch.float16)
            self.assertEqual(model.blocks[0].mlp.fc2.weight.dtype, torch.float16)
            self.assertEqual(model.condition_proj._lock_dtype, torch.float32)

    def test_bfloat16_models_keep_the_native_path(self):
        model = self.model(self.torch.bfloat16)
        self.assertFalse(model._jjp_safe_fp16)
        self.assertEqual(model.condition_proj.weight.dtype, self.torch.bfloat16)
        self.assertFalse(hasattr(model.condition_proj, '_lock_dtype'))

    def test_native_complete_transformer_packs_fp32_references_and_runs_pdd_heads(self):
        torch = self.torch
        model = self.Model(hidden_size=8, num_layers=2, token_refiner_num_layers=1,
                           num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=16,
                           text_dim=4, timestep_input_dim=8, time_embed_hidden_size=8,
                           time_embed_dim=8, rope_inv_freq_len=1, pdd_num_steps=8,
                           pdd_block_size=1, dtype=torch.float16)
        with torch.no_grad():
            for head in (model.final_layer.video_out, model.final_layer.audio_out):
                head.heads.zero_()
                head.head_biases.zero_()
            model.condition_proj.weight.fill_(96)
            model.condition_proj.bias.zero_()
            context = torch.full((1, 3, 4), 250.0, dtype=torch.float16)
            for kind, frames in (('image', 1), ('video', 2)):
                video, audio = torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 4)
                payload = {'text_token_tags': torch.ones(3, dtype=torch.long), 'fps': 24,
                           'frame_count': 5, 'attention_sparsity': 0.0,
                           'refs': [{'kind': kind, 'latent_t': frames, 'latent_h': 4, 'latent_w': 4}],
                           'cond_video_rows': torch.randn(frames*4, 96)}
                result_video, result_audio = model(video, audio, torch.tensor([1.0]), torch.tensor([1.0]), context, payload)
                self.assertEqual(result_video.shape, video.shape)
                self.assertEqual(result_audio.shape, audio.shape)
                self.assertTrue(bool(torch.isfinite(result_video).all()))
                self.assertTrue(bool(torch.isfinite(result_audio).all()))

    def test_mm_gp_loader_retains_fp32_islands_and_fp16_projections(self):
        import safetensors.torch
        from mmgp import offload
        torch = self.torch
        with tempfile.TemporaryDirectory() as d:
            model = self.model(torch.float16)
            weights = Path(d)/'tiny.safetensors'
            safetensors.torch.save_file(model.state_dict(), str(weights))
            restored = self.model(torch.float16)
            offload.load_model_data(restored, str(weights), default_dtype=torch.float16, writable_tensors=False)
            self.assertEqual(restored.condition_proj.weight.dtype, torch.float32)
            self.assertEqual(restored.blocks[0].norm1.weight.dtype, torch.float32)
            self.assertEqual(restored.blocks[0].attn.out_proj.weight.dtype, torch.float16)
            self.assertEqual(restored.blocks[0].mlp.fc2.weight.dtype, torch.float16)


if __name__ == '__main__':
    unittest.main()
